import os, sqlite3, threading, time, math, re
from datetime import datetime, timedelta, timezone
from contextlib import closing
from pathlib import Path
import requests
import pandas as pd
import feedparser
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

load_dotenv()
APP_TITLE = "MOEX AI Trading Assistant"
DB_PATH = os.getenv("DB_PATH", "assistant.db")
TICKERS = [x.strip().upper() for x in os.getenv("TICKERS", "SBER,GAZP,LKOH,ROSN,YDEX").split(",") if x.strip()]
BOARD = os.getenv("MOEX_BOARD", "TQBR")
MARKET_BOARDS = [x.strip().upper() for x in os.getenv("MARKET_BOARDS", "TQBR,TQTF,TQPI,TQDE,TQIR").split(",") if x.strip()]
REQUEST_PAUSE = max(.15, float(os.getenv("REQUEST_PAUSE_SECONDS", "0.35")))
SCAN_INTERVAL = max(0, int(os.getenv("SCAN_INTERVAL_MINUTES", "0")))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RSS_FEEDS = [x.strip() for x in os.getenv("RSS_FEEDS", "").split(",") if x.strip()]

app = FastAPI(title=APP_TITLE)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
scan_lock = threading.Lock()
scheduler_started = False
job_lock = threading.Lock()
job = {"status": "idle", "total": 0, "done": 0, "results": [], "errors": 0, "message": ""}
CATALOG_TTL = 6 * 3600
catalog_cache = {"at": 0, "items": []}

def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with closing(db()) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT, ticker TEXT,
            signal TEXT, score INTEGER, close REAL, ema20 REAL, ema50 REAL, ema200 REAL,
            rsi REAL, atr REAL, entry REAL, stop REAL, target REAL, reason TEXT)""")
        con.commit()

def iss_get(url, params):
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=22, headers={"User-Agent": "MOEX-Research-Scanner/2.0"})
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"ISS temporary error {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"Ошибка MOEX ISS: {last_error}")


def get_catalog(force=False):
    """Live MOEX share-market listing, paginated; prefer selected trading boards."""
    if not force and catalog_cache["items"] and time.time() - catalog_cache["at"] < CATALOG_TTL:
        return catalog_cache["items"]
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/securities.json"
    chosen = {}
    start = 0
    while True:
        block = iss_get(url, {"iss.meta": "off", "iss.only": "securities", "start": start})["securities"]
        cols = block["columns"]
        rows = block["data"]
        if not rows:
            break
        for row in rows:
            r = dict(zip(cols, row))
            ticker = str(r.get("SECID") or "").upper()
            board = str(r.get("BOARDID") or "").upper()
            if not re.fullmatch(r"[A-Z0-9_.-]{1,24}", ticker) or board not in MARKET_BOARDS:
                continue
            # The shares market includes some non-common-share instruments; retain
            # only equity-like listings when instrument type is supplied.
            item = {"ticker": ticker, "board": board, "name": r.get("SHORTNAME") or ticker}
            if ticker not in chosen or MARKET_BOARDS.index(board) < MARKET_BOARDS.index(chosen[ticker]["board"]):
                chosen[ticker] = item
        start += len(rows)
        if len(rows) < 50 or start > 10000:
            break
        time.sleep(REQUEST_PAUSE)
    items = sorted(chosen.values(), key=lambda x: x["ticker"])
    if not items:
        raise ValueError("MOEX ISS не вернул акции для выбранных режимов торгов")
    catalog_cache.update(at=time.time(), items=items)
    return items


def get_candles(ticker: str, days: int = 900, board: str = None) -> pd.DataFrame:
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board or BOARD}/securities/{ticker}/candles.json"
    params = {"from": start, "interval": 24, "iss.meta": "off", "iss.only": "candles"}
    block = iss_get(url, params).get("candles", {})
    rows = block.get("data", [])
    cols = block.get("columns", [])
    if not rows:
        raise ValueError("MOEX ISS не вернул дневные свечи. Проверьте тикер, режим торгов и доступность данных.")
    df = pd.DataFrame(rows, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["begin"] = pd.to_datetime(df["begin"])
    return df.dropna(subset=["close"]).sort_values("begin").drop_duplicates("begin").reset_index(drop=True)

def calculate(df: pd.DataFrame) -> dict:
    if len(df) < 210:
        raise ValueError(f"Недостаточно истории: {len(df)} дневных свечей; нужно не менее 210.")
    x = df.copy()
    c = x["close"]
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    x["ema200"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    x["rsi"] = 100 - 100 / (1 + rs)
    prev = c.shift(1)
    tr = pd.concat([x["high"] - x["low"], (x["high"] - prev).abs(), (x["low"] - prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["vol_avg20"] = x["volume"].rolling(20).mean()
    last = x.iloc[-1]
    score, reasons = 0, []
    checks = [
        (last.close > last.ema200, "цена выше EMA200"),
        (last.ema50 > last.ema200, "EMA50 выше EMA200"),
        (last.ema20 > last.ema50, "EMA20 выше EMA50"),
        (45 <= last.rsi <= 65, "RSI в диапазоне 45–65"),
        (last.close >= last.open and last.volume > last.vol_avg20, "положительная свеча на объёме выше среднего"),
    ]
    for ok, label in checks:
        if bool(ok):
            score += 1
            reasons.append(label)
    signal = "НАБЛЮДАТЬ"
    entry = stop = target = None
    if score >= 4 and last.close > last.ema200 and pd.notna(last.atr) and last.atr > 0:
        signal = "ВОЗМОЖНЫЙ БЫЧИЙ СЕТАП"
        entry = float(last.close)
        stop = entry - 1.5 * float(last.atr)
        target = entry + 3.0 * float(last.atr)
    return {
        "signal": signal, "score": score, "close": float(last.close),
        "ema20": float(last.ema20), "ema50": float(last.ema50),
        "ema200": float(last.ema200), "rsi": float(last.rsi),
        "atr": float(last.atr), "entry": entry, "stop": stop, "target": target,
        "reason": "; ".join(reasons) or "Фильтры стратегии не подтверждены",
        "avg_turnover_20": float((x["close"] * x["volume"]).tail(20).mean()),
        "date": str(last.begin.date())
    }

def save_signal(ticker: str, a: dict):
    with closing(db()) as con:
        con.execute("""INSERT INTO scans(scanned_at,ticker,signal,score,close,ema20,ema50,ema200,rsi,atr,entry,stop,target,reason)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now(timezone.utc).isoformat(), ticker, a["signal"], a["score"], a["close"],
         a["ema20"], a["ema50"], a["ema200"], a["rsi"], a["atr"], a["entry"], a["stop"], a["target"], a["reason"]))
        con.commit()

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Telegram не настроен"
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        r.raise_for_status()
        return bool(r.json().get("ok")), "Telegram отправил сообщение"
    except Exception as e:
        return False, str(e)

def scan_market(notify=True):
    """Legacy limited watchlist scan (kept for backwards compatibility)."""
    if not scan_lock.acquire(blocking=False):
        return {"status": "busy", "results": []}
    results = []
    try:
        for ticker in TICKERS:
            try:
                analysis = calculate(get_candles(ticker))
                analysis["ticker"] = ticker
                results.append(analysis)
                save_signal(ticker, analysis)
            except Exception as e:
                results.append({"ticker": ticker, "signal": "ОШИБКА ДАННЫХ", "score": 0, "reason": str(e)})
        return {"status": "ok", "scanned_at": datetime.now(timezone.utc).isoformat(), "results": results}
    finally:
        scan_lock.release()


def run_full_scan():
    try:
        items = get_catalog()
        with job_lock:
            job.update(status="running", total=len(items), done=0, results=[], errors=0, message="Анализ акций MOEX")
        for item in items:
            try:
                a = calculate(get_candles(item["ticker"], board=item["board"]))
                a.update(ticker=item["ticker"], board=item["board"], name=item["name"])
                with job_lock:
                    job["results"].append(a)
            except Exception:
                with job_lock:
                    job["errors"] += 1
            finally:
                with job_lock:
                    job["done"] += 1
                time.sleep(REQUEST_PAUSE)
        with job_lock:
            job.update(status="finished", message="Сканирование завершено", finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        with job_lock:
            job.update(status="error", message=str(e))


@app.post("/api/full-scan/start")
def start_full_scan():
    with job_lock:
        if job["status"] in ("loading", "running"):
            return {"status": "busy"}
        job.update(status="loading", total=0, done=0, results=[], errors=0, message="Получение каталога акций MOEX")
    threading.Thread(target=run_full_scan, daemon=True).start()
    return {"status": "started"}


@app.get("/api/full-scan/status")
def full_scan_status():
    with job_lock:
        result = dict(job)
        result["results"] = sorted(job["results"], key=lambda x: (-x["score"], -x["avg_turnover_20"], x["ticker"]))
    return result


@app.get("/api/backtest/{ticker}")
def backtest(ticker: str, board: str = "TQBR", fee_bps: float = 10, slippage_bps: float = 10):
    """Educational long-only next-open, fixed 10-bar holding test; not a full execution simulator."""
    if board not in MARKET_BOARDS or not re.fullmatch(r"[A-Z0-9_.-]{1,24}", ticker.upper()):
        raise HTTPException(status_code=400, detail="Некорректный тикер или режим торгов")
    if not (0 <= fee_bps <= 100 and 0 <= slippage_bps <= 100):
        raise HTTPException(status_code=400, detail="Комиссии должны быть от 0 до 100 б.п.")
    try:
        df = get_candles(ticker.upper(), days=900, board=board)
        trades = []
        i = 210
        cost = (fee_bps + slippage_bps) / 10000
        while i < len(df) - 11:
            historical = df.iloc[:i].copy()
            a = calculate(historical)
            if a["signal"] == "ВОЗМОЖНЫЙ БЫЧИЙ СЕТАП":
                entry = float(df.iloc[i]["open"])
                exit_price = float(df.iloc[i+10]["close"])
                if entry > 0 and exit_price > 0 and math.isfinite(entry) and math.isfinite(exit_price):
                    ret = (exit_price * (1-cost)) / (entry * (1+cost)) - 1
                    trades.append({"entry_date": str(df.iloc[i]["begin"].date()), "exit_date": str(df.iloc[i+10]["begin"].date()), "return_pct": round(ret*100, 2)})
                i += 11
            else:
                i += 1
        wins = sum(t["return_pct"] > 0 for t in trades)
        return {"ticker": ticker.upper(), "board": board, "trades": trades[-100:], "count": len(trades),
                "win_rate_pct": round(100*wins/len(trades), 1) if trades else None,
                "avg_return_pct": round(sum(t["return_pct"] for t in trades)/len(trades), 2) if trades else None,
                "method": "Вход по открытию следующей свечи, выход через 10 торговых свечей; комиссии и проскальзывание на обе стороны. Без дивидендов и корпоративных корректировок."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/catalog")
def catalog():
    items = get_catalog()
    return {"count": len(items), "items": items}


def news_items():
    items = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                items.append({
                    "title": entry.get("title", "Без заголовка"),
                    "link": entry.get("link", ""),
                    "source": feed.feed.get("title", feed_url),
                    "published": entry.get("published", entry.get("updated", ""))
                })
        except Exception:
            continue
    return items[:30]

def scheduler():
    while True:
        try:
            scan_market(notify=True)
        except Exception:
            pass
        time.sleep(SCAN_INTERVAL * 60)

@app.on_event("startup")
def startup():
    global scheduler_started
    init_db()
    if SCAN_INTERVAL > 0 and not scheduler_started:
        threading.Thread(target=scheduler, daemon=True).start()
        scheduler_started = True

@app.get("/", response_class=HTMLResponse)
def home():
    with open(STATIC_DIR / "index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/api/config")
def config():
    return {"tickers": TICKERS, "board": BOARD, "scan_interval_minutes": SCAN_INTERVAL,
            "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID), "full_market_scan": True}

@app.post("/api/scan")
def api_scan():
    return scan_market(notify=False)

@app.get("/api/history")
def history(limit: int = 100):
    with closing(db()) as con:
        rows = con.execute("SELECT scanned_at,ticker,signal,score,close,entry,stop,target,reason FROM scans ORDER BY id DESC LIMIT ?", (min(max(limit,1),500),)).fetchall()
    cols = ["scanned_at","ticker","signal","score","close","entry","stop","target","reason"]
    return [dict(zip(cols, row)) for row in rows]

@app.get("/api/news")
def api_news():
    return news_items()

@app.get("/api/chart/{ticker}")
def chart(ticker: str, board: str = ""):
    try:
        if board and board not in MARKET_BOARDS:
            raise ValueError("Недопустимый режим торгов")
        df = get_candles(ticker.upper(), board=board or BOARD)
        a = calculate(df)
        tail = df.tail(180)
        return {"ticker": ticker.upper(), "analysis": a,
                "candles": [{"date": str(r.begin.date()), "open": float(r.open), "high": float(r.high),
                             "low": float(r.low), "close": float(r.close), "volume": float(r.volume)}
                            for r in tail.itertuples()]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/test-telegram")
def test_telegram():
    ok, msg = send_telegram("MOEX AI Trading Assistant: тестовое сообщение. Уведомления подключены.")
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": msg}
