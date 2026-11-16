import os, sqlite3, threading, time, math, re
from datetime import datetime, timedelta, timezone
from contextlib import closing
from pathlib import Path
from zoneinfo import ZoneInfo
import requests
import pandas as pd
try:
    import feedparser
except ImportError:
    feedparser = None  # optional when running isolated unit tests
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
    """Fetch paginated MOEX ISS daily candles, not just the first page."""
    if not re.fullmatch(r"[A-Z0-9_.-]{1,24}", ticker.upper()):
        raise ValueError("Некорректный тикер")
    board = board or BOARD
    if board not in MARKET_BOARDS:
        raise ValueError("Некорректный режим торгов")
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities/{ticker.upper()}/candles.json"
    rows, cols, offset = [], [], 0
    while True:
        data = iss_get(url, {"from": start_date, "interval": 24, "iss.meta": "off", "iss.only": "candles", "start": offset})
        block = data.get("candles", {})
        page = block.get("data", [])
        cols = block.get("columns", cols)
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        if offset > 12000:
            break
        time.sleep(REQUEST_PAUSE)
    if not rows:
        raise ValueError("MOEX ISS не вернул свечи для тикера/режима")
    df = pd.DataFrame(rows, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("begin").drop_duplicates("begin")
    # MOEX may publish an INCOMPLETE daily candle for today's trading.
    # Conservatively exclude it from all scanners and backtests.
    today_moscow = datetime.now(ZoneInfo("Europe/Moscow")).date()
    df = df[df["begin"].dt.date < today_moscow].reset_index(drop=True)
    if df.empty:
        raise ValueError("MOEX ещё не вернул завершённых дневных свечей")
    return df

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


def simulate(df: pd.DataFrame, years: int, fee_bps: float, slippage_bps: float) -> dict:
    """Educational long-only, no overlap, next-day open and 10-bar close.
    Signal uses prior close only. Equity includes flat periods and mark-to-market
    during positions. No corporate-action adjustments or portfolio constraints.
    """
    if len(df) < 220:
        raise ValueError("Недостаточно истории для EMA200 и тестирования")
    x = df.copy()
    c = x["close"]
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    x["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    x["rsi"] = 100 - 100 / (1 + rs)
    x["vol_avg20"] = x["volume"].rolling(20).mean()
    scores = ((c > x["ema200"]).astype(int) +
              (x["ema50"] > x["ema200"]).astype(int) +
              (x["ema20"] > x["ema50"]).astype(int) +
              (x["rsi"].between(45,65)).astype(int) +
              ((c >= x["open"]) & (x["volume"] > x["vol_avg20"])).astype(int))
    signals = (scores >= 4) & (c > x["ema200"])
    end = x["begin"].max()
    cutoff = end - pd.DateOffset(years=years)
    start_idx = max(210, int(x["begin"].searchsorted(cutoff)))
    cost = (fee_bps + slippage_bps) / 10000
    trades = []
    equity, curve = 1.0, []
    i = start_idx
    while i < len(x):
        if i + 11 < len(x) and bool(signals.iloc[i-1]):
            entry_i, exit_i = i, i+10
            entry, exitp = float(x.iloc[entry_i]["open"]), float(x.iloc[exit_i]["close"])
            if entry > 0 and exitp > 0 and math.isfinite(entry) and math.isfinite(exitp):
                gross = exitp / entry
                factor = gross * (1-cost)/(1+cost)
                trades.append({"entry_date":str(x.iloc[entry_i]["begin"].date()),
                               "exit_date":str(x.iloc[exit_i]["begin"].date()),
                               "return_pct":round((factor-1)*100,3)})
                initial_equity = equity
                for j in range(entry_i,exit_i+1):
                    mark = float(x.iloc[j]["close"])
                    marked = initial_equity * (mark / (entry*(1+cost)))
                    if j == exit_i: marked = initial_equity*factor
                    curve.append({"date":str(x.iloc[j]["begin"].date()),"equity":round(marked,6)})
                equity *= factor
                i = exit_i+1
                continue
        curve.append({"date":str(x.iloc[i]["begin"].date()),"equity":round(equity,6)})
        i += 1
    returns = [t["return_pct"]/100 for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    gp = sum(wins); gl = -sum(losses)
    pf = round(gp/gl,3) if gl > 0 else None
    peak, dd = 1.0, 0.0
    for point in curve:
        peak = max(peak,point["equity"])
        dd = max(dd, (peak-point["equity"])/peak)
    return {"count":len(trades),"win_rate_pct":round(100*len(wins)/len(trades),2) if trades else None,
            "profit_factor":pf,"max_drawdown_pct":round(dd*100,2),
            "total_return_pct":round((equity-1)*100,2),
            "avg_return_pct":round(sum(returns)/len(returns)*100,2) if trades else None,
            "trades":trades,"equity_curve":curve,
            "first_date":str(x.iloc[start_idx]["begin"].date()),
            "last_date":str(end.date()),
            "method":"Сигнал по закрытию предыдущего дня; вход на следующем открытии, выход на закрытии через 10 свечей; одна позиция одновременно; фиксированная доля капитала 100%; комиссии и проскальзывание на обе стороны. Без дивидендов, корпоративных корректировок и налогов. Учебная модель, не исполнение на бирже."}


def validate_test(years, fee_bps, slippage_bps):
    if years not in (1,3,5):
        raise HTTPException(400, "Период: 1, 3 или 5 лет")
    if not (0 <= fee_bps <= 100 and 0 <= slippage_bps <= 100):
        raise HTTPException(400, "Комиссия и проскальзывание: 0–100 б.п.")


@app.get("/api/backtest/{ticker}")
def backtest(ticker: str, board: str = "TQBR", years: int = 3, fee_bps: float = 10, slippage_bps: float = 10):
    validate_test(years,fee_bps,slippage_bps)
    if board not in MARKET_BOARDS or not re.fullmatch(r"[A-Z0-9_.-]{1,24}", ticker.upper()):
        raise HTTPException(400,"Некорректный тикер или режим торгов")
    try:
        df = get_candles(ticker.upper(), days=years*366+550, board=board)
        result = simulate(df,years,fee_bps,slippage_bps)
        return {"ticker":ticker.upper(),"board":board,"years":years,**result}
    except Exception as e:
        raise HTTPException(400,detail=str(e))

market_test_lock = threading.Lock()
market_test = {"status":"idle","total":0,"done":0,"errors":0,"results":[],"message":""}


def run_market_backtest(years, fee_bps, slippage_bps):
    try:
        items = get_catalog()
        with market_test_lock:
            market_test.update(status="running",total=len(items),done=0,errors=0,results=[],message="Тестирование акций")
        for item in items:
            try:
                df = get_candles(item["ticker"],days=years*366+550,board=item["board"])
                result = simulate(df,years,fee_bps,slippage_bps)
                summary = {k:result[k] for k in ("count","win_rate_pct","profit_factor","max_drawdown_pct","total_return_pct","first_date","last_date")}
                summary.update(item)
                with market_test_lock: market_test["results"].append(summary)
            except Exception:
                with market_test_lock: market_test["errors"] += 1
            finally:
                with market_test_lock: market_test["done"] += 1
                time.sleep(REQUEST_PAUSE)
        with market_test_lock:
            market_test.update(status="finished",message="Тестирование завершено")
    except Exception as e:
        with market_test_lock: market_test.update(status="error",message=str(e))


@app.post("/api/market-backtest/start")
def start_market_backtest(years: int = 3, fee_bps: float = 10, slippage_bps: float = 10):
    validate_test(years,fee_bps,slippage_bps)
    with market_test_lock:
        if market_test["status"] in ("loading","running"):
            return {"status":"busy"}
        market_test.update(status="loading",total=0,done=0,errors=0,results=[],message="Получение каталога")
    threading.Thread(target=run_market_backtest,args=(years,fee_bps,slippage_bps),daemon=True).start()
    return {"status":"started"}


@app.get("/api/market-backtest/status")
def market_backtest_status():
    with market_test_lock:
        output = dict(market_test)
        output["results"] = sorted(market_test["results"],key=lambda r:(-r["count"],r["ticker"]))
    return output


@app.get("/api/catalog")
def catalog():
    items = get_catalog()
    return {"count": len(items), "items": items}


def news_items():
    items = []
    if feedparser is None:
        return items
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


# v8: one report combining proven technical indicators, completed timeframes,
# transparent risk scenarios and a strictly chronological diagnostic ML holdout.
# No broker API credentials, automatic orders, or LLM shell execution.
COMMON_NAMES = {"сбер": "SBER", "сбербанк": "SBER", "газпром": "GAZP",
                "лукойл": "LKOH", "роснефть": "ROSN", "яндекс": "YDEX"}

@app.get("/api/search")
def search_companies(q: str = ""):
    query = q.strip().casefold()
    if not query or len(query) > 80:
        raise HTTPException(400, "Введите название компании или тикер до 80 символов")
    known = COMMON_NAMES.get(query)
    if known:
        return {"items": [{"ticker": known, "board": "TQBR", "name": q.strip()}]}
    if re.fullmatch(r"[A-Z0-9_.-]{1,24}", q.upper()):
        return {"items": [{"ticker": q.upper(), "board": "TQBR", "name": q.upper()}]}
    try:
        catalog = get_catalog()
        items = [x for x in catalog if query in str(x['name']).casefold() or query in x['ticker'].casefold()]
        return {"items": items[:30]}
    except Exception as e:
        raise HTTPException(503, f"Не удалось получить список компаний MOEX: {e}")

@app.get("/api/research/{ticker}")
def research(ticker: str, board: str = "TQBR", risk_capital: float = 100000,
             risk_pct: float = 1.0):
    if board not in MARKET_BOARDS or not re.fullmatch(r"[A-Z0-9_.-]{1,24}", ticker.upper()):
        raise HTTPException(400, "Некорректный тикер или режим торгов")
    if not math.isfinite(risk_capital) or not 1000 <= risk_capital <= 100000000:
        raise HTTPException(400, "Учебный капитал должен быть от 1 000 до 100 000 000 ₽")
    if not math.isfinite(risk_pct) or not 0.1 <= risk_pct <= 3:
        raise HTTPException(400, "Риск должен быть от 0,1% до 3%")
    try:
        from research_desk import analyze_research, model_holdout, completed_daily
        raw = get_candles(ticker.upper(), days=3050, board=board)
        data = completed_daily(raw)
        result = analyze_research(data, risk_capital=risk_capital, risk_pct=risk_pct)
        try:
            model = model_holdout(data)
        except (ValueError, ArithmeticError) as e:
            model = {"status": "unavailable", "message": str(e)}
        # News are available only when real external RSS feeds are configured.
        headlines = []
        names = [ticker.upper().casefold()] + [k for k,v in COMMON_NAMES.items() if v==ticker.upper()]
        for item in news_items():
            title = str(item['title']).casefold()
            if any(re.search(r'(?<![\wа-яё])'+re.escape(name)+r'(?![\wа-яё])',title,re.I) for name in names):
                headlines.append(item)
        return {"ticker": ticker.upper(), "board": board, "research": result,
                "model": model, "news": headlines[:8], "rss_configured": bool(RSS_FEEDS),
                "candles_used": len(data), "source": "MOEX ISS (завершённые дневные свечи)",
                "not_realtime": True}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except (requests.RequestException, KeyError, IndexError) as e:
        raise HTTPException(503, detail=f"Ошибка данных MOEX: {e}")

@app.get("/api/strategy-lab/{ticker}")
def strategy_lab(ticker: str, board: str = "TQBR", years: int = 3, fee_bps: float = 10, slippage_bps: float = 10):
    from strategy_lab import backtest
    from research_desk import completed_daily
    validate_test(years, fee_bps, slippage_bps)
    if board not in MARKET_BOARDS or not re.fullmatch(r"[A-Z0-9_.-]{1,24}", ticker.upper()):
        raise HTTPException(400, "Некорректный тикер или режим торгов")
    try:
        df = completed_daily(get_candles(ticker.upper(), days=max(3000,years*366+1100), board=board))
        cutoff=df.begin.max()-pd.DateOffset(years=years)
        # 70/30 chronological split of the requested evaluation period; indicator
        # warmup is kept from earlier data, but trades start inside the period.
        begin_ix=max(215,int(df.begin.searchsorted(cutoff)))
        if len(df)-begin_ix<100:
            raise ValueError('Недостаточно завершённых свечей для честного сравнения')
        split_ix=begin_ix+int((len(df)-begin_ix)*0.7)
        modes=("adaptive", "adaptive_mtf", "momentum", "breakout")
        rows=[]
        for mode in modes:
            train=backtest(df.iloc[:split_ix].copy(),years,fee_bps,slippage_bps,mode,start_index=begin_ix)
            holdout=backtest(df,years,fee_bps,slippage_bps,mode,start_index=split_ix)
            rows.append({"strategy":mode, "train": {k:train[k] for k in ("count","win_rate_pct","profit_factor","max_drawdown_pct","total_return_pct")},
                        "test": {k:holdout[k] for k in ("count","win_rate_pct","profit_factor","max_drawdown_pct","total_return_pct")}})
        return {"ticker":ticker.upper(),"board":board,"years":years,
                "train_from":str(df.begin.iloc[begin_ix].date()),"test_from":str(df.begin.iloc[split_ix].date()),
                "test_to":str(df.begin.iloc[-1].date()),"results":rows,
                "note":"Фиксированные правила без подбора по тесту. Обучающий/контрольный период 70/30 по времени; модель держит до одной позиции в акции, капитал реинвестируется, комиссия и проскальзывание на входе и выходе. Без дивидендов, делистингов, налогов и корпоративных корректировок. Мало сделок = статистически ненадёжно."}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(503, detail=f"Не удалось проверить стратегии: {e}")
