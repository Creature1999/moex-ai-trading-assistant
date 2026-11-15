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
        if len(page) < 500 or offset > 12000:
            break
        time.sleep(REQUEST_PAUSE)
    if not rows:
        raise ValueError("MOEX ISS не вернул свечи для тикера/режима")
    df = pd.DataFrame(rows, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["begin"] = pd.to_datetime(df["begin"])
    return df.dropna(subset=["open", "high", "low", "close"]).sort_values("begin").drop_duplicates("begin").reset_index(drop=True)

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
    # Additional technical indicators; calculations use only past/current daily candles.
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    # Wilder smoothing for ADX(14).
    high_delta = x["high"].diff()
    low_delta = -x["low"].diff()
    plus_dm = high_delta.where((high_delta > low_delta) & (high_delta > 0), 0.0)
    minus_dm = low_delta.where((low_delta > high_delta) & (low_delta > 0), 0.0)
    atr_wilder = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_wilder.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_wilder.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    # Supertrend(10,3): use previous trailing bands to avoid future leakage.
    atr10 = tr.ewm(alpha=1/10, adjust=False).mean()
    mid = (x["high"] + x["low"]) / 2
    upper = (mid + 3 * atr10).to_numpy()
    lower = (mid - 3 * atr10).to_numpy()
    closes = c.to_numpy()
    st_upper, st_lower = upper.copy(), lower.copy()
    st_dir = [1] * len(x)
    st_line = [float("nan")] * len(x)
    st_line[0] = float(lower[0])
    for i in range(1, len(x)):
        if closes[i-1] <= st_upper[i-1]:
            st_upper[i] = min(upper[i], st_upper[i-1])
        if closes[i-1] >= st_lower[i-1]:
            st_lower[i] = max(lower[i], st_lower[i-1])
        if closes[i] > st_upper[i-1]:
            st_dir[i] = 1
        elif closes[i] < st_lower[i-1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i-1]
        st_line[i] = float(st_lower[i] if st_dir[i] == 1 else st_upper[i])
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
        "date": str(last.begin.date()),
        "supertrend": round(st_line[-1], 4), "supertrend_direction": "РОСТ" if st_dir[-1] == 1 else "СНИЖЕНИЕ",
        "macd": round(float(macd_line.iloc[-1]), 5),
        "macd_signal": round(float(macd_signal.iloc[-1]), 5),
        "macd_histogram": round(float(macd_hist.iloc[-1]), 5),
        "adx": round(float(adx.iloc[-1]), 2) if pd.notna(adx.iloc[-1]) else None,
        "plus_di": round(float(plus_di.iloc[-1]), 2) if pd.notna(plus_di.iloc[-1]) else None,
        "minus_di": round(float(minus_di.iloc[-1]), 2) if pd.notna(minus_di.iloc[-1]) else None,
        "volume": float(last.volume), "volume_avg20": float(last.vol_avg20),
        "volume_ratio": round(float(last.volume / last.vol_avg20), 2) if pd.notna(last.vol_avg20) and last.vol_avg20 > 0 else None,
        "indicator_summary": {
            "ema": "ВОСХОДЯЩИЙ" if last.close > last.ema200 and last.ema20 > last.ema50 > last.ema200 else "НЕ ПОДТВЕРЖДЁН",
            "supertrend": "ВОСХОДЯЩИЙ" if st_dir[-1] == 1 else "НИСХОДЯЩИЙ",
            "macd": "ПОЛОЖИТЕЛЬНЫЙ" if macd_hist.iloc[-1] > 0 else "ОТРИЦАТЕЛЬНЫЙ",
            "adx": "СИЛЬНЫЙ ТРЕНД" if pd.notna(adx.iloc[-1]) and adx.iloc[-1] >= 25 else "СЛАБЫЙ / НЕОПРЕДЕЛЁННЫЙ",
            "rsi": "ВЫШЕ 70" if last.rsi > 70 else ("НИЖЕ 30" if last.rsi < 30 else "30–70"),
            "volume": "ВЫШЕ СРЕДНЕГО" if last.volume > last.vol_avg20 else "НЕ ВЫШЕ СРЕДНЕГО",
        },
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



# Multi-strategy research engine. Signals are computed using information available
# at the close; entries occur on the following day's open.
def strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    c, h, l = x["close"], x["high"], x["low"]
    for n in (20, 50, 200):
        x[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    x["rsi"] = 100 - 100/(1 + up/down.replace(0, float("nan")))
    prev = c.shift()
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["vol_avg20"] = x["volume"].rolling(20).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    x["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    hd, ld = h.diff(), -l.diff()
    plus = hd.where((hd > ld) & (hd > 0), 0.0).ewm(alpha=1/14, adjust=False).mean()
    minus = ld.where((ld > hd) & (ld > 0), 0.0).ewm(alpha=1/14, adjust=False).mean()
    atr = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, float("nan"))
    pdi, mdi = 100*plus/atr, 100*minus/atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, float("nan"))
    x["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    x["plus_di"], x["minus_di"] = pdi, mdi
    atr10 = tr.ewm(alpha=1/10, adjust=False).mean()
    basic_up = ((h+l)/2 + 3*atr10).to_numpy()
    basic_low = ((h+l)/2 - 3*atr10).to_numpy()
    close = c.to_numpy()
    final_up, final_low = basic_up.copy(), basic_low.copy()
    direction = [1]*len(x)
    for i in range(1,len(x)):
        final_up[i] = basic_up[i] if (basic_up[i] < final_up[i-1] or close[i-1] > final_up[i-1]) else final_up[i-1]
        final_low[i] = basic_low[i] if (basic_low[i] > final_low[i-1] or close[i-1] < final_low[i-1]) else final_low[i-1]
        if close[i] > final_up[i-1]: direction[i] = 1
        elif close[i] < final_low[i-1]: direction[i] = -1
        else: direction[i] = direction[i-1]
    x["st_up"] = pd.Series(direction, dtype=int).eq(1)
    x["high20_previous"] = h.shift(1).rolling(20).max()
    return x

STRATEGY_LABELS = {
    "baseline": "Базовая EMA + RSI + Volume",
    "trend": "Trend Following: EMA + Supertrend + ADX",
    "momentum": "Momentum: EMA + MACD + RSI + Volume",
    "breakout": "Breakout: 20-дневный максимум + ADX + Volume",
}

def strategy_signal(x: pd.DataFrame, name: str) -> pd.Series:
    c = x["close"]
    trend = (c > x["ema200"]) & (x["ema50"] > x["ema200"])
    if name == "baseline":
        score = ((c > x["ema200"]).astype(int) + (x["ema50"] > x["ema200"]).astype(int)
                 + (x["ema20"] > x["ema50"]).astype(int) + x["rsi"].between(45,65).astype(int)
                 + ((c >= x["open"]) & (x["volume"] > x["vol_avg20"])).astype(int))
        return (score >= 4) & (c > x["ema200"])
    if name == "trend":
        return trend & x["st_up"] & (x["adx"] >= 23) & (x["plus_di"] > x["minus_di"])
    if name == "momentum":
        return trend & (x["ema20"] > x["ema50"]) & (x["macd_hist"] > 0) & x["rsi"].between(48,70) & (x["volume"] > x["vol_avg20"])
    if name == "breakout":
        return trend & (c > x["high20_previous"]) & (x["adx"] >= 20) & (x["volume"] > 1.2*x["vol_avg20"])
    raise ValueError("Неизвестная стратегия")

def simulate_strategy(x: pd.DataFrame, name: str, years: int, fee_bps: float, slippage_bps: float) -> dict:
    if len(x) < 230:
        raise ValueError("Недостаточно свечей для прогрева EMA200 и тестирования")
    signals = strategy_signal(x, name).fillna(False)
    cutoff = x["begin"].max() - pd.DateOffset(years=years)
    start = max(210, int(x["begin"].searchsorted(cutoff)))
    cost = (fee_bps+slippage_bps)/10000
    capital = 1.0
    curve, trades = [], []
    i = start
    while i < len(x):
        if bool(signals.iloc[i-1]) and i+10 < len(x):
            entry, exit_price = float(x.iloc[i]["open"]), float(x.iloc[i+10]["close"])
            if entry > 0 and exit_price > 0 and math.isfinite(entry) and math.isfinite(exit_price):
                factor = (exit_price/entry)*(1-cost)/(1+cost)
                trades.append({"entry_date":str(x.iloc[i]["begin"].date()), "exit_date":str(x.iloc[i+10]["begin"].date()), "return_pct":round((factor-1)*100,4)})
                base = capital
                for j in range(i,i+11):
                    marked = base*(float(x.iloc[j]["close"])/entry)/(1+cost)
                    if j == i+10: marked = base*factor
                    curve.append({"date":str(x.iloc[j]["begin"].date()),"equity":round(marked,6)})
                capital *= factor
                i += 11
                continue
        curve.append({"date":str(x.iloc[i]["begin"].date()),"equity":round(capital,6)})
        i += 1
    wins = [t["return_pct"] for t in trades if t["return_pct"] > 0]
    losses = [-t["return_pct"] for t in trades if t["return_pct"] < 0]
    pf = round(sum(wins)/sum(losses),3) if losses else None
    peak, max_dd = 1.0, 0.0
    for pt in curve:
        peak = max(peak,pt["equity"])
        max_dd = max(max_dd,(peak-pt["equity"])/peak)
    return {"strategy":name,"label":STRATEGY_LABELS[name],"count":len(trades),
            "win_rate_pct":round(100*len(wins)/len(trades),2) if trades else None,
            "profit_factor":pf,"max_drawdown_pct":round(100*max_dd,2),
            "total_return_pct":round(100*(capital-1),2),"trades":trades,"equity_curve":curve,
            "first_date":str(x.iloc[start]["begin"].date()),"last_date":str(x.iloc[-1]["begin"].date())}

@app.get("/api/strategies")
def available_strategies():
    return {"strategies": [{"id":k,"name":v} for k,v in STRATEGY_LABELS.items()],
            "method":"Дневные свечи; сигнал по закрытию, вход на следующем открытии; выход через 10 свечей; одна длинная позиция; комиссии и проскальзывание на обе стороны. Без дивидендов, корпоративных корректировок и налогов."}

@app.get("/api/strategies/compare/{ticker}")
def compare_strategies(ticker: str, board: str = "TQBR", years: int = 3, fee_bps: float = 10, slippage_bps: float = 10):
    validate_test(years,fee_bps,slippage_bps)
    ticker = ticker.upper()
    if board not in MARKET_BOARDS or not re.fullmatch(r"[A-Z0-9_.-]{1,24}",ticker):
        raise HTTPException(400,"Некорректный тикер или режим торгов")
    try:
        raw = get_candles(ticker, days=years*366+550, board=board)
        x = strategy_features(raw)
        results = [simulate_strategy(x,k,years,fee_bps,slippage_bps) for k in STRATEGY_LABELS]
        return {"ticker":ticker,"board":board,"years":years,"results":results,
                "warning":"Рейтинг по историческим данным не означает будущую доходность. Текущий каталог исключает выбывшие акции; нет дивидендов, корпоративных корректировок и налогов. Выбор лучшей стратегии по тем же данным ведёт к подгонке."}
    except Exception as e:
        raise HTTPException(400,str(e))


@app.get("/api/lookup")
def lookup(q: str):
    """Find one share by ticker or Russian company name and analyze it."""
    query = q.strip().casefold()
    if not query or len(query) > 80:
        raise HTTPException(status_code=400, detail="Введите название акции или тикер")
    try:
        items = get_catalog()
        exact = [x for x in items if x["ticker"].casefold() == query or str(x["name"]).casefold() == query]
        partial = [x for x in items if query in x["ticker"].casefold() or query in str(x["name"]).casefold()]
        matches = exact or partial
        if not matches:
            raise HTTPException(status_code=404, detail="Акция не найдена в текущем каталоге MOEX")
        if len(matches) > 1 and not exact:
            return {"matches": matches[:20], "ambiguous": True}
        item = matches[0]
        df = get_candles(item["ticker"], board=item["board"])
        analysis = calculate(df)
        tail = df.tail(180)
        candles = [{"date": str(r.begin.date()), "close": float(r.close)} for r in tail.itertuples()]
        return {"ambiguous": False, "item": item, "analysis": analysis, "candles": candles}
    except HTTPException:
        raise
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
