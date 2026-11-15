import os, sqlite3, threading, time
from datetime import datetime, timedelta, timezone
from contextlib import closing
from typing import Optional
import requests
import pandas as pd
import feedparser
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
APP_TITLE = "MOEX AI Trading Assistant"
DB_PATH = os.getenv("DB_PATH", "assistant.db")
TICKERS = [x.strip().upper() for x in os.getenv("TICKERS", "SBER,GAZP,LKOH,ROSN,YDEX").split(",") if x.strip()]
BOARD = os.getenv("MOEX_BOARD", "TQBR")
SCAN_INTERVAL = max(0, int(os.getenv("SCAN_INTERVAL_MINUTES", "0")))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RSS_FEEDS = [x.strip() for x in os.getenv("RSS_FEEDS", "").split(",") if x.strip()]

app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory="static"), name="static")
scan_lock = threading.Lock()
scheduler_started = False

def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with closing(db()) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT, ticker TEXT,
            signal TEXT, score INTEGER, close REAL, ema20 REAL, ema50 REAL, ema200 REAL,
            rsi REAL, atr REAL, entry REAL, stop REAL, target REAL, reason TEXT)""")
        con.commit()

def get_candles(ticker: str, days: int = 900) -> pd.DataFrame:
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{BOARD}/securities/{ticker}/candles.json"
    params = {"from": start, "interval": 24, "iss.meta": "off", "iss.only": "candles"}
    r = requests.get(url, params=params, timeout=25, headers={"User-Agent": "MOEX-AI-Trading-Assistant/1.0"})
    r.raise_for_status()
    block = r.json().get("candles", {})
    rows = block.get("data", [])
    cols = block.get("columns", [])
    if not rows:
        raise ValueError("MOEX ISS не вернул дневные свечи. Проверьте тикер, режим торгов и доступность данных.")
    df = pd.DataFrame(rows, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["begin"] = pd.to_datetime(df["begin"])
    return df.dropna(subset=["close"]).sort_values("begin").reset_index(drop=True)

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
    if score >= 4 and last.close > last.ema200 and pd.notna(last.atr):
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
                # Alert only on candidate setups; avoid repeated duplicate alerts in one scan.
                if notify and analysis["signal"] == "ВОЗМОЖНЫЙ БЫЧИЙ СЕТАП":
                    msg = (f"MOEX Trading Assistant · {ticker}\n{analysis['signal']}\n"
                           f"Оценка фильтров: {analysis['score']}/5\n"
                           f"Последнее закрытие: {analysis['close']:.2f}\n"
                           f"Учебный вход: {analysis['entry']:.2f}\nСтоп: {analysis['stop']:.2f}\n"
                           f"Цель 2R: {analysis['target']:.2f}\n{analysis['reason']}\n"
                           "Не является инвестиционной рекомендацией.")
                    send_telegram(msg)
            except Exception as e:
                results.append({"ticker": ticker, "signal": "ОШИБКА ДАННЫХ", "score": 0, "reason": str(e)})
        return {"status": "ok", "scanned_at": datetime.now(timezone.utc).isoformat(), "results": results}
    finally:
        scan_lock.release()

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
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/api/config")
def config():
    return {"tickers": TICKERS, "board": BOARD, "scan_interval_minutes": SCAN_INTERVAL,
            "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)}

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
def chart(ticker: str):
    try:
        df = get_candles(ticker.upper())
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
