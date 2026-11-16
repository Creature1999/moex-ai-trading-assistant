"""Explainable, non-trading MOEX research. All features use CLOSED daily candles.
No external LLM, brokerage account, autonomous order execution, or claimed guaranteed return.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import re
import numpy as np
import pandas as pd
from strategy_lab import indicators

MOSCOW = ZoneInfo('Europe/Moscow')

def completed_daily(df, asof=None):
    """Conservatively omit today's partial exchange daily candle."""
    today = (asof or datetime.now(MOSCOW)).date()
    x = df.copy().sort_values('begin').drop_duplicates('begin')
    x = x[pd.to_datetime(x.begin).dt.date < today].reset_index(drop=True)
    if len(x) < 220:
        raise ValueError('Недостаточно завершённых дневных свечей: нужно от 220')
    return x


def aggregate_bars(df, period):
    x = df.copy()
    x['period'] = pd.to_datetime(x.begin).dt.to_period(period)
    g = x.groupby('period', sort=True, observed=True)
    bars = g.agg(begin=('begin','last'),open=('open','first'),high=('high','max'),
                 low=('low','min'),close=('close','last'),volume=('volume','sum'))
    return bars.reset_index(drop=True)


def timeframe_summary(df, name, min_count):
    if len(df) < min_count:
        return {'timeframe':name,'status':'НЕДОСТАТОЧНО ДАННЫХ','bars':len(df),'direction':'unknown'}
    # Use EMA20/50 regardless of timeframe; don't treat correlated measures as votes.
    c = df.close.astype(float)
    e20 = c.ewm(span=20,adjust=False).mean()
    e50 = c.ewm(span=50,adjust=False).mean()
    macd = c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    macd_signal = macd.ewm(span=9,adjust=False).mean()
    close=float(c.iloc[-1]); e20v=float(e20.iloc[-1]); e50v=float(e50.iloc[-1])
    upward = close>e20v and e20v>e50v
    downward = close<e20v and e20v<e50v
    direction = 'up' if upward else 'down' if downward else 'mixed'
    return {'timeframe':name,'status': {'up':'ВОСХОДЯЩИЙ','down':'НИСХОДЯЩИЙ','mixed':'СМЕШАННЫЙ'}[direction],
            'direction':direction,'bars':len(df),'date':str(pd.Timestamp(df.begin.iloc[-1]).date()),
            'close':round(close,4),'ema20':round(e20v,4),'ema50':round(e50v,4),
            'macd_positive':bool(macd.iloc[-1]>macd_signal.iloc[-1])}


def complete_higher_periods(daily, period):
    # Completed bars only: a candle for the current calendar week/month is excluded.
    x = daily.copy(); x['period'] = pd.to_datetime(x.begin).dt.to_period(period)
    current_period = pd.Timestamp(datetime.now(MOSCOW).date()).to_period(period)
    return aggregate_bars(x[x.period < current_period],period)


def analyze_research(df, risk_capital=100000.0, risk_pct=1.0):
    x = completed_daily(df)
    daily = timeframe_summary(x,'1D',80)
    weekly = timeframe_summary(complete_higher_periods(x,'W-SUN'),'1W',60)
    monthly = timeframe_summary(complete_higher_periods(x,'M'),'1M',55)
    i = indicators(x)
    last = i.iloc[-1]
    rsi=float(last.rsi) if pd.notna(last.rsi) else None
    adx=float(last.adx) if pd.notna(last.adx) else None
    vol_ratio=float(last.volume/last.vol_avg20) if pd.notna(last.vol_avg20) and last.vol_avg20>0 else None
    atr=float(last.atr) if pd.notna(last.atr) and last.atr>0 else None
    macd_hist=float(last.macd-last.macd_signal)
    conditions={
        'Дневной тренд EMA':daily['direction']=='up',
        'Недельный тренд EMA':weekly['direction']=='up',
        'Месячный тренд не снижается':monthly['direction']!='down' and monthly['direction']!='unknown',
        'Supertrend растущий':bool(last.supertrend_up),
        'MACD выше сигнальной линии':macd_hist>0,
        'ADX не меньше 20':adx is not None and adx>=20,
        'RSI от 45 до 70':rsi is not None and 45<=rsi<=70,
        'Объём не меньше 80% среднего':vol_ratio is not None and vol_ratio>=.8,
    }
    # If monthly history is missing, no high-confidence setup.
    confirmed=all(conditions.values())
    close=float(last.close)
    entry=round(close+0.2*atr,2) if atr else None
    stop=round(entry-2*atr,2) if entry is not None else None
    target=round(entry+4*atr,2) if entry is not None else None
    unit_risk=(entry-stop) if entry is not None and stop is not None else 0
    allowed_risk=risk_capital*risk_pct/100
    shares=min(math.floor(allowed_risk/unit_risk),math.floor(risk_capital/entry)) if entry and unit_risk>0 else 0
    shares=max(0,shares)
    # not an order: lot size, spread, gaps, dividends, taxes are not modeled
    return {
        'asof':str(pd.Timestamp(last.begin).date()),'status':'НАБЛЮДАТЬ ЗА ТРИГГЕРОМ' if confirmed else 'НЕТ ПОДТВЕРЖДЁННОГО ВХОДА',
        'timeframes':[daily,weekly,monthly], 'checks':[{'name':k,'ok':bool(v)} for k,v in conditions.items()],
        'confirmed':sum(conditions.values()),'total_checks':len(conditions),
        'indicators':{'close':round(close,4),'ema20':round(float(last.ema20),4),
                      'ema50':round(float(last.ema50),4),'ema200':round(float(last.ema200),4),
                      'supertrend':round(float(last.supertrend),4) if pd.notna(last.supertrend) else None,
                      'supertrend_up':bool(last.supertrend_up),'macd_hist':round(macd_hist,4),
                      'rsi':round(rsi,2) if rsi is not None else None,
                      'adx':round(adx,2) if adx is not None else None,
                      'volume_ratio':round(vol_ratio,2) if vol_ratio is not None else None,
                      'atr':round(atr,4) if atr is not None else None},
        'scenario':{'entry_trigger':entry,'stop':stop,'target':target,'risk_reward':2.0 if atr else None,
                    'capital_rub':risk_capital,'risk_pct':risk_pct,
                    'max_theoretical_shares':shares,'theoretical_loss_rub':round(shares*unit_risk,2),
                    'valid_signal':confirmed,
                    'note':'Гипотетический сценарий по закрытой свече, НЕ приказ покупать. Не учтены лоты MOEX, гэпы, комиссии, спред, дивиденды и особенности исполнения.'},
        'method':'1D: последнее завершённое закрытие; 1W/1M: только завершённые календарные периоды; 8 проверок совпадения — НЕ вероятность прибыли.'}


def _sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z,-35,35)))


def model_holdout(df):
    """Small ridge logistic model. Target next session CLOSE > current CLOSE.
    Chronological train/test, 1-bar embargo and training-only normalization.
    Informative diagnostic, not a trading strategy or calibrated chance of profit.
    """
    x=completed_daily(df)
    d=indicators(x)
    c=d.close.astype(float)
    ret=c.pct_change()
    mat=pd.DataFrame({
        'r1':ret,'r5':c.pct_change(5), 'r20':c.pct_change(20),
        'ema_gap':(d.ema20/d.ema50-1), 'long_gap':c/d.ema200-1,
        'rsi':(d.rsi-50)/50, 'adx':d.adx/100,
        'atr_pct':d.atr/c,'vol_rel':d.volume/d.vol_avg20-1,
        'macd_pct':(d.macd-d.macd_signal)/c,
    }).replace([np.inf,-np.inf],np.nan)
    # Exclude latest target, which is unknown. All features are as-of their day's close.
    labeled=mat.iloc[:-1].copy()
    labeled['target']=(c.shift(-1).iloc[:-1].to_numpy() > c.iloc[:-1].to_numpy()).astype(float)
    labeled=labeled.dropna()
    if len(labeled)<240: return {'status':'insufficient','message':'Нужно от 240 размеченных дневных свечей'}
    X=labeled.drop(columns=['target']).to_numpy(dtype=float);y=labeled.target.to_numpy(dtype=float)
    split=int(len(X)*.7)
    # One-bar embargo at boundary: don't train with a label whose outcome reaches test day.
    trainX=X[:split-1];trainY=y[:split-1]
    testX=X[split:];testY=y[split:]
    mean=trainX.mean(axis=0);std=np.maximum(trainX.std(axis=0),1e-8)
    A=np.clip((trainX-mean)/std,-8,8)
    B=np.clip((testX-mean)/std,-8,8)
    # Intercept not regularized; simple fixed hyperparameters (no tuning on test).
    weights=np.zeros(A.shape[1]);bias=0.;lr=.08;lam=.03
    for _ in range(400):
        pred=_sigmoid(A@weights+bias)
        weights-=lr*(A.T@(pred-trainY)/len(A)+lam*weights)
        bias-=lr*np.mean(pred-trainY)
    prob=_sigmoid(B@weights+bias)
    accuracy=float(np.mean((prob>=.5)==testY))
    baseline=max(float(np.mean(testY)),1-float(np.mean(testY)))
    latest=mat.iloc[-1]
    p_next=float(_sigmoid(np.clip((latest.to_numpy(dtype=float)-mean)/std,-8,8)@weights+bias)) if latest.notna().all() else None
    return {'status':'ok','model':'Логистическая регрессия (ridge), без LLM',
            'horizon':'Следующая дневная свеча','prediction_up_score_pct':round(p_next*100,1) if p_next is not None else None,
            'direction':'рост' if p_next is not None and p_next>=.5 else 'снижение' if p_next is not None else 'нет данных',
            'test_accuracy_pct':round(accuracy*100,2),'baseline_accuracy_pct':round(baseline*100,2),
            'test_size':len(testY),'train_size':len(trainY),
            'beats_baseline':bool(accuracy>baseline),
            'test_from':str(pd.Timestamp(x.begin.iloc[labeled.index[split]]).date()),
            'note':'Оценка модели НЕ калиброванная вероятность сделки; тест — направление следующего закрытия, не доходность. Разделение по времени 70/30, 1-дневный embargo. Комиссии здесь не применимы; для торговли нужен отдельный бэктест.'}
