"""Educational adaptive trend strategy lab. No live trading or return promises."""
import math
import pandas as pd
import numpy as np

def indicators(df):
    x=df.copy().reset_index(drop=True)
    c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float)
    for n in (20,50,200): x[f'ema{n}']=c.ewm(span=n,adjust=False).mean()
    d=c.diff(); gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    x['rsi']=100-100/(1+gain/loss.replace(0,np.nan))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    x['macd']=e12-e26; x['macd_signal']=x.macd.ewm(span=9,adjust=False).mean()
    pc=c.shift(1); tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    x['atr']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    up=h.diff(); down=-l.diff(); plus=up.where((up>down)&(up>0),0); minus=down.where((down>up)&(down>0),0)
    pdi=100*plus.ewm(alpha=1/14,adjust=False).mean()/x.atr.replace(0,np.nan)
    mdi=100*minus.ewm(alpha=1/14,adjust=False).mean()/x.atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x['adx']=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x['vol_avg20']=x.volume.astype(float).rolling(20).mean()
    mid=(h+l)/2; upper=mid+3*x.atr; lower=mid-3*x.atr
    st=np.full(len(x),np.nan); direction=np.zeros(len(x),dtype=int)
    fu=upper.to_numpy(copy=True); fl=lower.to_numpy(copy=True)
    for i in range(1,len(x)):
        if not np.isfinite(fu[i]) or not np.isfinite(fl[i]):continue
        if np.isfinite(fu[i-1]) and (fu[i]>=fu[i-1] and c.iloc[i-1]<=fu[i-1]):fu[i]=fu[i-1]
        if np.isfinite(fl[i-1]) and (fl[i]<=fl[i-1] and c.iloc[i-1]>=fl[i-1]):fl[i]=fl[i-1]
        if direction[i-1]==0:direction[i]=1 if c.iloc[i]>fu[i] else -1
        elif direction[i-1]==1:direction[i]=-1 if c.iloc[i]<fl[i] else 1
        else:direction[i]=1 if c.iloc[i]>fu[i] else -1
        st[i]=fl[i] if direction[i]==1 else fu[i]
    x['supertrend']=st;x['supertrend_up']=direction==1
    return x

def strategy_signals(x,mode):
    trend=(x.close>x.ema200)&(x.ema50>x.ema200)
    if mode=='adaptive':
        return trend & x.supertrend_up & (x.adx>=20) & (x.macd>x.macd_signal) & x.rsi.between(45,72) & (x.volume>=x.vol_avg20*.8)
    if mode=='momentum':
        return trend & (x.macd>x.macd_signal) & x.rsi.between(50,70) & (x.volume>x.vol_avg20)
    if mode=='breakout':
        return trend & (x.close>x.high.shift(1).rolling(20).max()) & (x.volume>x.vol_avg20*1.2) & (x.adx>=20)
    raise ValueError('Неизвестная стратегия')

def backtest(df,years=3,fee_bps=10,slippage_bps=10,mode='adaptive'):
    if len(df)<230:raise ValueError('Недостаточно истории (нужно 230 свечей)')
    x=indicators(df); sig=strategy_signals(x,mode).fillna(False)
    cutoff=x.begin.max()-pd.DateOffset(years=years)
    start=max(210,int(x.begin.searchsorted(cutoff)))
    cost=(fee_bps+slippage_bps)/10000
    capital=1.; trades=[]; curve=[]; i=start
    while i<len(x):
        if i>=1 and sig.iloc[i-1] and np.isfinite(x.atr.iloc[i-1]) and x.atr.iloc[i-1]>0:
            entry=float(x.open.iloc[i])*(1+cost)
            if not np.isfinite(entry) or entry<=0:i+=1;continue
            stop=entry-2*float(x.atr.iloc[i-1]); peak=float(x.high.iloc[i-1]); exit_idx=None; exit_price=None; why=''
            for j in range(i,len(x)):
                # Stop based only on information known before this day's trading.
                if float(x.open.iloc[j])<=stop:
                    exit_price=float(x.open.iloc[j])*(1-cost);why='gap/stop';exit_idx=j;break
                if float(x.low.iloc[j])<=stop:
                    exit_price=stop*(1-cost);why='stop';exit_idx=j;break
                # Exit on next open after a close-based reversal, not on same close.
                if j>i and (not bool(x.supertrend_up.iloc[j-1]) or float(x.close.iloc[j-1])<float(x.ema20.iloc[j-1])):
                    exit_price=float(x.open.iloc[j])*(1-cost);why='trend exit';exit_idx=j;break
                peak=max(peak,float(x.high.iloc[j]));
                # Trailing stop updates for NEXT session only.
                stop=max(stop,peak-2.5*float(x.atr.iloc[j])) if np.isfinite(x.atr.iloc[j]) else stop
            if exit_idx is None:exit_idx=len(x)-1;exit_price=float(x.close.iloc[-1])*(1-cost);why='end of data'
            factor=max(0.,exit_price/entry);before=capital;capital*=factor
            trades.append({'entry_date':str(x.begin.iloc[i].date()),'exit_date':str(x.begin.iloc[exit_idx].date()),'return_pct':round((factor-1)*100,3),'exit_reason':why})
            for j in range(i,exit_idx+1):
                marked=capital if j==exit_idx else before*float(x.close.iloc[j])*(1-cost)/entry
                curve.append({'date':str(x.begin.iloc[j].date()),'equity':round(marked,6)})
            i=exit_idx+1
        else:
            curve.append({'date':str(x.begin.iloc[i].date()),'equity':round(capital,6)});i+=1
    returns=[t['return_pct']/100 for t in trades];wins=[r for r in returns if r>0];losses=[r for r in returns if r<0]
    gp=sum(wins);gl=-sum(losses);peak=1.;dd=0.
    for point in curve:
        peak=max(peak,point['equity']);dd=max(dd,(peak-point['equity'])/peak)
    return {'strategy':mode,'count':len(trades),'win_rate_pct':round(100*len(wins)/len(trades),2) if trades else None,
        'profit_factor':round(gp/gl,3) if gl>0 else None,'max_drawdown_pct':round(100*dd,2),
        'total_return_pct':round(100*(capital-1),2),'trades':trades,'equity_curve':curve,
        'first_date':str(x.begin.iloc[start].date()),'last_date':str(x.begin.iloc[-1].date()),
        'method':'Учебный тест: сигнал на закрытии, вход на следующем открытии, ATR-стоп и трейлинг, комиссия и проскальзывание на обе стороны. Без дивидендов и корректировок корпоративных событий.'}
