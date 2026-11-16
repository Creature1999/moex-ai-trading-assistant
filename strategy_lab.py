"""Educational adaptive trend strategy lab. No live trading or return promises."""
import math
import pandas as pd
import numpy as np

def indicators(df):
    x=df.copy().reset_index(drop=True)
    c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float)
    for n in (20,50,200): x[f'ema{n}']=c.ewm(span=n,adjust=False).mean()
    d=c.diff(); gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    x['rsi']=(100-100/(1+rs)).mask((loss==0)&(gain>0),100).mask((loss==0)&(gain==0),50)
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

def _completed_period_filters(x):
    """Historical multi-timeframe confirmation using PREVIOUS FINISHED bars only.
    Current-week/month OHLC may not be used to determine today's decision.
    """
    dates=pd.to_datetime(x.begin)
    result=[]
    for period,min_bars in [('W-SUN',50),('M',50)]:
        groups=dates.dt.to_period(period)
        grouped=x.assign(_period=groups).groupby('_period',sort=True,observed=True)
        closes=grouped.close.last().astype(float)
        ema20=closes.ewm(span=20,adjust=False).mean()
        ema50=closes.ewm(span=50,adjust=False).mean()
        # Signal of *previous* complete period; minimum bars reduces warmup bias.
        positive=((closes>ema20)&(ema20>ema50)&(pd.Series(range(len(closes)),index=closes.index)>=min_bars-1))
        previous=positive.shift(1).eq(True)
        result.append(groups.map(previous).fillna(False).astype(bool).to_numpy())
    return pd.Series(result[0]&result[1],index=x.index)

def strategy_signals(x,mode):
    """Fixed, inspectable rules. Signals may only use same-day closed information."""
    trend=(x.close>x.ema200)&(x.ema50>x.ema200)
    base=(trend & x.supertrend_up & (x.adx>=20) & (x.macd>x.macd_signal)
          & x.rsi.between(45,72) & (x.volume>=x.vol_avg20*.8))
    if mode=='adaptive':
        return base.fillna(False)
    if mode=='adaptive_mtf':
        return (base & _completed_period_filters(x)).fillna(False)
    if mode=='adaptive_pullback':
        # Enter on recovery above EMA20 in an existing upward market regime.
        recovered=(x.close>x.ema20)&(x.close.shift(1)<=x.ema20.shift(1))
        return (base & recovered).fillna(False)
    if mode=='momentum':
        return (trend & (x.macd>x.macd_signal) & x.rsi.between(50,70)
                & (x.volume>x.vol_avg20)).fillna(False)
    if mode=='breakout':
        return (trend & (x.close>x.high.shift(1).rolling(20).max())
                & (x.volume>x.vol_avg20*1.2) & (x.adx>=20)).fillna(False)
    raise ValueError('Неизвестная стратегия')


def _metrics(trades,curve,capital,mode,method):
    pnl=[t['pnl_cash'] for t in trades]
    gp=sum(max(v,0) for v in pnl); gl=-sum(min(v,0) for v in pnl)
    win=sum(v>0 for v in pnl)
    peak=1.; dd=0.
    for point in curve:
        peak=max(peak,point['equity'])
        dd=max(dd,(peak-point['equity'])/peak)
    return {'strategy':mode,'count':len(trades),
            'win_rate_pct':round(100*win/len(trades),2) if trades else None,
            'profit_factor':round(gp/gl,3) if gl>1e-12 else None,
            'max_drawdown_pct':round(100*dd,2),
            'total_return_pct':round(100*(capital-1),2),
            'trades':[{k:v for k,v in t.items() if k!='pnl_cash'} for t in trades],
            'equity_curve':curve,'method':method}


def backtest(df,years=3,fee_bps=10,slippage_bps=10,mode='adaptive',start_index=None,risk_pct=1.0):
    """Long-only, no leverage; ~risk_pct% equity at initial stop *before gaps*.

    A signal is formed at the close of t-1 and filled no earlier than open t.
    An entry is possible only on a *new* signal (false -> true), preventing
    immediate re-entry after a stop while the same setup persists.
    Stop updates based on t's high/ATR activate on t+1, not intraday on t.
    All account curves include mark-to-market equity while holding.
    """
    if len(df)<230:raise ValueError('Недостаточно истории (нужно 230 свечей)')
    if not (0<risk_pct<=3):raise ValueError('Риск должен быть между 0 и 3%')
    if not (0<=fee_bps<=100 and 0<=slippage_bps<=100):raise ValueError('Издержки 0–100 б.п.')
    x=indicators(df); sig=strategy_signals(x,mode)
    cutoff=x.begin.max()-pd.DateOffset(years=years)
    start=max(210,int(x.begin.searchsorted(cutoff))) if start_index is None else max(210,int(start_index))
    if start>=len(x):raise ValueError('Начало теста позже последней свечи')
    cost=(fee_bps+slippage_bps)/10000
    capital=1.; curve=[]; trades=[]
    position=None
    for i in range(start,len(x)):
        d=str(pd.Timestamp(x.begin.iloc[i]).date())
        opn=float(x.open.iloc[i]); high=float(x.high.iloc[i]); low=float(x.low.iloc[i]); close=float(x.close.iloc[i])
        if min(opn,high,low,close)<=0 or not all(map(math.isfinite,(opn,high,low,close))):
            raise ValueError('Обнаружена некорректная историческая свеча')
        if position is None:
            fresh=bool(sig.iloc[i-1]) and (i<2 or not bool(sig.iloc[i-2]))
            atr=float(x.atr.iloc[i-1])
            if fresh and math.isfinite(atr) and atr>0:
                entry=opn*(1+cost)
                stop=max(0.01,opn-2*atr)
                risk_per_share=entry-stop*(1-cost)
                qty=min(capital/entry,capital*risk_pct/100/risk_per_share) if risk_per_share>0 else 0.
                if qty>0:
                    before=capital
                    cash=capital-qty*entry
                    position={'qty':qty,'entry':entry,'capital_before':before,'cash':cash,
                              'entry_date':d,'stop':stop,'peak':high,'entry_idx':i}
        if position is not None:
            p=position; stop=p['stop']; why=None; sell=None
            # Reversal may only be seen after a completed previous day.
            if i>p['entry_idx'] and (not bool(x.supertrend_up.iloc[i-1]) or
                   float(x.close.iloc[i-1])<float(x.ema20.iloc[i-1])):
                sell=opn;why='trend reversal at next open'
            elif opn<=stop:
                sell=opn;why='opening gap below stop'
            elif low<=stop:
                sell=stop;why='stop intraday'
            elif i==len(x)-1:
                sell=close;why='end of backtest (marked at close)'
            if sell is not None:
                capital=p['cash']+p['qty']*sell*(1-cost)
                pnl=capital-p['capital_before']
                factor=(capital/p['capital_before']) if p['capital_before'] else 0
                trades.append({'entry_date':p['entry_date'],'exit_date':d,
                               'return_pct':round(100*(factor-1),3),
                               'pnl_cash':pnl,'exit_reason':why,
                               'risk_budget_pct':risk_pct})
                position=None
                curve.append({'date':d,'equity':capital})
                continue
            # This day's range updates next day's stop. NOT applied against today's low.
            old_peak=p['peak']; p['peak']=max(old_peak,high)
            atr_now=float(x.atr.iloc[i])
            if math.isfinite(atr_now) and atr_now>0:
                p['stop']=max(p['stop'],p['peak']-2.5*atr_now)
            # Conservative liquidation-value mark to market, including exit cost.
            equity=p['cash']+p['qty']*close*(1-cost)
            curve.append({'date':d,'equity':equity})
        else:
            curve.append({'date':d,'equity':capital})
    if not curve:raise ValueError('Нет свечей для расчёта')
    result=_metrics(trades,curve,capital,mode,
        'Учебный тест. Сигнал по закрытию; вход на следующем открытии только при новом сигнале; '
        'ATR-стоп 2×ATR, трейлинг 2.5×ATR, закрытие при развороте. '
        'Не более 1% риска капитала на первоначальном стопе (по умолчанию), без плеча. '
        'Комиссия и проскальзывание на входе и выходе. Гэп может превысить лимит риска. '
        'Нет дивидендов, налогов, корпоративных корректировок и настоящего исполнения.')
    result['first_date']=str(pd.Timestamp(x.begin.iloc[start]).date())
    result['last_date']=str(pd.Timestamp(x.begin.iloc[-1]).date())
    result['risk_pct']=risk_pct
    return result


def buy_and_hold(df,start_index,fee_bps=10,slippage_bps=10):
    """Fair fully-invested benchmark over precisely the same evaluation dates."""
    x=df.reset_index(drop=True)
    if start_index>=len(x):raise ValueError('Нет истории для удержания акции')
    cost=(fee_bps+slippage_bps)/10000
    entry=float(x.open.iloc[start_index])*(1+cost)
    if entry<=0:raise ValueError('Некорректная стартовая цена')
    qty=1/entry
    curve=[]
    for i in range(start_index,len(x)):
        last=float(x.close.iloc[i])*(1-cost)*qty
        curve.append({'date':str(pd.Timestamp(x.begin.iloc[i]).date()),'equity':last})
    ret=(curve[-1]['equity']-1)*100
    peak=1.;max_dd=0.
    for p in curve:
        peak=max(peak,p['equity']);max_dd=max(max_dd,100*(peak-p['equity'])/peak)
    return {'strategy':'buy_hold','total_return_pct':round(ret,2),
            'max_drawdown_pct':round(max_dd,2),'count':1,
            'win_rate_pct':None,'profit_factor':None}
