"""MOEX AI Trading Assistant v10: inspectable educational long-only backtests.

Only CLOSED daily candles are supplied by caller. Signals at t-1, execution at t.
No broker access, no optimisation on holdout, no future data in features.
Historical data is not total-return adjusted; do not treat as real trading outcomes.
"""
import math
from collections import Counter
import numpy as np
import pandas as pd

STRATEGIES = (
    'adaptive', 'adaptive_mtf', 'adaptive_pullback', 'momentum', 'breakout',
    'regime_pullback', 'regime_range',
)
STRATEGY_NAMES = {
    'adaptive': 'Adaptive Trend · ATR',
    'adaptive_mtf': 'Adaptive + 1W/1M · ATR',
    'adaptive_pullback': 'Adaptive Pullback (строгий) · ATR',
    'momentum': 'Momentum · ATR',
    'breakout': 'Breakout · ATR',
    'regime_pullback': 'Тренд + откат к EMA20 · ATR',
    'regime_range': 'Боковик + возврат к среднему · ATR',
}
REGIME_NAMES = {
    'up': 'Восходящий', 'down': 'Нисходящий',
    'sideways': 'Боковой', 'mixed': 'Неопределённый',
}


def indicators(df):
    x = df.copy().reset_index(drop=True)
    c = x.close.astype(float)
    h = x.high.astype(float)
    l = x.low.astype(float)
    for n in (20, 50, 200):
        x[f'ema{n}'] = c.ewm(span=n, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain/loss.replace(0, np.nan)
    x['rsi'] = (100-100/(1+rs)).mask((loss==0)&(gain>0),100).mask((loss==0)&(gain==0),50)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    x['macd'] = e12-e26
    x['macd_signal'] = x.macd.ewm(span=9, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    x['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up = h.diff()
    down = -l.diff()
    plus = up.where((up>down)&(up>0), 0)
    minus = down.where((down>up)&(down>0), 0)
    pdi = 100*plus.ewm(alpha=1/14, adjust=False).mean()/x.atr.replace(0,np.nan)
    mdi = 100*minus.ewm(alpha=1/14, adjust=False).mean()/x.atr.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x['plus_di'] = pdi
    x['minus_di'] = mdi
    x['adx'] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x['vol_avg20'] = x.volume.astype(float).rolling(20).mean()
    x['mean20'] = c.rolling(20).mean()
    x['std20'] = c.rolling(20).std()
    x['high20_before'] = h.shift(1).rolling(20).max()
    mid = (h+l)/2
    upper = mid+3*x.atr
    lower = mid-3*x.atr
    st = np.full(len(x),np.nan)
    direction = np.zeros(len(x),dtype=int)
    fu = upper.to_numpy(copy=True)
    fl = lower.to_numpy(copy=True)
    for i in range(1,len(x)):
        if not np.isfinite(fu[i]) or not np.isfinite(fl[i]):
            continue
        if np.isfinite(fu[i-1]) and fu[i]>=fu[i-1] and c.iloc[i-1]<=fu[i-1]:
            fu[i] = fu[i-1]
        if np.isfinite(fl[i-1]) and fl[i]<=fl[i-1] and c.iloc[i-1]>=fl[i-1]:
            fl[i] = fl[i-1]
        if direction[i-1]==0:
            direction[i] = 1 if c.iloc[i]>fu[i] else -1
        elif direction[i-1]==1:
            direction[i] = -1 if c.iloc[i]<fl[i] else 1
        else:
            direction[i] = 1 if c.iloc[i]>fu[i] else -1
        st[i] = fl[i] if direction[i]==1 else fu[i]
    x['supertrend'] = st
    x['supertrend_up'] = direction==1
    return x


def _completed_period_filters(x):
    """Confirm from the PREVIOUS fully finished week/month (no current partial bar).

    Unlike v9's 50 *monthly* bar warm-up, use EMA6/12 and 24 finished
    months; weekly uses EMA10/30 and 60 weeks. A missing history is not a BUY.
    """
    dates = pd.to_datetime(x.begin)
    confirmations = []
    for period, short, long, warmup in [('W-SUN',10,30,60), ('M',6,12,24)]:
        group = dates.dt.to_period(period)
        closes = x.assign(_period=group).groupby('_period', sort=True, observed=True).close.last().astype(float)
        fast = closes.ewm(span=short,adjust=False).mean()
        slow = closes.ewm(span=long,adjust=False).mean()
        ready = pd.Series(np.arange(len(closes))>=warmup-1,index=closes.index)
        confirmed = ((closes>slow)&(fast>slow)&ready).shift(1).eq(True)
        confirmations.append(group.map(confirmed).fillna(False).astype(bool).to_numpy())
    return pd.Series(confirmations[0]&confirmations[1],index=x.index)


def market_regimes(x):
    """Deterministic classification after each CLOSED session, never future bars."""
    strong = x.adx>=18
    up = strong & (x.close>x.ema200) & (x.ema50>x.ema200) & (x.plus_di>x.minus_di)
    down = strong & (x.close<x.ema200) & (x.ema50<x.ema200) & (x.minus_di>x.plus_di)
    sideways = x.adx<18
    return pd.Series(np.select([up,down,sideways],['up','down','sideways'],default='mixed'),index=x.index)


def _conditions(x,mode):
    if mode not in STRATEGIES:
        raise ValueError('Неизвестная стратегия')
    trend=(x.close>x.ema200)&(x.ema50>x.ema200)
    volume=x.volume.astype(float)
    regimes=market_regimes(x)
    base=[('Цена выше EMA200 и EMA50 выше EMA200',trend),
          ('Supertrend подтверждает рост',x.supertrend_up),
          ('ADX ≥ 20',x.adx>=20),
          ('MACD выше сигнальной линии',x.macd>x.macd_signal),
          ('RSI 45–72',x.rsi.between(45,72)),
          ('Объём ≥ 80% среднего',volume>=x.vol_avg20*.8)]
    if mode=='adaptive':
        return base
    if mode=='adaptive_mtf':
        return base+[('Подтверждение предыдущих 1W и 1M',_completed_period_filters(x))]
    if mode=='adaptive_pullback':
        recovered=(x.close>x.ema20)&(x.close.shift(1)<=x.ema20.shift(1))
        return base+[('Возврат над EMA20 после отката',recovered)]
    if mode=='momentum':
        return [('Долгосрочный тренд',trend),
                ('MACD положительный',x.macd>x.macd_signal),
                ('RSI 50–70',x.rsi.between(50,70)),
                ('Объём выше среднего',volume>x.vol_avg20)]
    if mode=='breakout':
        return [('Долгосрочный тренд',trend),
                ('Цена выше предшествующего 20-дневного максимума',x.close>x.high20_before),
                ('Объём выше 120% среднего',volume>x.vol_avg20*1.2),
                ('ADX ≥ 20',x.adx>=20)]
    if mode=='regime_pullback':
        # New hypothesis: relaxed recovery near EMA20 in a proven up-regime.
        close_to_ema=x.close.between(x.ema20*.985,x.ema20*1.025)
        return [('Режим: восходящий',regimes=='up'),
                ('Цена недалеко от EMA20',close_to_ema),
                ('Сегодняшнее закрытие выше вчерашнего',x.close>x.close.shift(1)),
                ('RSI 40–72',x.rsi.between(40,72)),
                ('Объём ≥ 50% среднего',volume>=x.vol_avg20*.5)]
    if mode=='regime_range':
        # New hypothesis: buy mild recovery after an oversold sideways day.
        return [('Режим: боковой',regimes=='sideways'),
                ('Цена ниже среднего − 1σ',x.close<x.mean20-x.std20),
                ('RSI ≤ 45',x.rsi<=45),
                ('Закрытие выше предыдущего',x.close>x.close.shift(1)),
                ('Объём ≥ 40% среднего',volume>=x.vol_avg20*.4)]
    raise ValueError('Неизвестная стратегия')


def strategy_signals(x,mode):
    """Exactly the same predicates used by the diagnostic funnel."""
    gates=_conditions(x,mode)
    result=pd.Series(True,index=x.index)
    for _,gate in gates:
        result &= gate.fillna(False).astype(bool)
    return result.fillna(False)


def signal_diagnostics(x,mode,start_index,end_index=None):
    """Sequential funnel: a failure is assigned to the FIRST failed gate.

    Gates are run on closed daily bars; they do not count executable trades.
    Fresh signals are false->true changes, before position constraints.
    """
    end=len(x) if end_index is None else min(len(x),int(end_index))
    start=max(0,int(start_index))
    if start>=end:
        return {'days':0,'signal_days':0,'fresh_signals':0,'gates':[], 'regimes':{},'last_regime':'mixed'}
    gates=_conditions(x,mode)
    current=pd.Series(True,index=x.index)
    funnel=[]
    for name,gate in gates:
        yes=gate.fillna(False).astype(bool)
        before=int(current.iloc[start:end].sum())
        current &= yes
        after=int(current.iloc[start:end].sum())
        funnel.append({'name':name,'passed':after,'rejected_here':before-after})
    fresh=current & ~current.shift(1,fill_value=False)
    regimes=market_regimes(x)
    counts=regimes.iloc[start:end].value_counts()
    decisions=[]
    for i in range(max(start,end-15),end):
        failed=next((name for name,gate in gates if not bool(gate.fillna(False).iloc[i])),None)
        if failed:
            decision='Без сигнала — '+failed
        elif bool(fresh.iloc[i]):
            decision='Новый сигнал — возможный вход на следующем открытии, если позиция свободна'
        else:
            decision='Условия выполнены, но сигнал уже действовал (не новый вход)'
        decisions.append({'date':str(pd.Timestamp(x.begin.iloc[i]).date()),
                          'close':round(float(x.close.iloc[i]),4),
                          'regime':str(regimes.iloc[i]),
                          'new_signal':bool(fresh.iloc[i]),
                          'decision':decision})
    return {'days':end-start, 'signal_days':int(current.iloc[start:end].sum()),
            'fresh_signals':int(fresh.iloc[start:end].sum()),
            'gates':funnel,
            'regimes':{key:int(counts.get(key,0)) for key in REGIME_NAMES},
            'last_regime':str(regimes.iloc[end-1]),
            'last_decisions':decisions,
            'note':'Причины на последних 15 свечах относятся к закрытию дня и возможному входу на следующей сессии. Это не журнал реальных заявок. Фильтры применяются последовательно.'}


def _metrics(trades,curve,capital,mode,method):
    pnl=[t['pnl_cash'] for t in trades]
    gp=sum(max(v,0) for v in pnl)
    gl=-sum(min(v,0) for v in pnl)
    win=sum(v>0 for v in pnl)
    peak=1.0
    dd=0.0
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
    """Long only, no leverage. Signals close(t-1)->entry open(t).

    Stop/trailing updates use t's completed candle and apply from t+1.
    Exit on an opening gap below stop at OPEN, not idealised stop price.
    Fee+slippage applied on BOTH legs; does not handle corporate actions.
    """
    if len(df)<230:
        raise ValueError('Недостаточно истории (нужно 230 свечей)')
    if not (0<risk_pct<=3):
        raise ValueError('Риск должен быть между 0 и 3%')
    if not (0<=fee_bps<=100 and 0<=slippage_bps<=100):
        raise ValueError('Издержки 0–100 б.п.')
    x=indicators(df)
    sig=strategy_signals(x,mode)
    cutoff=x.begin.max()-pd.DateOffset(years=years)
    start=max(210,int(x.begin.searchsorted(cutoff))) if start_index is None else max(210,int(start_index))
    if start>=len(x):
        raise ValueError('Начало теста позже последней свечи')
    cost=(fee_bps+slippage_bps)/10000
    capital=1.0
    curve=[]
    trades=[]
    position=None
    for i in range(start,len(x)):
        d=str(pd.Timestamp(x.begin.iloc[i]).date())
        opn=float(x.open.iloc[i]); high=float(x.high.iloc[i]); low=float(x.low.iloc[i]); close=float(x.close.iloc[i])
        if min(opn,high,low,close)<=0 or not all(map(math.isfinite,(opn,high,low,close))):
            raise ValueError('Обнаружена некорректная историческая свеча')
        # A new signal may open only if there was no active signal the prior day.
        if position is None:
            fresh=bool(sig.iloc[i-1]) and (i<2 or not bool(sig.iloc[i-2]))
            atr=float(x.atr.iloc[i-1])
            if fresh and math.isfinite(atr) and atr>0:
                entry=opn*(1+cost)
                stop=max(.01,opn-2*atr)
                exit_at_stop=stop*(1-cost)
                risk_per_share=entry-exit_at_stop
                qty=min(capital/entry,capital*risk_pct/100/risk_per_share) if risk_per_share>0 else 0.
                if qty>0:
                    before=capital
                    cash=capital-qty*entry
                    position={'qty':qty,'entry':entry,'raw_entry':opn,
                              'capital_before':before,'cash':cash,
                              'entry_date':d,'stop':stop,'initial_stop':stop,
                              'planned_risk_cash':qty*risk_per_share,
                              'peak':high,'entry_idx':i,
                              'position_pct':100*qty*entry/before}
        if position is not None:
            p=position
            stop=p['stop'];why=None;sell=None
            # Trend reversal is known from the completed PRIOR bar.
            if i>p['entry_idx'] and (not bool(x.supertrend_up.iloc[i-1]) or
                    float(x.close.iloc[i-1])<float(x.ema20.iloc[i-1])):
                sell=opn;why='Разворот тренда — выход на следующем открытии'
            elif opn<=stop:
                sell=opn;why='Гэп ниже стопа — выход по открытию'
            elif low<=stop:
                sell=stop;why='Срабатывание ATR-стопа внутри дня'
            elif i==len(x)-1:
                sell=close;why='Конец истории — закрытие по последней цене'
            if sell is not None:
                net_exit=sell*(1-cost)
                capital=p['cash']+p['qty']*net_exit
                pnl=capital-p['capital_before']
                factor=capital/p['capital_before'] if p['capital_before'] else 0
                r_multiple=pnl/p['planned_risk_cash'] if p['planned_risk_cash']>0 else None
                trades.append({
                    'entry_date':p['entry_date'],'exit_date':d,
                    'entry_price':round(p['raw_entry'],4),'entry_price_with_costs':round(p['entry'],4),
                    'exit_price':round(sell,4),'exit_price_after_costs':round(net_exit,4),
                    'initial_stop':round(p['initial_stop'],4),'stop_at_exit':round(stop,4),
                    'bars_held':i-p['entry_idx']+1,
                    'position_pct':round(p['position_pct'],2),
                    'return_pct':round(100*(factor-1),3),
                    'r_multiple':round(r_multiple,3) if r_multiple is not None else None,
                    'pnl_cash':pnl,'exit_reason':why,'risk_budget_pct':risk_pct,
                })
                position=None
                curve.append({'date':d,'equity':capital})
                continue
            # Today's intraday high only updates tomorrow's stop.
            p['peak']=max(p['peak'],high)
            atr_now=float(x.atr.iloc[i])
            if math.isfinite(atr_now) and atr_now>0:
                p['stop']=max(p['stop'],p['peak']-2.5*atr_now)
            equity=p['cash']+p['qty']*close*(1-cost)
            curve.append({'date':d,'equity':equity})
        else:
            curve.append({'date':d,'equity':capital})
    result=_metrics(trades,curve,capital,mode,
        'Сигнал после закрытия, вход на следующем открытии по свежему сигналу. '
        'Стоп 2×ATR, трейлинг 2.5×ATR со следующего дня. Без плеча. '
        'Расчётный риск 1% на исходном стопе; гэпы могут увеличить убыток. '
        'Комиссия и проскальзывание на обе стороны. Без дивидендов, корпоративных событий, налогов и точного биржевого исполнения.')
    result['first_date']=str(pd.Timestamp(x.begin.iloc[start]).date())
    result['last_date']=str(pd.Timestamp(x.begin.iloc[-1]).date())
    result['risk_pct']=risk_pct
    result['diagnostics']=signal_diagnostics(x,mode,start)
    result['exit_reasons']=dict(Counter(t['exit_reason'] for t in result['trades']))
    result['note']='Доходность дана в % условного капитала; цены в рублях или единицах котировки. Нулевая доходность при 0 сделок НЕ означает успешную стратегию.'
    return result


def buy_and_hold(df,start_index,fee_bps=10,slippage_bps=10):
    """Fully-invested, price-only benchmark over the very same evaluation dates."""
    x=df.reset_index(drop=True)
    if start_index>=len(x):
        raise ValueError('Нет истории для удержания акции')
    cost=(fee_bps+slippage_bps)/10000
    entry=float(x.open.iloc[start_index])*(1+cost)
    if entry<=0:
        raise ValueError('Некорректная стартовая цена')
    qty=1/entry
    curve=[]
    for i in range(start_index,len(x)):
        last=float(x.close.iloc[i])*(1-cost)*qty
        curve.append({'date':str(pd.Timestamp(x.begin.iloc[i]).date()),'equity':last})
    ret=(curve[-1]['equity']-1)*100
    peak=1.0;max_dd=0.0
    for p in curve:
        peak=max(peak,p['equity'])
        max_dd=max(max_dd,100*(peak-p['equity'])/peak)
    return {'strategy':'buy_hold','total_return_pct':round(ret,2),
            'max_drawdown_pct':round(max_dd,2),'count':1,
            'win_rate_pct':None,'profit_factor':None}
