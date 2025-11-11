import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import skew, kurtosis

def compute_weekly_features(df_1s, funding=None, oi=None, liq=None, book=None):
    df = df_1s.copy()
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    df.set_index('open_time', inplace=True)
    df = df.sort_index()
    
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_ret'] = df['ret'].abs()
    
    weekly = df.resample('W-MON', label='left', closed='left')
    records = []
    for week_start, group in weekly:
    if len(group) < 1000:
    continue
    
    week = {
        'week_start': week_start,
        'week_end': group.index[-1],
        'ret_total': group['close'].iloc[-1] / group['close'].iloc[0] - 1,
        'ret_mean': group['ret'].mean(),
        'ret_std': group['ret'].std(),
        'ret_skew': skew(group['ret'].dropna()),
        'ret_kurt': kurtosis(group['ret'].dropna()),
        'ret_max': group['ret'].max(),
        'ret_min': group['ret'].min(),
        'realized_vol': np.sqrt((group['ret'] ** 2).sum()),
        'drawdown': group['close'].div(group['close'].cummax()).sub(1).min(),
        'runup': group['close'].div(group['close'].cummin()).sub(1).max(),
        'vol_total': group['volume'].sum(),
        'vol_mean': group['volume'].mean(),
        'vol_std': group['volume'].std(),
        'vol_max': group['volume'].max(),
        'vol_skew': skew(group['volume'].dropna()),
        'ret_over_1pct': (group['ret'].abs() > 0.01).sum()
    }
    records.append(week)
    return pd.DataFrame(records)
