# eth_weekly_feature_extractor.py
#
# Spark equivalent feature engineering function for ETH data

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import skew, kurtosis, zscore


def compute_weekly_features(df_1s, funding=None, oi=None, liq=None, book=None):
    """
    Parameters:
        df_1s: DataFrame with columns [open_time, open, high, low, close, volume]
        funding: Optional DataFrame with [timestamp, fundingRate]
        oi: Optional DataFrame with [timestamp, openInterest]
        liq: Optional DataFrame with [timestamp, side, price, qty, ...]
        book: Optional DataFrame with top-of-book data or order book snapshots
    Returns:
        weekly_features: DataFrame of features per week
    """
    df = df_1s.copy()
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    df.set_index('open_time', inplace=True)
    df = df.sort_index()

    # Compute returns
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_ret'] = df['ret'].abs()

    # Group by weekly windows (start Monday)
    weekly = df.resample('W-MON', label='left', closed='left')

    records = []
    for week_start, group in weekly:
        if len(group) < 1000:
            continue

        week = {}
        week['week_start'] = week_start
        week['week_end'] = group.index[-1]

        # Trend & returns
        week['ret_total'] = group['close'].iloc[-1] / group['close'].iloc[0] - 1
        week['ret_mean'] = group['ret'].mean()
        week['ret_std'] = group['ret'].std()
        week['ret_skew'] = skew(group['ret'].dropna())
        week['ret_kurt'] = kurtosis(group['ret'].dropna())
        week['ret_max'] = group['ret'].max()
        week['ret_min'] = group['ret'].min()

        # Volatility and drawdowns
        week['realized_vol'] = np.sqrt((group['ret']**2).sum())
        week['drawdown'] = group['close'].div(group['close'].cummax()).sub(1).min()
        week['runup'] = group['close'].div(group['close'].cummin()).sub(1).max()

        # Volume dynamics
        week['vol_total'] = group['volume'].sum()
        week['vol_mean'] = group['volume'].mean()
        week['vol_std'] = group['volume'].std()
        week['vol_max'] = group['volume'].max()
        week['vol_skew'] = skew(group['volume'].dropna())

        # Tail event count (extreme returns)
        threshold = 0.01
        week['ret_over_1pct'] = (group['ret'].abs() > threshold).sum()

        # Time-of-day seasonality
        group['hour'] = group.index.hour
        by_hour = group.groupby('hour')['abs_ret'].mean()
        week['vol_intraday_ratio'] = by_hour.max() / (by_hour.min() + 1e-8)
        week['vol_peak_hour'] = by_hour.idxmax()

        # Funding
        if funding is not None:
            f = funding.set_index('timestamp')
            f = f[week_start:group.index[-1]]
            if len(f):
                week['funding_mean'] = f['fundingRate'].mean()
                week['funding_std'] = f['fundingRate'].std()
                week['funding_range'] = f['fundingRate'].max() - f['fundingRate'].min()
                week['funding_sign_changes'] = (np.sign(f['fundingRate']).diff() != 0).sum()

        # Open interest
        if oi is not None:
            o = oi.set_index('timestamp')
            o = o[week_start:group.index[-1]]
            if len(o):
                week['oi_mean'] = o['openInterest'].mean()
                week['oi_change'] = o['openInterest'].iloc[-1] - o['openInterest'].iloc[0]
                week['oi_std'] = o['openInterest'].std()

        # Liquidations
        if liq is not None:
            l = liq.set_index('timestamp')
            l = l[week_start:group.index[-1]]
            if len(l):
                week['liq_total'] = l['qty'].sum()
                week['liq_count'] = len(l)
                week['liq_max'] = l['qty'].max()

        records.append(week)

    return pd.DataFrame(records)


def compute_cluster_alphas(df_weekly, fwd_returns, cluster_col='cluster'):
    """
    Parameters:
        df_weekly: DataFrame with per-week features and a 'cluster' column
        fwd_returns: Series of forward 1-week or 1-day returns indexed by week_start
        cluster_col: str name of the cluster column
    Returns:
        DataFrame with average forward returns and risk metrics per cluster
    """
    df = df_weekly.copy()
    df = df.merge(fwd_returns.rename("fwd_ret"), left_on="week_start", right_index=True, how="left")

    group = df.groupby(cluster_col)
    summary = group["fwd_ret"].agg([
        ("mean_return", "mean"),
        ("std_return", "std"),
        ("sharpe", lambda x: x.mean() / (x.std() + 1e-8)),
        ("skew", skew),
        ("kurtosis", kurtosis),
        ("count", "count")
    ])
    return summary.sort_values("mean_return", ascending=False)


# Example usage:
# df = pd.read_csv("eth_weekly_features.csv")
# fwd_ret = pd.read_csv("fwd_returns.csv", index_col=0, parse_dates=True)["fwd_ret"]
# alphas = compute_cluster_alphas(df, fwd_ret)
# print(alphas)


# Spark version for data engineering

def compute_aggregated_features_spark(df, window='1 week'):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import DoubleType

    df = df.withColumn("timestamp", F.col("open_time").cast("timestamp"))
    df = df.withColumn("ret", F.log(F.col("close") / F.lag("clos
