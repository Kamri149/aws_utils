import numpy as np
import pandas as pd


def add_v1_kline_features_1s_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar-aligned v1 feature builder for Binance 1s klines.

    Applies Option A:
      - Reindex to full 1-second calendar
      - Fill missing seconds conservatively
      - Compute v1 RL features (no sub-second data)

    Required columns:
      open_time, open, high, low, close, volume

    Assumptions:
      - Single asset
      - open_time is datetime-like or convertible
      - 1 row intended to represent 1 second
    """

    out = df.copy()

    # --- ensure datetime index ---
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.sort_values("open_time").set_index("open_time")

    # --- build full 1s calendar ---
    full_index = pd.date_range(
        start=out.index.min(),
        end=out.index.max(),
        freq="1S",
        tz=out.index.tz
    )

    out = out.reindex(full_index)

    # --- conservative filling for missing seconds ---
    eps = 1e-12

    # prices
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].fillna(out["close"].shift(1))
    out["high"] = out["high"].fillna(out[["open", "close"]].max(axis=1))
    out["low"] = out["low"].fillna(out[["open", "close"]].min(axis=1))

    # volume & optional kline fields
    for c in ["volume", "quote_volume", "trades", "taker_base", "taker_quote"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    # --- base micro features ---
    out["micro_return"] = np.log((out["close"] + eps) / (out["open"] + eps))
    out["micro_range"] = out["high"] - out["low"]

    # --- close-to-close dynamics ---
    out["log_close_return_1s"] = np.log((out["close"] + eps) / (out["close"].shift(1) + eps))
    out["abs_close_return_1s"] = out["log_close_return_1s"].abs()

    out["rolling_max_return_60s"] = out["log_close_return_1s"].rolling(60).max()
    out["rolling_min_return_60s"] = out["log_close_return_1s"].rolling(60).min()

    # --- volatility structure ---
    out["roll_vol_5s"] = out["micro_return"].rolling(5).std()
    out["roll_vol_60s"] = out["micro_return"].rolling(60).std()
    out["roll_vol_300s"] = out["micro_return"].rolling(300).std()

    out["vol_ratio_5s_60s"] = out["roll_vol_5s"] / (out["roll_vol_60s"] + eps)
    out["vol_ratio_60s_300s"] = out["roll_vol_60s"] / (out["roll_vol_300s"] + eps)
    out["vol_acceleration_60s"] = out["roll_vol_60s"] - out["roll_vol_300s"]

    # --- volume structure ---
    out["roll_volume_5s"] = out["volume"].rolling(5).sum()
    out["roll_volume_60s"] = out["volume"].rolling(60).sum()

    vol_mean_60 = out["volume"].rolling(60).mean()
    vol_std_60 = out["volume"].rolling(60).std()

    out["volume_zscore_60s"] = (out["volume"] - vol_mean_60) / (vol_std_60 + eps)
    out["volume_ratio_5s_60s"] = out["roll_volume_5s"] / (out["roll_volume_60s"] + eps)
    out["volume_acceleration_60s"] = out["roll_volume_5s"] - vol_mean_60

    # --- candle structure ---
    oc_max = out[["open", "close"]].max(axis=1)
    oc_min = out[["open", "close"]].min(axis=1)

    out["body_size"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - oc_max
    out["lower_wick"] = oc_min - out["low"]
    out["body_to_range_ratio"] = out["body_size"] / (out["micro_range"].abs() + eps)

    # --- range dynamics ---
    range_mean_60 = out["micro_range"].rolling(60).mean()
    range_std_60 = out["micro_range"].rolling(60).std()

    out["range_zscore_60s"] = (out["micro_range"] - range_mean_60) / (range_std_60 + eps)
    out["range_ratio_5s_60s"] = out["micro_range"].rolling(5).mean() / (range_mean_60 + eps)

    out["roll_vol_3600s"] = out["micro_return"].rolling(3600).std()
    out["vol_regime_60_3600"] = out["roll_vol_60s"] / (out["roll_vol_3600s"] + eps)

    # compression percentile over 300s
    def _pct_rank_last(x: np.ndarray) -> float:
        last = x[-1]
        return float(np.mean(x <= last))

    pr_300 = out["micro_range"].rolling(300).apply(_pct_rank_last, raw=True)
    out["range_compression_300s"] = 1.0 - pr_300

    # --- event flags ---
    out["is_price_spike"] = (
        out["micro_range"] > 3.0 * out["micro_range"].rolling(300).mean()
    )

    out["is_volume_spike"] = (
        out["volume"] > 3.0 * out["volume"].rolling(300).mean()
    )

    # --- event memory (seconds since last spike) ---
    t = np.arange(len(out), dtype=np.int64)

    price_spike = out["is_price_spike"].to_numpy()
    volume_spike = out["is_volume_spike"].to_numpy()

    last_price_idx = pd.Series(np.where(price_spike, t, np.nan)).ffill().to_numpy()
    last_volume_idx = pd.Series(np.where(volume_spike, t, np.nan)).ffill().to_numpy()

    out["secs_since_price_spike"] = t - last_price_idx
    out["secs_since_volume_spike"] = t - last_volume_idx

    out.loc[np.isnan(last_price_idx), "secs_since_price_spike"] = np.nan
    out.loc[np.isnan(last_volume_idx), "secs_since_volume_spike"] = np.nan

    # --- pressure proxy ---
    out["signed_return_volume"] = out["micro_return"] * out["volume"]

    return out.reset_index().rename(columns={"index": "open_time"})
