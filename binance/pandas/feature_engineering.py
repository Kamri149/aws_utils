import numpy as np
import pandas as pd


def add_v1_kline_features_1s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a v1 feature set from Binance 1s kline columns:
      open_time, open, high, low, close, volume, close_time,
      quote_volume, trades, taker_base, taker_quote

    Computes the previously listed v1_additional_features plus the base
    micro_return/micro_range used by your original features.

    Assumptions:
      - 1 row == 1 second (row-based rolling windows)
      - df is a single asset; if multi-asset, groupby before calling
      - open_time is sortable (int ms or datetime)
      - open/high/low/close/volume are numeric

    Returns a copy of df with new feature columns added.
    """
    out = df.copy()

    # ---- sort & ensure numeric ----
    out = out.sort_values("open_time")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # ---- base micro features (1s only) ----
    eps = 1e-12  # guard against divide-by-zero
    out["micro_return"] = np.log((out["close"] + eps) / (out["open"] + eps))
    out["micro_range"] = out["high"] - out["low"]

    # ---- close-to-close dynamics ----
    out["log_close_return_1s"] = np.log((out["close"] + eps) / (out["close"].shift(1) + eps))
    out["abs_close_return_1s"] = out["log_close_return_1s"].abs()

    # rolling extremes of close-to-close return
    out["rolling_max_return_60s"] = out["log_close_return_1s"].rolling(60).max()
    out["rolling_min_return_60s"] = out["log_close_return_1s"].rolling(60).min()

    # ---- volatility structure (requires your original roll vols if you want ratios)
    # We'll compute the needed roll vols here to keep this self-contained.
    out["roll_vol_5s"] = out["micro_return"].rolling(5).std()
    out["roll_vol_60s"] = out["micro_return"].rolling(60).std()
    out["roll_vol_300s"] = out["micro_return"].rolling(300).std()

    out["vol_ratio_5s_60s"] = out["roll_vol_5s"] / (out["roll_vol_60s"] + eps)
    out["vol_ratio_60s_300s"] = out["roll_vol_60s"] / (out["roll_vol_300s"] + eps)
    out["vol_acceleration_60s"] = out["roll_vol_60s"] - out["roll_vol_300s"]

    # ---- volume structure ----
    vol_mean_60 = out["volume"].rolling(60).mean()
    vol_std_60 = out["volume"].rolling(60).std()

    out["volume_zscore_60s"] = (out["volume"] - vol_mean_60) / (vol_std_60 + eps)
    out["roll_volume_5s"] = out["volume"].rolling(5).sum()
    out["roll_volume_60s"] = out["volume"].rolling(60).sum()
    out["volume_ratio_5s_60s"] = out["roll_volume_5s"] / (out["roll_volume_60s"] + eps)

    out["volume_acceleration_60s"] = out["roll_volume_5s"] - out["volume"].rolling(60).mean()

    # ---- candle structure ----
    oc_max = out[["open", "close"]].max(axis=1)
    oc_min = out[["open", "close"]].min(axis=1)

    out["body_size"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - oc_max
    out["lower_wick"] = oc_min - out["low"]
    out["body_to_range_ratio"] = out["body_size"] / (out["micro_range"].abs() + eps)

    # ---- range dynamics ----
    range_mean_60 = out["micro_range"].rolling(60).mean()
    range_std_60 = out["micro_range"].rolling(60).std()
    out["range_zscore_60s"] = (out["micro_range"] - range_mean_60) / (range_std_60 + eps)

    out["range_ratio_5s_60s"] = out["micro_range"].rolling(5).mean() / (range_mean_60 + eps)

    # compression: 1 - percentile_rank over last 300s (higher => more compressed)
    # This is heavier than other features but still OK in pandas for typical sizes.
    def _pct_rank_last(x: np.ndarray) -> float:
        # percentile rank of the last element within the window [0,1]
        last = x[-1]
        return float(np.mean(x <= last))

    pr_300 = out["micro_range"].rolling(300).apply(_pct_rank_last, raw=True)
    out["range_compression_300s"] = 1.0 - pr_300

    # ---- event memory (seconds since last spike) ----
    # spikes defined exactly like earlier: current > 3 * rolling mean over 300s
    price_spike = out["micro_range"] > 3.0 * out["micro_range"].rolling(300).mean()
    volume_spike = out["volume"] > 3.0 * out["volume"].rolling(300).mean()

    # Create a "time index" in seconds based on row order (since 1 row == 1 second)
    t = np.arange(len(out), dtype=np.int64)

    # last index where spike was True (forward-filled)
    last_price_spike_idx = pd.Series(np.where(price_spike.to_numpy(), t, np.nan)).ffill().to_numpy()
    last_volume_spike_idx = pd.Series(np.where(volume_spike.to_numpy(), t, np.nan)).ffill().to_numpy()

    out["secs_since_price_spike"] = t - last_price_spike_idx
    out["secs_since_volume_spike"] = t - last_volume_spike_idx

    # If no spike has happened yet, ffill is NaN => keep NaN (or set to a large number)
    out.loc[np.isnan(last_price_spike_idx), "secs_since_price_spike"] = np.nan
    out.loc[np.isnan(last_volume_spike_idx), "secs_since_volume_spike"] = np.nan

    # ---- optional pressure proxy ----
    out["signed_return_volume"] = out["micro_return"] * out["volume"]

    return out
