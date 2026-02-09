#!/usr/bin/env python3
"""
Ray feature builder for K-Means regime training.

Reads daily-partitioned 1s klines from S3, builds 60s bars, computes rolling
"regime features" per day (with a tail window for continuity), and writes results back to S3.

Input example (daily partitioned):
  s3://eth-kbarczak/binance_vision/market=spot/type=klines/symbol=ETHUSDT/interval=1s/date=2024-01-01/*.parquet
or:
  .../dt=2024-01-01/*.parquet
or:
  .../2024-01-01/*.parquet

Outputs:
  minute bars (optional):
    s3://.../minute_bars_60s/symbol=ETHUSDT/date=YYYY-MM-DD/part-0.parquet
  regime features:
    s3://.../regime_features_60s/symbol=ETHUSDT/date=YYYY-MM-DD/part-0.parquet

Requires:
  pip install -U ray pandas numpy pyarrow s3fs

Run (example):
  python build_regime_features_ray.py \
    --ray-address auto \
    --in-s3 "s3://eth-kbarczak/binance_vision/market=spot/type=klines/symbol=ETHUSDT/interval=1s/" \
    --symbol ETHUSDT \
    --out-features-s3 "s3://eth-kbarczak/derived/regime_features_60s/" \
    --out-minute-s3 "s3://eth-kbarczak/derived/minute_bars_60s/" \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --workers 64
"""

from __future__ import annotations

import argparse
import os
import re
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray

# ---------- Rolling windows (in minutes) ----------
W24 = 24 * 60
W168 = 168 * 60

# Regime feature columns (used downstream for KMeans)
FEATURE_COLS = [
    "trend24",
    "chop24",
    "taker_imb24",
    "taker_share24",
    "rv_burst_1h",
    "rv24_over_rv168",
    "vol24_over_vol168",
    "rv24_z168",
]

# ---------- S3 utilities ----------
def _norm_s3_prefix(p: str) -> str:
    if not p.startswith("s3://"):
        raise ValueError(f"Expected s3:// prefix, got: {p}")
    return p if p.endswith("/") else p + "/"


def _split_s3(p: str) -> Tuple[str, str]:
    # returns (bucket, key_prefix)
    m = re.match(r"^s3://([^/]+)/?(.*)$", p)
    if not m:
        raise ValueError(f"Invalid S3 path: {p}")
    return m.group(1), m.group(2)


def _s3_glob(prefix: str, pattern: str) -> List[str]:
    """
    Glob using s3fs. Returns s3://... paths.
    """
    import s3fs  

    fs = s3fs.S3FileSystem(anon=False)
    pref = _norm_s3_prefix(prefix)
    bucket, key_pref = _split_s3(pref)
    path = f"{bucket}/{key_pref}{pattern}"
    matches = fs.glob(path)
    return [f"s3://{m}" for m in matches]


def _write_parquet_s3(df: pd.DataFrame, out_s3_dir: str, filename: str = "part-0.parquet") -> str:
    import s3fs  # noqa: F401

    fs = s3fs.S3FileSystem(anon=False)
    out_s3_dir = _norm_s3_prefix(out_s3_dir)
    out_path = out_s3_dir + filename
    with fs.open(out_path, "wb") as f:
        df.to_parquet(f, index=False)
    return out_path


def _read_parquet_s3(paths: List[str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p, columns=columns) for p in paths]
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


# ---------- Date helpers ----------
def date_range(start_date: str, end_date: str) -> List[str]:
    d0 = datetime.fromisoformat(start_date).date()
    d1 = datetime.fromisoformat(end_date).date()
    if d1 < d0:
        raise ValueError("end_date must be >= start_date")
    out = []
    d = d0
    while d <= d1:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


# ---------- 1s -> 60s bars ----------
def _detect_day_partition_paths(in_s3: str, day: str) -> List[str]:
    """
    Try common partition patterns:
      date=YYYY-MM-DD/
      dt=YYYY-MM-DD/
      YYYY-MM-DD/
    and return parquet file paths.
    """
    in_s3 = _norm_s3_prefix(in_s3)
    candidates = [
        f"date={day}/*.parquet",
        f"dt={day}/*.parquet",
        f"{day}/*.parquet",
        f"date={day}/**/*.parquet",
        f"dt={day}/**/*.parquet",
    ]
    for pat in candidates:
        hits = _s3_glob(in_s3, pat)
        if hits:
            return sorted(hits)
    return []


def _coerce_ts(df: pd.DataFrame) -> pd.Series:
    """
    Make a UTC timestamp series from common Binance kline schemas.
    Prefer open_time (ms) if present, else ts, else open_time-like in seconds.
    """
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        if ts.notna().any():
            return ts

    for c in ["open_time", "openTime", "start_time", "startTime", "timestamp", "openTimeMs"]:
        if c in df.columns:
            v = df[c]
            if pd.api.types.is_numeric_dtype(v):
                median = float(np.nanmedian(v.to_numpy(dtype="float64")))
                unit = "ms" if median > 10_000_000_000 else "s"
                return pd.to_datetime(v, unit=unit, utc=True, errors="coerce")
            return pd.to_datetime(v, utc=True, errors="coerce")

    raise ValueError(f"Could not detect timestamp column. Found: {list(df.columns)}")


def _col_pick(df: pd.DataFrame, *names: str) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def build_minute_bars_from_1s(df_1s: pd.DataFrame) -> pd.DataFrame:
    df = df_1s.copy()
    df["ts"] = _coerce_ts(df)
    df = df.dropna(subset=["ts"]).sort_values("ts")

    c_open = _col_pick(df, "open")
    c_high = _col_pick(df, "high")
    c_low = _col_pick(df, "low")
    c_close = _col_pick(df, "close")
    c_vol = _col_pick(df, "volume", "base_volume")

    if not all([c_open, c_high, c_low, c_close, c_vol]):
        missing = [n for n, c in [("open", c_open), ("high", c_high), ("low", c_low),
                                 ("close", c_close), ("volume", c_vol)] if c is None]
        raise ValueError(f"Missing required kline cols: {missing}. Found: {list(df.columns)}")

    c_taker_base = _col_pick(
        df,
        "taker_buy_base_asset_volume",
        "taker_buy_base_vol",
        "taker_base",
        "takerBuyBase",
    )

    for c in [c_open, c_high, c_low, c_close, c_vol]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if c_taker_base:
        df[c_taker_base] = pd.to_numeric(df[c_taker_base], errors="coerce")
    else:
        df["__taker_base__"] = np.nan
        c_taker_base = "__taker_base__"

    df["minute"] = df["ts"].dt.floor("60s")

    # Robust aggregation (avoids scalar-dict DataFrame construction issues)
    out = (
        df.groupby("minute", sort=True)
          .agg(
              open=(c_open, "first"),
              high=(c_high, "max"),
              low=(c_low, "min"),
              close=(c_close, "last"),
              volume=(c_vol, "sum"),
              taker_base=(c_taker_base, "sum"),
          )
          .reset_index()
          .rename(columns={"minute": "ts"})
    )

    logc = np.log(out["close"])
    out["ret1m"] = logc.diff().fillna(0.0)
    out["ret2_1m"] = out["ret1m"] ** 2
    out["absret1m"] = out["ret1m"].abs()
    return out


# ---------- Regime features ----------
def compute_regime_features_minute(minute_df: pd.DataFrame) -> pd.DataFrame:
    df = minute_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")

    rv24 = df["ret2_1m"].rolling(W24, min_periods=6 * 60).sum() ** 0.5
    mar24 = df["absret1m"].rolling(W24, min_periods=6 * 60).mean()
    chop24 = mar24 / rv24.replace(0, np.nan)

    logc = np.log(df["close"])
    trend24 = (logc - logc.shift(W24)) / (W24 * 60)

    taker_share = df["taker_base"] / df["volume"].replace(0, np.nan)
    taker_imb = (2.0 * df["taker_base"] - df["volume"]) / df["volume"].replace(0, np.nan)
    taker_share24 = taker_share.rolling(W24, min_periods=6 * 60).mean()
    taker_imb24 = taker_imb.rolling(W24, min_periods=6 * 60).mean()

    rv1h_var = df["ret2_1m"].rolling(60, min_periods=20).sum()
    rv24_var = df["ret2_1m"].rolling(W24, min_periods=6 * 60).sum()
    rv_burst_1h = rv1h_var / rv24_var.replace(0, np.nan)

    rv168 = df["ret2_1m"].rolling(W168, min_periods=3 * 24 * 60).sum() ** 0.5
    rv24_over_rv168 = rv24 / rv168.replace(0, np.nan)

    vol24 = df["volume"].rolling(W24, min_periods=6 * 60).sum()
    vol168 = df["volume"].rolling(W168, min_periods=3 * 24 * 60).sum()
    vol24_over_vol168 = vol24 / vol168.replace(0, np.nan)

    rv24_mean168 = rv24.rolling(W168, min_periods=3 * 24 * 60).mean()
    rv24_std168 = rv24.rolling(W168, min_periods=3 * 24 * 60).std()
    rv24_z168 = (rv24 - rv24_mean168) / rv24_std168.replace(0, np.nan)

    feat = pd.DataFrame(
        {
            "trend24": trend24,
            "chop24": chop24,
            "taker_imb24": taker_imb24,
            "taker_share24": taker_share24,
            "rv_burst_1h": rv_burst_1h,
            "rv24_over_rv168": rv24_over_rv168,
            "vol24_over_vol168": vol24_over_vol168,
            "rv24_z168": rv24_z168,
        },
        index=df.index,
    ).replace([np.inf, -np.inf], np.nan)

    # IMPORTANT: keep ts as a column
    return feat.reset_index().rename(columns={"index": "ts"})


# ---------- Ray tasks ----------
@ray.remote(num_cpus=1)
def build_minute_and_features_for_day(
    *,
    in_s3: str,
    out_minute_s3: Optional[str],
    out_features_s3: str,
    symbol: str,
    day: str,
    tail_days: int = 7,
) -> Dict[str, str]:
    """
    For a given day:
      - load raw 1s for [day - tail_days ... day]
      - build minute bars
      - compute regime features over the whole window
      - slice only "day" rows
      - write features to S3
      - optionally also write minute bars for "day"
    """
    try:
        d = datetime.strptime(day, "%Y-%m-%d").date()
        dates = [(d - timedelta(days=i)).isoformat() for i in range(tail_days, 0, -1)] + [day]

        minute_parts: List[pd.DataFrame] = []
        used_inputs: List[str] = []

        for dt in dates:
            paths = _detect_day_partition_paths(in_s3, dt)
            if not paths:
                continue
            used_inputs.append(f"{dt}:{len(paths)}")
            raw = _read_parquet_s3(paths)
            if raw.empty:
                continue

            minute = build_minute_bars_from_1s(raw)

            day_start = pd.Timestamp(dt, tz="UTC")
            day_end = day_start + pd.Timedelta(days=1)
            minute = minute[(minute["ts"] >= day_start) & (minute["ts"] < day_end)].copy()

            if not minute.empty:
                minute_parts.append(minute)

        if not minute_parts:
            return {"day": day, "status": "no_data", "inputs": ",".join(used_inputs)}

        minute_all = pd.concat(minute_parts, ignore_index=True)
        minute_all["ts"] = pd.to_datetime(minute_all["ts"], utc=True)
        minute_all = minute_all.sort_values("ts")

        feat_all = compute_regime_features_minute(minute_all)

        day_start = pd.Timestamp(day, tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)
        feat_day = feat_all[(feat_all["ts"] >= day_start) & (feat_all["ts"] < day_end)].copy()

        out_feat_dir = _norm_s3_prefix(out_features_s3) + f"symbol={symbol}/date={day}/"
        feat_path = _write_parquet_s3(feat_day, out_feat_dir, "part-0.parquet")

        out_min_path = ""
        if out_minute_s3:
            minute_day = minute_all[(minute_all["ts"] >= day_start) & (minute_all["ts"] < day_end)].copy()
            out_min_dir = _norm_s3_prefix(out_minute_s3) + f"symbol={symbol}/date={day}/"
            out_min_path = _write_parquet_s3(minute_day, out_min_dir, "part-0.parquet")

        return {
            "day": day,
            "status": "ok",
            "features": feat_path,
            "minute": out_min_path,
            "n_feat_rows": str(len(feat_day)),
            "inputs": ",".join(used_inputs),
        }

    except Exception as e:
        return {
            "day": day,
            "status": "error",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def _bounded_gather(obj_refs: List[ray.ObjectRef], max_in_flight: int) -> List[dict]:
    """
    Avoid huge driver memory spikes by draining results gradually.
    """
    pending = list(obj_refs)
    out: List[dict] = []
    in_flight: List[ray.ObjectRef] = []

    while pending and len(in_flight) < max_in_flight:
        in_flight.append(pending.pop())

    while in_flight:
        done, in_flight = ray.wait(in_flight, num_returns=1)
        out.extend(ray.get(done))
        while pending and len(in_flight) < max_in_flight:
            in_flight.append(pending.pop())

    return out


# ---------- Main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ray-address", default="auto", help='Ray address ("auto" for cluster)')
    ap.add_argument("--in-s3", required=True, help="S3 prefix containing daily-partitioned 1s data")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    ap.add_argument("--out-features-s3", required=True, help="S3 prefix for regime features output")
    ap.add_argument("--out-minute-s3", default="", help="(Optional) S3 prefix for minute bars output")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tail-days", type=int, default=7, help="How many prior days to include for rolling continuity")
    ap.add_argument("--workers", type=int, default=64, help="Max in-flight Ray tasks (driver-side throttle)")
    ap.add_argument("--print-tracebacks", action="store_true", help="Print full tracebacks for first errors")
    args = ap.parse_args()

    in_s3 = _norm_s3_prefix(args.in_s3)
    out_features_s3 = _norm_s3_prefix(args.out_features_s3)
    out_minute_s3 = _norm_s3_prefix(args.out_minute_s3) if args.out_minute_s3 else None

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    days = date_range(args.start_date, args.end_date)
    refs = [
        build_minute_and_features_for_day.remote(
            in_s3=in_s3,
            out_minute_s3=out_minute_s3,
            out_features_s3=out_features_s3,
            symbol=args.symbol,
            day=d,
            tail_days=args.tail_days,
        )
        for d in days
    ]

    results = _bounded_gather(refs, max_in_flight=max(1, int(args.workers)))

    ok = [r for r in results if r.get("status") == "ok"]
    errs = [r for r in results if r.get("status") == "error"]
    nodata = [r for r in results if r.get("status") == "no_data"]

    print("Done.")
    print(f"  ok={len(ok)}")
    print(f"  error={len(errs)}")
    print(f"  no_data={len(nodata)}")

    if errs:
        print("\nFirst errors:")
        for r in errs[:10]:
            print(" ", r.get("day"), "-", r.get("error"))
            if args.print_tracebacks:
                print(r.get("traceback", ""))

    if nodata:
        print("\nNo-data days (first 20):", [r.get("day") for r in nodata[:20]])

    if ok:
        print("\nExample output:")
        print("  features:", ok[0].get("features"))
        if out_minute_s3:
            print("  minute:", ok[0].get("minute"))

    ray.shutdown()


if __name__ == "__main__":
    main()