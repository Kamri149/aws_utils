from __future__ import annotations
import io
import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Sequence, Iterable


def fetch_binance_vision_klines(
    symbol: str = "ETHUSDT",
    interval: str = "1s",
    start_date: str = "2023-01-09",
    end_date: str = "2023-01-15",
    market: str = "spot",
    save_dir: str = "./binance_data"
) -> pd.DataFrame:
    """
    Download historical klines from Binance Vision (daily zipped CSVs).

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. 'ETHUSDT'.
    interval : str
        Candle interval, e.g. '1s', '1m', '1h', '1d'.
    start_date, end_date : str
        Inclusive date range (YYYY-MM-DD).
    market : {'spot','futures'}
        Data type to pull.
    save_dir : str
        Local folder to cache .zip files.

    Returns
    -------
    pd.DataFrame with all klines merged and sorted by open_time (UTC).
    """
    base_url = "https://data.binance.vision"
    prefix = f"data/{market}/daily/klines/{symbol.upper()}/{interval}"
    os.makedirs(save_dir, exist_ok=True)

    def daterange(start, end):
        d = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        while d <= e:
            yield d.strftime("%Y-%m-%d")
            d += timedelta(days=1)

    frames = []
    for d_str in daterange(start_date, end_date):
        fname = f"{symbol.upper()}-{interval}-{d_str}.zip"
        url = f"{base_url}/data/{market}/daily/klines/{symbol.upper()}/{interval}/{fname}"
        local_zip = os.path.join(save_dir, fname)

        # Download if missing
        if not os.path.exists(local_zip):
            print(f"⬇️  {url}")
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"⚠️  Skipped {d_str}, HTTP {r.status_code}")
                continue
            with open(local_zip, "wb") as f:
                f.write(r.content)

        # Extract CSV from zip in memory
        with zipfile.ZipFile(local_zip, "r") as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csvfile:
                df = pd.read_csv(csvfile, header=None)
                df.columns = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_base", "taker_quote", "ignore"
                ]
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                num_cols = ["open","high","low","close","volume",
                            "quote_volume","taker_base","taker_quote"]
                df[num_cols] = df[num_cols].astype(float)
                df["trades"] = df["trades"].astype(int)
                frames.append(df.drop(columns=["ignore"]))

    if not frames:
        raise ValueError("No data downloaded for given range.")

    df_all = pd.concat(frames).sort_values("open_time").reset_index(drop=True)
    print(f"✅ Loaded {len(df_all)} rows from {start_date} → {end_date}")
    return df_all


def load_binance_vision_klines_local(
    symbol: str = "ETHUSDT",
    interval: str = "1s",
    start_date: str = "2023-01-09",
    end_date: str = "2023-01-15",
    save_dir: str = "./binance_data",
    allow_missing_days: bool = True,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load Binance Vision daily kline ZIPs from a local folder (no downloading).
    Returns a DataFrame in the same format as the fetch_binance_vision_klines() function.

    Expected local filenames:
        {SYMBOL}-{INTERVAL}-{YYYY-MM-DD}.zip
    Each zip contains a single CSV with 12 columns:
        open_time, open, high, low, close, volume,
        close_time, quote_volume, trades, taker_base, taker_quote, ignore

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. 'ETHUSDT'.
    interval : str
        Candle interval, e.g. '1s', '1m', '1h', '1d'.
    start_date, end_date : str
        Inclusive date range (YYYY-MM-DD).
    save_dir : str
        Folder containing the downloaded ZIP files.
    allow_missing_days : bool
        If True, skip missing ZIPs; if False, raise an error if any day is missing.
    start_ts, end_ts : Optional[str]
        Optional UTC datetime filters applied after loading (e.g. '2023-01-10 08:30:00').

    Returns
    -------
    pd.DataFrame
        Columns:
          open_time (UTC datetime), open, high, low, close, volume,
          close_time (UTC datetime), quote_volume, trades, taker_base, taker_quote
        Sorted by open_time.
    """

    def daterange(start: str, end: str) -> Sequence[str]:
        d = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        out = []
        while d <= e:
            out.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        return out

    symbol_u = symbol.upper()
    os.makedirs(save_dir, exist_ok=True)

    frames = []
    missing = []

    for d_str in daterange(start_date, end_date):
        fname = f"{symbol_u}-{interval}-{d_str}.zip"
        local_zip = os.path.join(save_dir, fname)

        if not os.path.exists(local_zip):
            missing.append(local_zip)
            if allow_missing_days:
                continue
            raise FileNotFoundError(f"Missing required ZIP: {local_zip}")

        with zipfile.ZipFile(local_zip, "r") as zf:
            # Binance Vision daily kline zips typically contain exactly one CSV
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found inside {local_zip}")
            csv_name = csv_names[0]

            with zf.open(csv_name) as csvfile:
                df = pd.read_csv(csvfile, header=None)
                if df.shape[1] < 12:
                    raise ValueError(f"Unexpected column count in {local_zip}: got {df.shape[1]}, expected 12")

                df.columns = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_base", "taker_quote", "ignore",
                ]

                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

                num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]
                df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(float)
                df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)

                frames.append(df.drop(columns=["ignore"]))

    if not frames:
        raise ValueError(f"No data loaded for given range from {save_dir}: {start_date} → {end_date}")

    df_all = pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)

    # Optional datetime filtering
    if start_ts is not None:
        start_dt = pd.to_datetime(start_ts, utc=True)
        df_all = df_all[df_all["open_time"] >= start_dt]
    if end_ts is not None:
        end_dt = pd.to_datetime(end_ts, utc=True)
        df_all = df_all[df_all["open_time"] <= end_dt]

    df_all = df_all.reset_index(drop=True)

    if missing and allow_missing_days:
        print(f"⚠️  Skipped {len(missing)} missing day ZIP(s) in {start_date} → {end_date}")

    print(f"✅ Loaded {len(df_all)} rows from local ZIPs: {start_date} → {end_date}")
    return df_all


def load_any_datafile_local(
    path: str,
    *,
    filetype: Optional[str] = None,
    compression: Optional[str] = None,
    csv_kwargs: Optional[dict] = None,
    columns: Optional[Sequence[str]] = None,
    require_exact_columns: bool = True,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    ts_col: Optional[str] = None,
    ts_unit: Optional[str] = None,
    tz_utc: bool = True,
) -> pd.DataFrame:
    """
    Generic local data loader with ZIP support and optional column naming.

    New:
      - `columns`: explicitly set column names after load
      - `require_exact_columns`: enforce strict column count matching

    The rest of the behaviour mirrors your Binance local loader:
      - path-based format detection
      - ZIP container handling
      - optional timestamp filtering
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    csv_kwargs = csv_kwargs or {}

    def infer_type(p: str) -> str:
        ext = os.path.splitext(p)[1].lower().lstrip(".")
        mapping = {
            "csv": "csv",
            "tsv": "tsv",
            "parquet": "parquet",
            "pq": "parquet",
            "json": "json",
            "jsonl": "jsonl",
            "ndjson": "jsonl",
            "feather": "feather",
            "pkl": "pickle",
            "pickle": "pickle",
        }
        if ext not in mapping:
            raise ValueError(f"Unsupported file extension: .{ext}")
        return mapping[ext]

    def read_by_type(ft: str, p: str, data: Optional[bytes] = None) -> pd.DataFrame:
        if ft in {"csv", "tsv"}:
            sep = "\t" if ft == "tsv" else ","
            if data is None:
                return pd.read_csv(p, sep=sep, compression=compression, **csv_kwargs)
            return pd.read_csv(io.BytesIO(data), sep=sep, **csv_kwargs)

        if ft == "parquet":
            return pd.read_parquet(io.BytesIO(data)) if data else pd.read_parquet(p)

        if ft in {"json", "jsonl"}:
            lines = ft == "jsonl"
            return (
                pd.read_json(io.BytesIO(data), lines=lines)
                if data
                else pd.read_json(p, lines=lines)
            )

        if ft == "feather":
            return pd.read_feather(io.BytesIO(data)) if data else pd.read_feather(p)

        if ft == "pickle":
            return pd.read_pickle(io.BytesIO(data)) if data else pd.read_pickle(p)

        raise ValueError(f"Unsupported filetype: {ft}")

    # ---------- load ----------
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            inner = [
                n for n in zf.namelist()
                if os.path.splitext(n)[1].lower() in {
                    ".csv", ".tsv", ".parquet", ".pq",
                    ".json", ".jsonl", ".ndjson",
                    ".feather", ".pkl", ".pickle"
                }
            ]
            if not inner:
                raise ValueError(f"No supported data files found in ZIP: {path}")

            inner_name = inner[0]
            ft = filetype or infer_type(inner_name)

            with zf.open(inner_name) as f:
                df = read_by_type(ft, inner_name, data=f.read())
    else:
        ft = filetype or infer_type(path)
        df = read_by_type(ft, path)

    # ---------- column handling (NEW) ----------
    if columns is not None:
        if require_exact_columns and len(columns) != df.shape[1]:
            raise ValueError(
                f"Column count mismatch: "
                f"{len(columns)} names provided, but file has {df.shape[1]} columns"
            )
        df.columns = list(columns)

    # ---------- optional timestamp filtering ----------
    if ts_col is not None and (start_ts or end_ts):
        if ts_col not in df.columns:
            raise ValueError(f"ts_col '{ts_col}' not found in DataFrame")

        if ts_unit is not None:
            df[ts_col] = pd.to_datetime(df[ts_col], unit=ts_unit, utc=True)
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=tz_utc, errors="coerce")

        if start_ts:
            df = df[df[ts_col] >= pd.to_datetime(start_ts, utc=True)]
        if end_ts:
            df = df[df[ts_col] <= pd.to_datetime(end_ts, utc=True)]

        df = df.reset_index(drop=True)

    return df
