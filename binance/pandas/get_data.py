import io
import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Sequence


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
