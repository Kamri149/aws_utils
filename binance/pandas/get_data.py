import io
import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta

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
