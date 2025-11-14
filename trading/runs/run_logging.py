# trading/run_logging.py
import uuid
import datetime
import pandas as pd
import io
import boto3
from trading.config import STORAGE

s3 = boto3.client("s3")

def log_window_metric_run(
    exchange: str,
    asset: str,
    timeframe: str,
    start_dt: str,
    end_dt: str,
    input_snapshot_id: str,
    code_version: str,
):
    run_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat()

    record = {
        "run_id": run_id,
        "exchange": exchange,
        "asset": asset,
        "timeframe": timeframe,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "input_snapshot_id": input_snapshot_id,
        "code_version": code_version,
        "created_at": now,
    }

    df = pd.DataFrame([record])

    key = f"{STORAGE.metric_runs_path.rstrip('/')}/dt={start_dt}/run_{run_id}.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    s3.put_object(Bucket=STORAGE.bucket, Key=key, Body=buf.getvalue())

    return run_id
