# trading/export_metric_definitions.py
import json
import datetime
from dataclasses import asdict
from typing import List

import pandas as pd

from trading.config import STORAGE
from trading.metrics_registry import METRICS, MetricDefinition

def metric_to_record(m: MetricDefinition) -> dict:
    now = datetime.datetime.utcnow().isoformat()
    return {
        "metric_id": m.metric_id,
        "description": m.description,
        # store list as JSON so Glue can read it as string initially
        "granularity_json": json.dumps(m.granularity),
        "version": m.version,
        "active": True,
        "created_at": now,
        "updated_at": now,
    }

def export_metric_definitions(output_uri: str | None = None) -> str:
    """
    Export all MetricDefinition objects to a Parquet file (local or S3).

    If output_uri is None, uses STORAGE.metric_definitions_path on your S3 bucket.
    """
    if output_uri is None:
        # e.g. s3://trading-lake/meta/metric_definitions/metric_definitions.parquet
        output_uri = f"s3://{STORAGE.bucket}/{STORAGE.metric_definitions_path.lstrip('/')}"

    records: List[dict] = [metric_to_record(m) for m in METRICS.values()]
    df = pd.DataFrame(records)

    # Requires pyarrow + s3fs if you're writing to s3:// URI from Python
    df.to_parquet(output_uri, index=False)

    return output_uri

if __name__ == "__main__":
    uri = export_metric_definitions()
    print(f"Wrote metric definitions to: {uri}")
