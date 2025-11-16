# trading/features/metrics_price.py

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import hashlib
import datetime
from pathlib import Path

from trading.features.metric_definition import MetricDefinition  # adjust import if needed


# -------- Spark UDFs --------
@pandas_udf(DoubleType())
def skew_udf(x: pd.Series) -> float:
    return float(skew(x.dropna())) if len(x.dropna()) > 0 else np.nan


@pandas_udf(DoubleType())
def kurtosis_udf(x: pd.Series) -> float:
    return float(kurtosis(x.dropna())) if len(x.dropna()) > 0 else np.nan


# -------- Registry content (price metrics) --------
PRICE_METRICS: Dict[str, MetricDefinition] = {}


def _register(metric: MetricDefinition) -> MetricDefinition:
    PRICE_METRICS[metric.metric_id] = metric
    return metric


_register(MetricDefinition(
    metric_id="ret_total",
    description="Total return over the window",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: ((F.last("close") / F.first("close")) - 1).alias("ret_total"),
    pandas_fn=lambda df: df["close"].iloc[-1] / df["close"].iloc[0] - 1,
))

_register(MetricDefinition(
    metric_id="ret_mean",
    description="Mean return over the window",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.mean("ret").alias("ret_mean"),
    pandas_fn=lambda df: df["ret"].mean(),
))

_register(MetricDefinition(
    metric_id="ret_std",
    description="Standard deviation of returns",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.stddev("ret").alias("ret_std"),
    pandas_fn=lambda df: df["ret"].std(),
))

_register(MetricDefinition(
    metric_id="ret_skew",
    description="Skewness of returns",
    entity_type="price",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: skew_udf("ret").alias("ret_skew"),
    pandas_fn=lambda df: skew(df["ret"].dropna()),
))

_register(MetricDefinition(
    metric_id="ret_kurt",
    description="Kurtosis of returns",
    entity_type="price",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: kurtosis_udf("ret").alias("ret_kurt"),
    pandas_fn=lambda df: kurtosis(df["ret"].dropna()),
))

_register(MetricDefinition(
    metric_id="ret_max",
    description="Max return",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.max("ret").alias("ret_max"),
    pandas_fn=lambda df: df["ret"].max(),
))

_register(MetricDefinition(
    metric_id="ret_min",
    description="Min return",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.min("ret").alias("ret_min"),
    pandas_fn=lambda df: df["ret"].min(),
))

_register(MetricDefinition(
    metric_id="realized_vol",
    description="Realized volatility over the window",
    entity_type="price",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: F.sqrt(F.sum(F.pow("ret", 2))).alias("realized_vol"),
    pandas_fn=lambda df: np.sqrt((df["ret"] ** 2).sum()),
))

_register(MetricDefinition(
    metric_id="vol_total",
    description="Total trading volume",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.sum("volume").alias("vol_total"),
    pandas_fn=lambda df: df["volume"].sum(),
))

_register(MetricDefinition(
    metric_id="vol_mean",
    description="Mean trading volume",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.mean("volume").alias("vol_mean"),
    pandas_fn=lambda df: df["volume"].mean(),
))

_register(MetricDefinition(
    metric_id="vol_std",
    description="Standard deviation of volume",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.stddev("volume").alias("vol_std"),
    pandas_fn=lambda df: df["volume"].std(),
))

_register(MetricDefinition(
    metric_id="vol_max",
    description="Maximum trading volume",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.max("volume").alias("vol_max"),
    pandas_fn=lambda df: df["volume"].max(),
))

_register(MetricDefinition(
    metric_id="vol_skew",
    description="Skewness of volume",
    entity_type="price",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: skew_udf("volume").alias("vol_skew"),
    pandas_fn=lambda df: skew(df["volume"].dropna()),
))

_register(MetricDefinition(
    metric_id="ret_over_1pct",
    description="Count of returns exceeding 1% absolute",
    entity_type="price",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.sum(F.when(F.abs("ret") > 0.01, 1).otherwise(0)).alias("ret_over_1pct"),
    pandas_fn=lambda df: (df["ret"].abs() > 0.01).sum(),
))


# --------- Pandas convenience for clustering / local work ---------
def compute_price_metrics_pandas(df: pd.DataFrame,
                                 granularity: str = "weekly") -> pd.Series:
    """
    Compute all price metrics for a single window using pandas.
    Assumes df has at least 'close' and 'volume' columns.
    """
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["abs_ret"] = df["ret"].abs()

    result = {}
    for m in PRICE_METRICS.values():
        if m.pandas_fn is not None and granularity in m.granularity:
            try:
                result[m.metric_id] = m.pandas_fn(df)
            except Exception:
                result[m.metric_id] = np.nan
    return pd.Series(result)


# --------- Simple fingerprint + local run-record (optional) ---------
def compute_fingerprint(df: pd.DataFrame) -> str:
    """Quick-and-dirty fingerprint for small pandas DataFrames."""
    sample = df.head(1000).to_string()
    return hashlib.sha256(sample.encode()).hexdigest()


def record_metric_run(metric_name,
                      version,
                      window_start,
                      window_end,
                      fingerprint,
                      registry_path: str = "metric_runs.csv") -> None:
    """
    Local CSV-based run log (useful for notebooks / dev).
    For production use S3 + metric_runs table instead.
    """
    record = {
        "metric_name": metric_name,
        "version": version,
        "window_start": str(window_start),
        "window_end": str(window_end),
        "input_fingerprint": fingerprint,
        "run_timestamp": datetime.datetime.utcnow().isoformat(),
    }
    out_df = pd.DataFrame([record])
    path = Path(registry_path)
    out_df.to_csv(path, mode="a", index=False, header=not path.exists())
