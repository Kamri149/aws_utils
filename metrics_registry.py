# trading/metrics_registry.py
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

# -------- Spark UDFs --------
@pandas_udf(DoubleType())
def skew_udf(x: pd.Series) -> float:
    return float(skew(x.dropna())) if len(x.dropna()) > 0 else np.nan

@pandas_udf(DoubleType())
def kurtosis_udf(x: pd.Series) -> float:
    return float(kurtosis(x.dropna())) if len(x.dropna()) > 0 else np.nan


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    description: str
    granularity: List[str]
    version: str
    spark_agg: Callable[[Any], Any]          # df -> Column
    pandas_fn: Optional[Callable[[pd.DataFrame], Any]] = None


# -------- Registry content (your metrics) --------
METRICS: Dict[str, MetricDefinition] = {}

def _register(metric: MetricDefinition):
    METRICS[metric.metric_id] = metric
    return metric

_register(MetricDefinition(
    metric_id="ret_total",
    description="Total return over the window",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: ((F.last("close") / F.first("close")) - 1).alias("ret_total"),
    pandas_fn=lambda df: df["close"].iloc[-1] / df["close"].iloc[0] - 1
))

_register(MetricDefinition(
    metric_id="ret_mean",
    description="Mean return over the window",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.mean("ret").alias("ret_mean"),
    pandas_fn=lambda df: df["ret"].mean()
))

_register(MetricDefinition(
    metric_id="ret_std",
    description="Standard deviation of returns",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.stddev("ret").alias("ret_std"),
    pandas_fn=lambda df: df["ret"].std()
))

_register(MetricDefinition(
    metric_id="ret_skew",
    description="Skewness of returns",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: skew_udf("ret").alias("ret_skew"),
    pandas_fn=lambda df: skew(df["ret"].dropna())
))

_register(MetricDefinition(
    metric_id="ret_kurt",
    description="Kurtosis of returns",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: kurtosis_udf("ret").alias("ret_kurt"),
    pandas_fn=lambda df: kurtosis(df["ret"].dropna())
))

_register(MetricDefinition(
    metric_id="ret_max",
    description="Max return",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.max("ret").alias("ret_max"),
    pandas_fn=lambda df: df["ret"].max()
))

_register(MetricDefinition(
    metric_id="ret_min",
    description="Min return",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.min("ret").alias("ret_min"),
    pandas_fn=lambda df: df["ret"].min()
))

_register(MetricDefinition(
    metric_id="realized_vol",
    description="Realized volatility over the window",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: F.sqrt(F.sum(F.pow("ret", 2))).alias("realized_vol"),
    pandas_fn=lambda df: np.sqrt((df["ret"] ** 2).sum())
))

_register(MetricDefinition(
    metric_id="vol_total",
    description="Total trading volume",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.sum("volume").alias("vol_total"),
    pandas_fn=lambda df: df["volume"].sum()
))

_register(MetricDefinition(
    metric_id="vol_mean",
    description="Mean trading volume",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.mean("volume").alias("vol_mean"),
    pandas_fn=lambda df: df["volume"].mean()
))

_register(MetricDefinition(
    metric_id="vol_std",
    description="Standard deviation of volume",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.stddev("volume").alias("vol_std"),
    pandas_fn=lambda df: df["volume"].std()
))

_register(MetricDefinition(
    metric_id="vol_max",
    description="Maximum trading volume",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.max("volume").alias("vol_max"),
    pandas_fn=lambda df: df["volume"].max()
))

_register(MetricDefinition(
    metric_id="vol_skew",
    description="Skewness of volume",
    granularity=["weekly"],
    version="1.0",
    spark_agg=lambda df: skew_udf("volume").alias("vol_skew"),
    pandas_fn=lambda df: skew(df["volume"].dropna())
))

_register(MetricDefinition(
    metric_id="ret_over_1pct",
    description="Count of returns exceeding 1% absolute",
    granularity=["daily", "weekly"],
    version="1.0",
    spark_agg=lambda df: F.sum(F.when(F.abs("ret") > 0.01, 1).otherwise(0)).alias("ret_over_1pct"),
    pandas_fn=lambda df: (df["ret"].abs() > 0.01).sum()
))


def compute_metrics_pandas(df: pd.DataFrame, granularity: str = "weekly") -> pd.Series:
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["abs_ret"] = df["ret"].abs()
    result = {}
    for m in METRICS.values():
        if m.pandas_fn is not None and granularity in m.granularity:
            try:
                result[m.metric_id] = m.pandas_fn(df)
            except Exception:
                result[m.metric_id] = np.nan
    return pd.Series(result)


def compute_metrics_spark(
    df,
    granularity: str = "weekly",
    *,
    time_col: str = "open_time",
    close_col: str = "close",
) -> Dict[str, float]:
    """
    Compute all registered metrics for a single window of data in Spark,
    following the same idea as `compute_metrics_pandas`.

    Assumes `df` contains at least:
      - a time column (default: open_time)
      - a close price column (default: close)
      - a volume column (for volume-based metrics)

    Returns a dict: {metric_id: value}.
    """

    # --- ensure returns are present, like in compute_metrics_pandas ---
    w = W.orderBy(time_col)
    df_with = (
        df
        .withColumn("ret", F.log(F.col(close_col) / F.lag(close_col).over(w)))
        .withColumn("abs_ret", F.abs(F.col("ret")))
    )

    # --- build expressions for all metrics valid at this granularity ---
    agg_exprs = []
    metric_ids_in_order = []

    for m in METRICS.values():
        if granularity in m.granularity and m.spark_agg is not None:
            # m.spark_agg expects the df (for flexibility), but most of your
            # lambdas just ignore it and reference columns by name.
            expr = m.spark_agg(df_with)
            agg_exprs.append(expr)
            metric_ids_in_order.append(m.metric_id)

    if not agg_exprs:
        raise ValueError(f"No Spark metrics registered for granularity={granularity!r}")

    # Single-row DataFrame with one column per metric
    metrics_row_df = df_with.agg(*agg_exprs)
    row = metrics_row_df.collect()[0]

    # Map metric_id -> value (using the aliases set in spark_agg)
    result: Dict[str, float] = {}
    for metric_id in metric_ids_in_order:
        # Column names are the aliases defined in spark_agg (e.g. "ret_total")
        # which we ensured match metric_id.
        result[metric_id] = row[metric_id]

    return result


# --------- Very simple fingerprint + run-record (local CSV) ---------
def compute_fingerprint(df: pd.DataFrame) -> str:
    sample = df.head(1000).to_string()
    return hashlib.sha256(sample.encode()).hexdigest()

def record_metric_run(metric_name, version, window_start, window_end,
                      fingerprint, registry_path="metric_runs.csv"):
    record = {
        "metric_name": metric_name,
        "version": version,
        "window_start": str(window_start),
        "window_end": str(window_end),
        "input_fingerprint": fingerprint,
        "run_timestamp": datetime.datetime.utcnow().isoformat()
    }
    out_df = pd.DataFrame([record])
    path = Path(registry_path)
    out_df.to_csv(path, mode="a", index=False, header=not path.exists())
