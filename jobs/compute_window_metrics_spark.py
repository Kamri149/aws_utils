# jobs/compute_window_metrics_spark.py
import argparse
import os

from pyspark.sql import SparkSession, functions as F

from trading.config import STORAGE
from trading.windows import add_returns, add_time_buckets
from trading.metrics_registry import METRICS
from trading.snapshots import compute_input_snapshot_id
from trading.run_logging import log_window_metric_run

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--start_dt", required=True)  # 2025-10-01
    p.add_argument("--end_dt", required=True)    # 2025-11-01
    p.add_argument("--window", default="1 week")
    p.add_argument("--granularity", default="weekly")
    return p.parse_args()

def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName(f"window_metrics_{args.exchange}_{args.asset}")
        .getOrCreate()
    )

    code_version = os.getenv("CODE_VERSION", "dev")  # e.g. git commit hash

    base = f"s3://{STORAGE.bucket}/{STORAGE.curated_ohlcv_prefix}"
    path = (
        f"{base}/exchange={args.exchange}/asset={args.asset}/"
        f"dt>={args.start_dt}/dt<={args.end_dt}/"
    )

    df = spark.read.parquet(path)

    # Compute snapshot id from Spark's view of the input files
    input_snapshot_id = compute_input_snapshot_id(df)

    df = add_returns(df)
    df = add_time_buckets(df, window=args.window)

    aggs = [
        F.min("timestamp").alias("window_start"),
        F.max("timestamp").alias("window_end"),
    ]

    for m in METRICS.values():
        if args.granularity in m.granularity:
            aggs.append(m.spark_agg(df))

    grouped = (
        df.groupBy("exchange", "asset", "bucket_start", "bucket_end")
          .agg(*aggs)
    )

    out_base = f"s3://{STORAGE.bucket}/{STORAGE.window_metrics_prefix}"
    out_path = (
        f"{out_base}/timeframe=1w/exchange={args.exchange}/asset={args.asset}/"
    )

    (
        grouped
        .withColumn("run_date", F.current_date())
        .repartition("run_date")
        .write
        .mode("append")
        .partitionBy("run_date")
        .parquet(out_path)
    )

    # Log the run metadata (one record per job)
    log_window_metric_run(
        exchange=args.exchange,
        asset=args.asset,
        timeframe="1w",
        start_dt=args.start_dt,
        end_dt=args.end_dt,
        input_snapshot_id=input_snapshot_id,
        code_version=code_version,
    )

    spark.stop()

if __name__ == "__main__":
    main()
