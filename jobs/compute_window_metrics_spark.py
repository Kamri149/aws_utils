# jobs/compute_window_metrics_spark.py
import argparse
from pyspark.sql import SparkSession, functions as F
from trading.config import STORAGE
from trading.windows import add_returns, add_time_buckets
from trading.metrics_registry import METRICS

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--start_dt", required=True)  # e.g. 2025-10-01
    p.add_argument("--end_dt", required=True)    # e.g. 2025-11-01
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

    # ---- Load 1s OHLCV from curated layer ----
    base = f"s3://{STORAGE.bucket}/{STORAGE.curated_ohlcv_prefix}"
    path = (
        f"{base}/exchange={args.exchange}/asset={args.asset}/"
        f"dt>={args.start_dt}/dt<={args.end_dt}/"
    )

    df = spark.read.parquet(path)

    # Ensure required columns exist: open_time, close, volume, etc.
    df = add_returns(df)
    df = add_time_buckets(df, window=args.window)

    # ---- Build aggregations from registry ----
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

    # ---- Write out wide window metrics ----
    out_base = f"s3://{STORAGE.bucket}/{STORAGE.window_metrics_prefix}"
    out_path = (
        f"{out_base}/timeframe=1w/exchange={args.exchange}/asset={args.asset}/"
    )

    (
        grouped
        .withColumn("run_date", F.current_date())
        .repartition("run_date")   # simple partition
        .write
        .mode("append")
        .partitionBy("run_date")
        .parquet(out_path)
    )

    spark.stop()

if __name__ == "__main__":
    main()
