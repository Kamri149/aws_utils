# trading/windows.py
from pyspark.sql import functions as F, Window as W
from pyspark.sql import DataFrame

def add_returns(df: DataFrame) -> DataFrame:
    # Assumes cols: open_time, close
    df = df.withColumn("timestamp", F.col("open_time").cast("timestamp"))
    w = W.orderBy("timestamp")
    df = df.withColumn("ret", F.log(F.col("close") / F.lag("close").over(w)))
    df = df.withColumn("abs_ret", F.abs("ret"))
    return df

def add_time_buckets(df: DataFrame, window: str = "1 week") -> DataFrame:
    # Adds a struct column 'bucket' with start/end, and flat cols bucket_start/end
    df = df.withColumn("bucket", F.window("timestamp", window))
    df = df.withColumn("bucket_start", F.col("bucket").start)
    df = df.withColumn("bucket_end", F.col("bucket").end)
    return df
