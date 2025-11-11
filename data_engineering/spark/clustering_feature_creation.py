from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

def compute_aggregated_features_spark(df, window='1 week'):
  
  df = df.withColumn("timestamp", F.col("open_time").cast("timestamp"))
  df = df.withColumn("ret", F.log(F.col("close") / F.lag("close").over(Window.orderBy("timestamp"))))
  df = df.withColumn("abs_ret", F.abs("ret"))
  
  df = df.withColumn("bucket", F.window("timestamp", window))
  
  agg = df.groupBy("bucket").agg(
  F.first("timestamp").alias("start"),
  F.last("timestamp").alias("end"),
  ((F.last("close") / F.first("close")) - 1).alias("ret_total"),
  F.mean("ret").alias("ret_mean"),
  F.stddev("ret").alias("ret_std"),
  F.max("ret").alias("ret_max"),
  F.min("ret").alias("ret_min"),
  F.sqrt(F.sum(F.pow("ret", 2))).alias("realized_vol"),
  F.sum("volume").alias("vol_total"),
  F.mean("volume").alias("vol_mean"),
  F.stddev("volume").alias("vol_std"),
  F.max("volume").alias("vol_max"),
  F.expr("percentile_approx(abs_ret, 0.99)").alias("abs_ret_99pct"),
  F.expr("sum(case when abs(ret) > 0.01 then 1 else 0 end)").alias("ret_over_1pct")
  )
  
  return agg
