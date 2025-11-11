# Spark dependencies
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf

# Python-native libs
import pandas as pd
from scipy.stats import skew, kurtosis

@F.pandas_udf(DoubleType())
def skew_udf(x: pd.Series) -> float:
  return skew(x.dropna())

@F.pandas_udf(DoubleType())
def kurtosis_udf(x: pd.Series) -> float:
  return kurtosis(x.dropna())

def compute_aggregated_features_spark(df, window='1 week'):
  
  df = df.withColumn("timestamp", F.col("open_time").cast("timestamp"))
  window_spec = Window.orderBy("timestamp")
  
  df = df.withColumn("ret", F.log(F.col("close") / F.lag("close").over(window_spec)))
  df = df.withColumn("abs_ret", F.abs("ret"))
  
  df = df.withColumn("bucket", F.window("timestamp", window))
  
  agg = df.groupBy("bucket").agg(
    F.min("timestamp").alias("week_start"),
    F.max("timestamp").alias("week_end"),
    ((F.last("close") / F.first("close")) - 1).alias("ret_total"),
    F.mean("ret").alias("ret_mean"),
    F.stddev("ret").alias("ret_std"),
    skew_udf("ret").alias("ret_skew"),
    kurtosis_udf("ret").alias("ret_kurt"),
    F.max("ret").alias("ret_max"),
    F.min("ret").alias("ret_min"),
    F.sqrt(F.sum(F.pow("ret", 2))).alias("realized_vol"),
    F.sum("volume").alias("vol_total"),
    F.mean("volume").alias("vol_mean"),
    F.stddev("volume").alias("vol_std"),
    F.max("volume").alias("vol_max"),
    skew_udf("volume").alias("vol_skew"),
    F.sum(F.when(F.abs("ret") > 0.01, 1).otherwise(0)).alias("ret_over_1pct")
  )
  return agg
