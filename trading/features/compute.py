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
