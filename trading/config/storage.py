# trading/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class StorageConfig:
    bucket: str = "trading-kbarczak"     # or keep your existing bucket
    project_root: str = "data"         # <— updated, generic namespace

    raw_root: str = "raw"
    curated_root: str = "curated"
    metrics_root: str = "metrics"
    meta_root: str = "meta"

STORAGE = StorageConfig()


# @dataclass(frozen=True)
# class StorageConfig:
#     bucket: str = "trading-lake"
#     curated_ohlcv_prefix: str = "curated/ohlcv_1s"
#     window_metrics_prefix: str = "metrics/window_metrics"
#     metric_definitions_path: str = "meta/metric_definitions/metric_definitions.parquet"
#     metric_runs_path: str = "meta/metric_runs/metric_runs.parquet"

# STORAGE = StorageConfig()
