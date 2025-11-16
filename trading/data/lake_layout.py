from trading.config.storage import STORAGE

def raw_trades_prefix(asset_class: str = "crypto",
                      source: str = "cryptolake") -> str:
    return (
        f"{STORAGE.project_root}/"
        f"{STORAGE.raw_root}/"
        f"{asset_class}/source={source}"
    )

def curated_ohlcv_prefix(asset_class: str = "crypto") -> str:
    return (
        f"{STORAGE.project_root}/"
        f"{STORAGE.curated_root}/"
        f"{asset_class}/ohlcv_1s"
    )

def window_metrics_prefix(entity: str,
                          asset_class: str = "crypto",
                          timeframe: str = "1w") -> str:
    return (
        f"{STORAGE.project_root}/"
        f"{STORAGE.metrics_root}/window_metrics/"
        f"entity={entity}/asset_class={asset_class}/timeframe={timeframe}"
    )

def metric_definitions_path() -> str:
    return (
        f"{STORAGE.project_root}/"
        f"{STORAGE.meta_root}/metric_definitions/metric_definitions.parquet"
    )

def metric_runs_root() -> str:
    return (
        f"{STORAGE.project_root}/"
        f"{STORAGE.meta_root}/metric_runs"
    )
