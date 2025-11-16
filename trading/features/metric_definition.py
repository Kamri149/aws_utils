from dataclasses import dataclass
from typing import Callable, List, Optional, Any
import pandas as pd

@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    description: str
    entity_type: str              # 'price', 'book', 'funding', 'oi', 'liq'
    granularity: List[str]
    version: str
    spark_agg: Callable[[Any], Any]         # df -> Column
    pandas_fn: Optional[Callable[[pd.DataFrame], Any]] = None
