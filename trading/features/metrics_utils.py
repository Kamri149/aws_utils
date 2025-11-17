import hashlib
from typing import Dict
from .metric_definition import MetricDefinition

def metrics_set_fingerprint(metrics: Dict[str, MetricDefinition]) -> str:
    parts = [f"{m.metric_id}:{m.version}" for m in metrics.values()]
    parts.sort()
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()
