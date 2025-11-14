# trading/snapshots.py
import hashlib
from pyspark.sql import DataFrame

def compute_input_snapshot_id(df: DataFrame) -> str:
    """
    Compute a stable fingerprint of the input data based on file paths.

    This ignores file contents but is usually enough for versioning runs:
    if your data files change, their paths or timestamps will too.
    """
    files = sorted(df.inputFiles())
    payload = "\n".join(files)
    return hashlib.sha256(payload.encode()).hexdigest()
