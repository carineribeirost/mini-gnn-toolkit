"""
GPU memory helpers and cuDF/pandas dispatch.

Usage pattern:
    df = load_dataframe(path)   # returns cuDF if available and file is large, else pandas
    ...
    free_gpu_dataframes()       # call before JAX initialises to avoid VRAM conflict
"""
import gc
from pathlib import Path

# Size threshold above which cuDF is preferred (rows).
_CUDF_THRESHOLD = 100_000

try:
    import cudf as _cudf
    _CUDF_AVAILABLE = True
except ImportError:
    _CUDF_AVAILABLE = False


def cudf_available() -> bool:
    return _CUDF_AVAILABLE


def load_dataframe(path: str | Path, **kwargs):
    """Load a CSV into cuDF (GPU) or pandas depending on file size and availability."""
    path = Path(path)
    if _CUDF_AVAILABLE:
        n_rows = sum(1 for _ in open(path)) - 1  # subtract header
        if n_rows >= _CUDF_THRESHOLD:
            return _cudf.read_csv(path, **kwargs)
    import pandas as pd
    return pd.read_csv(path, **kwargs)


def to_pandas(df):
    """Convert a cuDF or pandas DataFrame to pandas."""
    if _CUDF_AVAILABLE and isinstance(df, _cudf.DataFrame):
        return df.to_pandas()
    return df


def free_gpu_dataframes() -> None:
    """
    Release cuDF GPU memory and run gc before JAX initialises.
    Call this at the end of any preprocessing step that used cuDF.
    """
    if not _CUDF_AVAILABLE:
        return
    try:
        import rmm
        gc.collect()
        rmm.reinitialize(pool_allocator=False)
    except Exception:
        gc.collect()
