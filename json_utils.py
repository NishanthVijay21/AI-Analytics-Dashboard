"""
json_utils.py
Shared helpers for safely turning pandas/DuckDB query results into JSON —
used anywhere a DataFrame becomes a Vega-Lite spec's "data.values" or gets
json.dumps'd directly (chart embedding, spec downloads, dashboard export).

DuckDB's DATE/TIMESTAMP columns come back from `.fetchdf()` as pandas
Timestamp objects (or, less often, plain `datetime.date`), and plain
`json.dumps` doesn't know how to serialize either — hence
"TypeError: Object of type Timestamp is not JSON serializable" whenever a
query touches a date/time column. These helpers convert those (and a few
other pandas/numpy scalar types) to plain JSON-safe values in one place, so
every call site doesn't need its own ad-hoc fix.
"""

import datetime
import decimal
import math

import numpy as np
import pandas as pd


def json_default(obj):
    """
    Pass as `default=` to json.dumps(...) to handle any pandas/numpy/date
    value that survives past `dataframe_records` (or wasn't run through it).
    """
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dataframe_records(df: pd.DataFrame) -> list:
    """
    `df.to_dict(orient="records")` but JSON/Vega-Lite-safe: datetime-like
    columns are converted to ISO 8601 strings up front (Vega-Lite auto-parses
    those when a field is encoded as "temporal"), and any other stray
    pandas/numpy scalar per-cell is normalized as a fallback.
    """
    if df is None or df.empty:
        return []

    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif pd.api.types.is_timedelta64_dtype(safe[col]):
            safe[col] = safe[col].astype(str)

    records = safe.to_dict(orient="records")

    # Belt-and-suspenders: catches object-dtype columns holding raw
    # datetime.date/Timestamp/Decimal/numpy-scalar values that the dtype
    # checks above wouldn't have touched.
    for row in records:
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date, datetime.time)):
                row[k] = v.isoformat()
            elif isinstance(v, np.integer):
                row[k] = int(v)
            elif isinstance(v, np.floating):
                fv = float(v)
                row[k] = None if math.isnan(fv) else fv
            elif isinstance(v, np.bool_):
                row[k] = bool(v)
            elif isinstance(v, decimal.Decimal):
                row[k] = float(v)
    return records
