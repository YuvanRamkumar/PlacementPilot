"""
Append one Parquet row per request to logs/requests.parquet.
No raw PII is stored — only redacted text and metadata.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "requests.parquet"

_SCHEMA = pa.schema([
    pa.field("request_id", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("endpoint", pa.string()),
    pa.field("match_pct", pa.float64()),
    pa.field("keyword_coverage_pct", pa.float64()),
    pa.field("num_requirements", pa.int64()),
    pa.field("num_met", pa.int64()),
    pa.field("num_partial", pa.int64()),
    pa.field("num_missing", pa.int64()),
    pa.field("error", pa.string()),
])


def log_analysis(
    *,
    endpoint: str = "/analyze",
    match_pct: float = 0.0,
    keyword_coverage_pct: float = 0.0,
    num_requirements: int = 0,
    num_met: int = 0,
    num_partial: int = 0,
    num_missing: int = 0,
    error: str = "",
) -> str:
    """Write one row to the Parquet log. Returns the request_id."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    request_id = str(uuid.uuid4())
    row = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "match_pct": match_pct,
        "keyword_coverage_pct": keyword_coverage_pct,
        "num_requirements": num_requirements,
        "num_met": num_met,
        "num_partial": num_partial,
        "num_missing": num_missing,
        "error": error,
    }
    table = pa.Table.from_pydict({k: [v] for k, v in row.items()}, schema=_SCHEMA)

    if LOG_FILE.exists():
        existing = pq.read_table(LOG_FILE)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, LOG_FILE)
    else:
        pq.write_table(table, LOG_FILE)

    return request_id
