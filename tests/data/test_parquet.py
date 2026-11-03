"""
Data quality checks on the Parquet request log (logs/requests.parquet).
Uses DuckDB SQL to verify: schema, null-safety, uniqueness, value ranges, type drift.

The fixture redirects logging to a temp file and seeds five known rows so tests
are fully self-contained and don't depend on prior API calls.
"""
import os

os.environ.setdefault("LLM_PROVIDER", "fake")

import pyarrow.parquet as pq
import pytest
import duckdb

from app.logging_ import log_analysis


@pytest.fixture(scope="module")
def log_path(tmp_path_factory):
    """Redirect the log module to a temp Parquet file; seed 5 known rows."""
    import app.logging_ as log_mod

    tmp = tmp_path_factory.mktemp("data_tests")
    original_dir = log_mod.LOG_DIR
    original_file = log_mod.LOG_FILE
    log_mod.LOG_DIR = tmp
    log_mod.LOG_FILE = tmp / "requests.parquet"

    log_analysis(endpoint="/analyze", match_pct=80.0, keyword_coverage_pct=65.0,
                 num_requirements=5, num_met=4, num_partial=0, num_missing=1)
    log_analysis(endpoint="/analyze", match_pct=50.0, keyword_coverage_pct=40.0,
                 num_requirements=4, num_met=2, num_partial=1, num_missing=1)
    log_analysis(endpoint="/rewrite")
    log_analysis(endpoint="/chat")
    log_analysis(endpoint="/analyze", match_pct=0.0, error="simulated error")

    path = log_mod.LOG_FILE
    yield path

    log_mod.LOG_DIR = original_dir
    log_mod.LOG_FILE = original_file


def _q(log_path, sql: str) -> int:
    """Run a scalar COUNT query against the Parquet file."""
    posix = str(log_path).replace("\\", "/")
    return duckdb.sql(sql.format(p=posix)).fetchone()[0]


# ── Existence ─────────────────────────────────────────────────────────────────

def test_parquet_file_created(log_path):
    assert log_path.exists(), "log_analysis must create the Parquet file"


def test_row_count(log_path):
    n = _q(log_path, "SELECT COUNT(*) FROM read_parquet('{p}')")
    assert n == 5, f"Expected 5 seeded rows, got {n}"


# ── Null checks ───────────────────────────────────────────────────────────────

def test_no_null_request_ids(log_path):
    n = _q(log_path, "SELECT COUNT(*) FROM read_parquet('{p}') WHERE request_id IS NULL")
    assert n == 0, "request_id must never be NULL"


def test_no_null_timestamps(log_path):
    n = _q(log_path, "SELECT COUNT(*) FROM read_parquet('{p}') WHERE timestamp IS NULL")
    assert n == 0, "timestamp must never be NULL"


def test_no_null_endpoints(log_path):
    n = _q(log_path, "SELECT COUNT(*) FROM read_parquet('{p}') WHERE endpoint IS NULL")
    assert n == 0


# ── Uniqueness ────────────────────────────────────────────────────────────────

def test_no_duplicate_request_ids(log_path):
    n = _q(
        log_path,
        "SELECT COUNT(*) FROM ("
        "  SELECT request_id FROM read_parquet('{p}')"
        "  GROUP BY request_id HAVING COUNT(*) > 1)",
    )
    assert n == 0, "request_id must be unique per request"


# ── Value ranges ──────────────────────────────────────────────────────────────

def test_match_pct_in_range(log_path):
    n = _q(
        log_path,
        "SELECT COUNT(*) FROM read_parquet('{p}') WHERE match_pct < 0 OR match_pct > 100",
    )
    assert n == 0, "match_pct must be 0–100"


def test_keyword_coverage_in_range(log_path):
    n = _q(
        log_path,
        "SELECT COUNT(*) FROM read_parquet('{p}') "
        "WHERE keyword_coverage_pct < 0 OR keyword_coverage_pct > 100",
    )
    assert n == 0, "keyword_coverage_pct must be 0–100"


def test_counts_non_negative(log_path):
    n = _q(
        log_path,
        "SELECT COUNT(*) FROM read_parquet('{p}') "
        "WHERE num_met < 0 OR num_partial < 0 OR num_missing < 0 OR num_requirements < 0",
    )
    assert n == 0


# ── Schema / type drift ───────────────────────────────────────────────────────

def test_schema_has_required_columns(log_path):
    schema = pq.read_schema(log_path)
    required = {
        "request_id", "timestamp", "endpoint",
        "match_pct", "keyword_coverage_pct",
        "num_requirements", "num_met", "num_partial", "num_missing",
        "error",
    }
    missing = required - set(schema.names)
    assert not missing, f"Parquet schema missing columns: {missing}"


def test_no_schema_drift(log_path):
    schema = pq.read_schema(log_path)
    types = {f.name: str(f.type) for f in schema}
    assert types["request_id"] == "string"
    assert types["timestamp"] == "string"
    assert types["endpoint"] == "string"
    assert types["match_pct"] == "double"
    assert types["num_requirements"] == "int64"
    assert types["error"] == "string"


# ── Endpoint values ───────────────────────────────────────────────────────────

def test_endpoint_values_are_valid(log_path):
    posix = str(log_path).replace("\\", "/")
    rows = duckdb.sql(
        f"SELECT DISTINCT endpoint FROM read_parquet('{posix}')"
    ).fetchall()
    valid = {"/analyze", "/rewrite", "/chat", "/health"}
    for (ep,) in rows:
        assert ep in valid, f"Unexpected endpoint value in log: {ep!r}"
