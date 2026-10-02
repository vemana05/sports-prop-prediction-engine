import os

import duckdb
import pytest

DEFAULT_DB_PATH = "data/warehouse.duckdb"


@pytest.fixture(scope="session")
def conn():
    path = os.environ.get("DUCKDB_PATH", DEFAULT_DB_PATH)
    if not os.path.exists(path):
        pytest.skip(f"no warehouse at {path} — run the ingestion runner first")
    connection = duckdb.connect(path, read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def nfl_conn():
    path = os.environ.get("NFL_DUCKDB_PATH", "data/nfl_warehouse.duckdb")
    if not os.path.exists(path):
        pytest.skip(f"no NFL warehouse at {path} — run backend.ingest.nfl_nflverse first")
    connection = duckdb.connect(path, read_only=True)
    yield connection
    connection.close()
