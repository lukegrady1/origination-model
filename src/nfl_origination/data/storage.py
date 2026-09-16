"""Parquet persistence helpers and read-only DuckDB views over saved artifacts."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from nfl_origination.errors import MissingDataError


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise MissingDataError(f"artifact not found: {path}")
    return pd.read_parquet(path)


def duckdb_views(tables: dict[str, Path]) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with one read-only view per parquet artifact.

    DuckDB is an analytical convenience over the Parquet source of truth, never a second copy.
    """
    con = duckdb.connect(database=":memory:")
    for name, path in tables.items():
        if not Path(path).exists():
            continue
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{Path(path).as_posix()}')")
    return con
