"""
data_engine.py  — v3
Multi-table DuckDB engine.

Changes from v2
  - CSV / TSV / Parquet are loaded straight into DuckDB from a temp file
    (no pandas round-trip); Excel still goes through pandas.
  - run_query(): AST validation (sql_validator), table + column allowlist,
    timeout (con.interrupt), row cap, execution timing, result cache.
  - Cache is invalidated by a per-engine `data_version` counter that bumps on any
    load / replace / merge / drop.
  - Schema refresh is a single scan per table instead of two queries per column.
  - merge_tables() goes through the same validator as run_query().
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from sql_validator import SQLValidationError, validate_sql


class QueryTimeout(RuntimeError):
    pass


def sanitize_ident(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if re.match(r"^\d", cleaned):
        cleaned = f"_{cleaned}"
    return cleaned or "dataset"


def _q(ident: str) -> str:
    """Quote an identifier for DuckDB."""
    return '"' + ident.replace('"', '""') + '"'


@dataclass
class QueryResult:
    df: pd.DataFrame
    elapsed_ms: float
    truncated: bool            # True if the row cap cut the result short
    total_rows: int | None     # full row count when truncated (None if unknown)
    cached: bool = False


class DataEngine:
    def __init__(
        self,
        memory_limit: str = "2GB",
        max_rows: int = 10_000,
        timeout_s: float = 30.0,
        cache_size: int = 64,
    ):
        self.con = duckdb.connect(database=":memory:", config={"memory_limit": memory_limit})
        self.tables: dict[str, dict] = {}
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self.data_version = 0
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ versioning / cache

    def _bump(self) -> None:
        self.data_version += 1
        self._cache.clear()

    # ------------------------------------------------------------------ loading

    def _unique_name(self, filename: str) -> str:
        base = sanitize_ident(Path(filename).stem)
        name, idx = base, 2
        while name in self.tables:
            name = f"{base}_{idx}"
            idx += 1
        return name

    @staticmethod
    def _spool_to_disk(uploaded_file, suffix: str) -> tuple[str, bool]:
        """Return (path, is_temp). Streamlit uploads are streamed to disk, not read into pandas."""
        if isinstance(uploaded_file, (str, os.PathLike)):
            return str(uploaded_file), False
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(uploaded_file, out, length=1024 * 1024)
        return path, True

    def load_file(self, uploaded_file, filename: str) -> str:
        """Load one file into a new DuckDB table. Returns the table name."""
        table_name = self._unique_name(filename)
        suffix = Path(filename.lower()).suffix

        if suffix in (".csv", ".tsv", ".txt", ".parquet"):
            path, is_temp = self._spool_to_disk(uploaded_file, suffix)
            try:
                reader = "read_parquet(?)" if suffix == ".parquet" else "read_csv_auto(?, sample_size=100000)"
                try:
                    self.con.execute(f"CREATE TABLE {_q(table_name)} AS SELECT * FROM {reader}", [path])
                except duckdb.Error:
                    if suffix == ".parquet":
                        raise
                    # messy CSV that DuckDB's sniffer rejects: fall back to pandas' lenient parser
                    df = pd.read_csv(path, sep=None, engine="python")
                    self._create_from_df(table_name, df)
            finally:
                if is_temp:
                    os.remove(path)
        elif suffix in (".xlsx", ".xls"):
            self._create_from_df(table_name, pd.read_excel(uploaded_file))
        else:
            raise ValueError("Unsupported file type — upload .csv, .tsv, .parquet or .xlsx.")

        self.tables[table_name] = {"filename": filename, "schema": [], "row_count": 0}
        self._refresh_schema(table_name)
        self._bump()
        return table_name

    def load_dataframe(self, df: pd.DataFrame, name: str, filename: str = "(dataframe)") -> str:
        """Used by tests / the eval harness."""
        table_name = self._unique_name(name)
        self._create_from_df(table_name, df)
        self.tables[table_name] = {"filename": filename, "schema": [], "row_count": 0}
        self._refresh_schema(table_name)
        self._bump()
        return table_name

    def _create_from_df(self, table_name: str, df: pd.DataFrame, replace: bool = False) -> None:
        self.con.register("_incoming", df)
        try:
            verb = "CREATE OR REPLACE TABLE" if replace else "CREATE TABLE"
            self.con.execute(f"{verb} {_q(table_name)} AS SELECT * FROM _incoming")
        finally:
            self.con.unregister("_incoming")

    def drop_table(self, table_name: str) -> None:
        if table_name in self.tables:
            self.con.execute(f"DROP TABLE IF EXISTS {_q(table_name)}")
            del self.tables[table_name]
            self._bump()

    # ------------------------------------------------------------------ preprocessing i/o

    def get_dataframe(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        sql = f"SELECT * FROM {_q(table_name)}" + (f" LIMIT {int(limit)}" if limit else "")
        return self.con.execute(sql).fetchdf()

    def sample_dataframe(self, table_name: str, n: int = 100_000) -> tuple[pd.DataFrame, bool]:
        """Return (df, was_sampled). Small tables come back whole."""
        total = self.tables[table_name]["row_count"]
        if total <= n:
            return self.get_dataframe(table_name), False
        df = self.con.execute(
            f"SELECT * FROM {_q(table_name)} USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE (42)"
        ).fetchdf()
        return df, True

    def replace_table_data(self, table_name: str, df: pd.DataFrame) -> None:
        """Overwrite (or create) a table from a DataFrame; refreshes schema + invalidates cache."""
        self._create_from_df(table_name, df, replace=True)
        if table_name not in self.tables:
            self.tables[table_name] = {"filename": "(derived)", "schema": [], "row_count": 0}
        self._refresh_schema(table_name)
        self._bump()

    # ------------------------------------------------------------------ merging / saving query results

    def merge_tables(self, sql: str, result_table: str) -> pd.DataFrame:
        """
        Persist a validated SELECT as a new table (used by the merge panel and
        'Save as new dataset'). No row cap: the whole result is materialised
        inside DuckDB, never pulled into pandas.
        """
        vq = validate_sql(sql, self.tables.keys(), self._allowed_columns())
        result_table = sanitize_ident(result_table)
        if result_table.lower() in vq.tables:
            raise ValueError(
                f"Result table name '{result_table}' is also a source table in this query — pick a new name."
            )
        self._check_single_select(vq.sql)
        self.con.execute(f"CREATE OR REPLACE TABLE {_q(result_table)} AS\n{vq.sql}")
        self.tables[result_table] = {"filename": "(merged)", "schema": [], "row_count": 0}
        self._refresh_schema(result_table)
        self._bump()
        return self.con.execute(f"SELECT * FROM {_q(result_table)} LIMIT 100").fetchdf()

    # ------------------------------------------------------------------ schema

    def _refresh_schema(self, table_name: str) -> None:
        """One table scan: row count, per-column non-null count and a sample value."""
        describe = self.con.execute(f"DESCRIBE {_q(table_name)}").fetchall()
        cols = [(r[0], r[1]) for r in describe]

        exprs = ["COUNT(*)"]
        for name, _ in cols:
            exprs.append(f"COUNT({_q(name)})")
            exprs.append(f"ANY_VALUE({_q(name)})")
        # fetchall (not fetchone) so the cursor is fully drained before the next statement
        row = self.con.execute(f"SELECT {', '.join(exprs)} FROM {_q(table_name)}").fetchall()[0]
        total = row[0]

        schema = []
        for i, (name, ctype) in enumerate(cols):
            non_null, sample = row[1 + 2 * i], row[2 + 2 * i]
            null_pct = round(100.0 * (total - non_null) / total, 1) if total else 0.0
            schema.append({"name": name, "type": ctype, "sample": sample, "null_pct": null_pct})

        self.tables[table_name]["schema"] = schema
        self.tables[table_name]["row_count"] = total

    def all_schemas_description(self) -> str:
        parts = []
        for tname, meta in self.tables.items():
            cols = ", ".join(f"{c['name']} ({c['type']})" for c in meta["schema"])
            parts.append(f"TABLE {tname}: {cols}")
        return "\n".join(parts)

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def _allowed_columns(self) -> dict[str, list[str]]:
        return {t: [c["name"] for c in m["schema"]] for t, m in self.tables.items()}

    # ------------------------------------------------------------------ query

    def is_safe(self, sql: str) -> bool:
        try:
            validate_sql(sql, self.tables.keys(), self._allowed_columns())
            return True
        except SQLValidationError:
            return False

    def _check_single_select(self, sql: str) -> None:
        """Second opinion from DuckDB's own parser (guards against parser disagreement)."""
        stmts = self.con.extract_statements(sql)
        if len(stmts) != 1 or stmts[0].type != duckdb.StatementType.SELECT:
            raise SQLValidationError("Only a single SELECT statement is allowed.")

    def _execute(self, sql: str, timeout_s: float) -> pd.DataFrame:
        cur = self.con.cursor()
        fired = threading.Event()

        def _kill():
            fired.set()
            cur.interrupt()

        timer = threading.Timer(timeout_s, _kill)
        timer.start()
        try:
            return cur.execute(sql).fetchdf()
        except duckdb.InterruptException as e:
            raise QueryTimeout(f"Query exceeded {timeout_s:g}s and was cancelled.") from e
        except duckdb.Error as e:
            if fired.is_set():
                raise QueryTimeout(f"Query exceeded {timeout_s:g}s and was cancelled.") from e
            raise
        finally:
            timer.cancel()
            cur.close()

    def run_query(
        self,
        sql: str,
        max_rows: int | None = None,
        timeout_s: float | None = None,
        use_cache: bool = True,
    ) -> QueryResult:
        """Validate -> (cache) -> run with timeout and row cap. Raises SQLValidationError / QueryTimeout / duckdb.Error."""
        vq = validate_sql(sql, self.tables.keys(), self._allowed_columns())
        self._check_single_select(vq.sql)

        limit = int(max_rows or self.max_rows)
        timeout = float(timeout_s or self.timeout_s)
        key = (" ".join(vq.sql.split()), limit, self.data_version)

        if use_cache and key in self._cache:
            self._cache.move_to_end(key)
            df, truncated, total = self._cache[key]
            return QueryResult(df.copy(), 0.0, truncated, total, cached=True)

        t0 = time.perf_counter()
        # fetch limit+1 rows so we can tell whether the result was cut off
        df = self._execute(f"SELECT * FROM (\n{vq.sql}\n) AS _q LIMIT {limit + 1}", timeout)
        truncated = len(df) > limit
        total = None
        if truncated:
            df = df.iloc[:limit].copy()
            try:
                total = int(self._execute(f"SELECT COUNT(*) AS n FROM (\n{vq.sql}\n) AS _q", timeout)["n"].iloc[0])
            except (QueryTimeout, duckdb.Error):
                total = None
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if use_cache:
            self._cache[key] = (df.copy(), truncated, total)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return QueryResult(df, elapsed_ms, truncated, total, cached=False)

    def query_internal(self, sql: str) -> pd.DataFrame:
        """For SQL the *app* generated itself (profiling, alert scans). Never pass LLM/user SQL here."""
        return self.con.execute(sql).fetchdf()
