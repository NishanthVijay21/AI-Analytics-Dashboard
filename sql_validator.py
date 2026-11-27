"""
sql_validator.py
Parse-based validation for LLM-generated SQL (replaces the old regex filter).

What it enforces
  1. Exactly ONE statement, and its root is SELECT / UNION / INTERSECT / EXCEPT
     (a leading WITH is just part of a SELECT in the AST).
  2. No write/DDL/command nodes anywhere in the tree (INSERT, COPY, ATTACH, ...).
  3. No file/env/catalog access: table functions such as read_csv('/etc/passwd'),
     scalar functions such as read_text / getenv, schema-qualified names such as
     information_schema.tables, and `FROM 'file.csv'` replacement scans.
  4. Table allowlist: every table referenced must be a loaded table (or a CTE
     defined in the same query).
  5. Column allowlist (best effort): every referenced column must exist on one of
     the referenced tables, or be an alias defined in the query.

A string literal such as  WHERE status = 'UPDATE'  is fine: we inspect the AST,
not the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

MAX_SQL_CHARS = 20_000


class SQLValidationError(ValueError):
    """Raised when generated SQL is not allowed to run."""


def _node_types(names: Iterable[str]) -> tuple:
    # getattr-guarded so a sqlglot version that renames a node doesn't crash import
    return tuple(getattr(exp, n) for n in names if hasattr(exp, n))


_ALLOWED_ROOTS = _node_types(["Select", "Union", "Intersect", "Except"])

_FORBIDDEN_NODES = _node_types([
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "AlterColumn",
    "Command", "Copy", "Attach", "Detach", "Pragma", "Set", "Use", "Merge", "TruncateTable",
    "Transaction", "Commit", "Rollback", "Install", "LoadData", "Export", "Grant",
    "Into",            # SELECT ... INTO new_table
    "Describe", "Show", "Summarize",
])

# Never allowed anywhere (scalar or table position)
_FORBIDDEN_FUNCS_EXACT = {
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "parquet_metadata",
    "parquet_schema", "parquet_file_metadata", "parquet_kv_metadata",
    "read_json", "read_json_auto", "read_json_objects", "read_ndjson", "read_ndjson_objects",
    "read_text", "read_blob", "glob", "sniff_csv", "csv_scan",
    "query", "query_table", "getenv", "current_setting", "which_secret",
    "load", "install", "iceberg_scan", "delta_scan", "sqlite_scan", "postgres_scan",
}
_FORBIDDEN_FUNC_PREFIXES = ("duckdb_", "pragma_", "read_", "st_read")

# Table functions that are harmless (no I/O) and sometimes useful (date spines)
_SAFE_TABLE_FUNCS = {"range", "generate_series", "unnest"}


@dataclass
class ValidatedQuery:
    sql: str                      # cleaned original text (no trailing ';')
    tables: set = field(default_factory=set)
    columns: set = field(default_factory=set)


def _func_name(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower()
    return None


def _is_forbidden_func(name: str) -> bool:
    return name in _FORBIDDEN_FUNCS_EXACT or name.startswith(_FORBIDDEN_FUNC_PREFIXES)


def clean_sql(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip()


def validate_sql(
    sql: str,
    allowed_tables: Iterable[str],
    allowed_columns: Mapping[str, Iterable[str]] | None = None,
) -> ValidatedQuery:
    """
    Raise SQLValidationError if `sql` is not a safe, read-only query over
    `allowed_tables`. `allowed_columns` maps table name -> column names; pass
    None to skip the column check.
    """
    text = clean_sql(sql)
    if not text:
        raise SQLValidationError("Empty SQL.")
    if len(text) > MAX_SQL_CHARS:
        raise SQLValidationError("SQL is too long.")

    try:
        statements = [s for s in sqlglot.parse(text, read="duckdb") if s is not None]
    except SqlglotError as e:
        raise SQLValidationError(f"Could not parse SQL: {str(e)[:200]}") from e

    if len(statements) != 1:
        raise SQLValidationError("Only a single SQL statement is allowed.")
    tree = statements[0]

    # unwrap "(SELECT ...)" at the root
    while isinstance(tree, exp.Subquery):
        tree = tree.this
    if not isinstance(tree, _ALLOWED_ROOTS):
        raise SQLValidationError(
            f"Only SELECT / WITH ... SELECT queries are allowed (got {type(tree).__name__.upper()})."
        )

    for node in tree.walk():
        node = node[0] if isinstance(node, tuple) else node
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLValidationError(f"{type(node).__name__.upper()} is not allowed in a query.")
        fname = _func_name(node)
        if fname and _is_forbidden_func(fname):
            raise SQLValidationError(f"Function {fname}() is not allowed.")

    allowed = {t.lower() for t in allowed_tables}
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    used_tables: set[str] = set()

    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            # table-valued function (read_csv was already rejected above)
            fname = _func_name(t.this) if isinstance(t.this, exp.Expression) else None
            if fname is None and isinstance(t.this, (exp.GenerateSeries,)):
                fname = "generate_series"
            if fname not in _SAFE_TABLE_FUNCS:
                raise SQLValidationError("Table functions are not allowed in FROM.")
            continue
        if t.args.get("db") or t.args.get("catalog"):
            raise SQLValidationError("Schema-qualified table names are not allowed.")
        name = t.name.lower()
        if name in cte_names:
            continue
        if name not in allowed:
            raise SQLValidationError(f"Table '{t.name}' is not one of the loaded tables.")
        used_tables.add(name)

    used_columns: set[str] = set()
    if allowed_columns is not None:
        cols_by_table = {k.lower(): {c.lower() for c in v} for k, v in allowed_columns.items()}
        visible = set().union(*(cols_by_table.get(t, set()) for t in used_tables)) if used_tables else set()

        defined = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
        defined |= {i.name.lower() for i in tree.find_all(exp.TableAlias)}
        for al in tree.find_all(exp.TableAlias):
            defined |= {c.name.lower() for c in al.args.get("columns") or []}
        defined |= {i.name.lower() for lam in tree.find_all(exp.Lambda)
                    for i in lam.find_all(exp.Identifier)}

        for col in tree.find_all(exp.Column):
            if isinstance(col.this, exp.Star):
                continue
            name = col.name.lower()
            used_columns.add(name)
            if name in visible or name in defined:
                continue
            raise SQLValidationError(
                f"Column '{col.name}' does not exist in the referenced tables."
            )

    return ValidatedQuery(sql=text, tables=used_tables, columns=used_columns)
