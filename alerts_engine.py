"""
alerts_engine.py
Proactive checks that run without the user asking:

  1. scan_table()      — automatic alerts: recent-period anomalies, missing values,
                         duplicates, extreme outliers.
  2. evaluate_rules()  — user-defined threshold rules ("alert me if SUM(amount) falls 20% month over month").

Everything is computed with DuckDB SQL (nothing is pulled into pandas except small
per-period aggregates), and nothing here calls the LLM or sends data anywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_engine import DataEngine, _q

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
_NUMERIC_PREFIXES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UBIGINT", "UINTEGER",
                     "USMALLINT", "UTINYINT", "FLOAT", "DOUBLE", "REAL", "DECIMAL")
_AVG_NAME = re.compile(r"(price|rate|pct|percent|ratio|score|age|avg|mean|temp|rating)", re.I)
_ID_NAME = re.compile(r"(^|_)id$|^id$", re.I)
GRAINS = ("day", "week", "month")


@dataclass
class Alert:
    severity: str       # high | medium | low | info
    kind: str           # anomaly | quality | rule
    table: str
    title: str
    detail: str = ""


# ═══════════════════════════════════════════════════════════ helpers

def _is_numeric(ctype: str) -> bool:
    return ctype.upper().startswith(_NUMERIC_PREFIXES)


def _is_temporal(ctype: str) -> bool:
    u = ctype.upper()
    return u == "DATE" or u.startswith("TIMESTAMP")


def _measure_columns(schema: list[dict]) -> list[str]:
    return [c["name"] for c in schema if _is_numeric(c["type"]) and not _ID_NAME.search(c["name"])]


def _choose_grain(min_d: pd.Timestamp, max_d: pd.Timestamp) -> str | None:
    span = (max_d - min_d).days
    if span >= 180:
        return "month"
    if span >= 56:
        return "week"
    if span >= 14:
        return "day"
    return None


def _period_is_partial(period_start: pd.Timestamp, max_date: pd.Timestamp, grain: str) -> bool:
    """True if the data ends before the last period is over (e.g. data stops on the 14th)."""
    if grain == "day":
        return False
    if grain == "week":
        end = period_start + pd.Timedelta(days=6)
    else:
        end = period_start + pd.offsets.MonthEnd(0)
    return max_date.normalize() < end.normalize()


def robust_anomaly(history, latest, z_thresh: float = 3.5, min_rel: float = 0.2, min_history: int = 5):
    """
    Compare `latest` to `history` using median/MAD (robust to earlier spikes).
    Returns {"z", "rel_change", "median"} if anomalous, else None.
    """
    h = np.asarray([x for x in history if x is not None and not np.isnan(x)], dtype=float)
    if len(h) < min_history or latest is None or np.isnan(latest):
        return None
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)))
    rel = (latest - med) / abs(med) if med else None

    if mad > 0:
        z = 0.6745 * (latest - med) / mad
    else:
        sd = float(np.std(h))
        z = (latest - med) / sd if sd > 0 else (np.inf if latest != med else 0.0)

    if abs(z) >= z_thresh and (rel is None or abs(rel) >= min_rel):
        return {"z": float(z), "rel_change": rel, "median": med}
    return None


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}"


# ═══════════════════════════════════════════════════════════ automatic scans

def _date_range(engine: DataEngine, table: str, col: str):
    r = engine.query_internal(f"SELECT MIN({_q(col)}) AS lo, MAX({_q(col)}) AS hi FROM {_q(table)}")
    lo, hi = r["lo"].iloc[0], r["hi"].iloc[0]
    if pd.isna(lo) or pd.isna(hi):
        return None, None
    return pd.Timestamp(lo), pd.Timestamp(hi)


def _anomaly_alerts(engine: DataEngine, table: str, schema: list[dict]) -> list[Alert]:
    date_cols = [c["name"] for c in schema if _is_temporal(c["type"])]
    measures = _measure_columns(schema)[:6]
    if not date_cols:
        return []
    dcol = date_cols[0]
    lo, hi = _date_range(engine, table, dcol)
    if lo is None:
        return []
    grain = _choose_grain(lo, hi)
    if grain is None:
        return []

    aggs = ["COUNT(*) AS \"__rows\""]
    for m in measures:
        fn = "AVG" if _AVG_NAME.search(m) else "SUM"
        aggs.append(f'{fn}({_q(m)}) AS {_q(m)}')
    df = engine.query_internal(
        f"SELECT date_trunc('{grain}', {_q(dcol)}) AS p, {', '.join(aggs)} "
        f"FROM {_q(table)} WHERE {_q(dcol)} IS NOT NULL GROUP BY 1 ORDER BY 1"
    )
    df["p"] = pd.to_datetime(df["p"])

    note = ""
    if len(df) and _period_is_partial(df["p"].iloc[-1], hi, grain):
        df = df.iloc[:-1]                      # don't flag an unfinished period as a "drop"
        note = f" (last incomplete {grain} excluded)"
    if len(df) < 6:
        return []

    alerts: list[Alert] = []
    last_label = df["p"].iloc[-1].strftime("%Y-%m-%d" if grain != "month" else "%Y-%m")
    for col in ["__rows"] + measures:
        series = df[col].astype(float).tolist()
        res = robust_anomaly(series[:-1], series[-1])
        if not res:
            continue
        name = "Row volume" if col == "__rows" else col
        direction = "above" if res["z"] > 0 else "below"
        rel = f"{abs(res['rel_change']) * 100:.0f}%" if res["rel_change"] is not None else "far"
        sev = "high" if (abs(res["z"]) >= 6 or (res["rel_change"] and abs(res["rel_change"]) >= 0.5)) else "medium"
        alerts.append(Alert(
            sev, "anomaly", table,
            f"{name} in {last_label} is {rel} {direction} its usual level",
            f"{_fmt(series[-1])} vs a typical {_fmt(res['median'])} per {grain} over the previous "
            f"{len(series) - 1} {grain}s{note}.",
        ))
    return alerts


def _quality_alerts(engine: DataEngine, table: str, meta: dict, dup_scan_limit: int = 5_000_000) -> list[Alert]:
    alerts: list[Alert] = []
    schema, n = meta["schema"], meta["row_count"]
    if n == 0:
        return [Alert("high", "quality", table, "Table is empty")]

    # missing values (already computed during schema refresh)
    for c in schema:
        if c["null_pct"] >= 50:
            alerts.append(Alert("high", "quality", table, f"`{c['name']}` is {c['null_pct']}% empty"))
        elif c["null_pct"] >= 20:
            alerts.append(Alert("medium", "quality", table, f"`{c['name']}` has {c['null_pct']}% missing values"))

    # exact duplicate rows
    if n <= dup_scan_limit:
        distinct = engine.query_internal(f"SELECT COUNT(*) AS n FROM (SELECT DISTINCT * FROM {_q(table)})")["n"].iloc[0]
        dups = n - int(distinct)
        if dups > 0:
            pct = 100 * dups / n
            alerts.append(Alert("medium" if pct >= 1 else "low", "quality", table,
                                f"{dups:,} duplicate rows ({pct:.1f}%)"))

    # extreme outliers (beyond 3x IQR)
    measures = _measure_columns(schema)[:8]
    if measures:
        qexpr = ", ".join(
            f"quantile_cont({_q(m)}, 0.25) AS {_q(m + '__q1')}, quantile_cont({_q(m)}, 0.75) AS {_q(m + '__q3')}"
            for m in measures)
        q = engine.query_internal(f"SELECT {qexpr} FROM {_q(table)}").iloc[0]
        bounds = {}
        for m in measures:
            q1, q3 = q[m + "__q1"], q[m + "__q3"]
            if pd.notna(q1) and pd.notna(q3) and q3 > q1:
                bounds[m] = (q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))
        if bounds:
            cexpr = ", ".join(
                f"SUM(CASE WHEN {_q(m)} < {lo} OR {_q(m)} > {hi} THEN 1 ELSE 0 END) AS {_q(m)}"
                for m, (lo, hi) in bounds.items())
            counts = engine.query_internal(f"SELECT {cexpr} FROM {_q(table)}").iloc[0]
            for m in bounds:
                k = int(counts[m] or 0)
                if k > 0:
                    alerts.append(Alert("low", "quality", table,
                                        f"{k:,} extreme outlier(s) in `{m}`",
                                        f"Values beyond 3×IQR: outside [{_fmt(bounds[m][0])}, {_fmt(bounds[m][1])}]."))
    return alerts


def scan_table(engine: DataEngine, table: str) -> list[Alert]:
    meta = engine.tables[table]
    out: list[Alert] = []
    for fn in (lambda: _anomaly_alerts(engine, table, meta["schema"]), lambda: _quality_alerts(engine, table, meta)):
        try:
            out.extend(fn())
        except Exception as e:                  # a failing check must never break the dashboard
            out.append(Alert("info", "quality", table, "A check could not run", str(e)[:150]))
    return out


def scan_all(engine: DataEngine) -> list[Alert]:
    alerts = [a for t in engine.tables for a in scan_table(engine, t)]
    return sorted(alerts, key=lambda a: SEVERITY_ORDER.get(a.severity, 9))


# ═══════════════════════════════════════════════════════════ user-defined rules

@dataclass
class AlertRule:
    name: str
    table: str
    expression: str                 # aggregate expression, e.g. SUM(amount)
    mode: str = "value"             # "value" | "period_change_pct"
    operator: str = "<"             # "<" | ">"
    threshold: float = 0.0
    date_column: str | None = None  # required for period_change_pct
    grain: str = "month"


@dataclass
class RuleResult:
    rule: AlertRule
    value: float | None = None
    fired: bool = False
    message: str = ""
    error: str | None = None


def _compare(v: float, op: str, thr: float) -> bool:
    return v < thr if op == "<" else v > thr


def evaluate_rule(engine: DataEngine, rule: AlertRule) -> RuleResult:
    res = RuleResult(rule)
    meta = engine.tables.get(rule.table)
    if not meta:
        res.error = f"Table '{rule.table}' not found."
        return res
    try:
        if rule.operator not in ("<", ">"):
            raise ValueError("Operator must be < or >.")
        if rule.mode == "value":
            sql = f"SELECT {rule.expression} AS v FROM {_q(rule.table)}"
            v = engine.run_query(sql, use_cache=False).df["v"].iloc[0]       # validated, timed out, capped
            res.value = None if pd.isna(v) else float(v)
            if res.value is not None:
                res.fired = _compare(res.value, rule.operator, rule.threshold)
                res.message = f"{rule.expression} = {_fmt(res.value)} (alert if {rule.operator} {rule.threshold:g})"
        elif rule.mode == "period_change_pct":
            cols = {c["name"] for c in meta["schema"]}
            if rule.date_column not in cols:
                raise ValueError(f"Date column '{rule.date_column}' not found.")
            if rule.grain not in GRAINS:
                raise ValueError("Grain must be day, week or month.")
            sql = (f"SELECT date_trunc('{rule.grain}', {_q(rule.date_column)}) AS p, {rule.expression} AS v "
                   f"FROM {_q(rule.table)} WHERE {_q(rule.date_column)} IS NOT NULL GROUP BY 1 ORDER BY 1")
            df = engine.run_query(sql, use_cache=False).df
            _, hi = _date_range(engine, rule.table, rule.date_column)
            if len(df) and _period_is_partial(pd.Timestamp(df["p"].iloc[-1]), hi, rule.grain):
                df = df.iloc[:-1]
            if len(df) < 2:
                res.message = "Not enough complete periods yet."
                return res
            prev, last = float(df["v"].iloc[-2]), float(df["v"].iloc[-1])
            if prev == 0:
                res.message = "Previous period is 0 — % change undefined."
                return res
            res.value = (last - prev) / abs(prev) * 100
            res.fired = _compare(res.value, rule.operator, rule.threshold)
            p_last = pd.Timestamp(df["p"].iloc[-1]).strftime("%Y-%m-%d")
            res.message = (f"{rule.expression} changed {res.value:+.1f}% in the latest complete {rule.grain} "
                           f"({p_last}): {_fmt(prev)} → {_fmt(last)} (alert if {rule.operator} {rule.threshold:g}%)")
        else:
            raise ValueError("Unknown rule mode.")
    except Exception as e:
        res.error = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
    return res


def evaluate_rules(engine: DataEngine, rules: list[AlertRule]) -> list[RuleResult]:
    return [evaluate_rule(engine, r) for r in rules]


def rule_alerts(results: list[RuleResult]) -> list[Alert]:
    return [Alert("high", "rule", r.rule.table, f"Rule “{r.rule.name}” triggered", r.message)
            for r in results if r.fired]


# ═══════════════════════════════════════════════════════════ persistence

def save_rules(path: str | Path, rules: list[AlertRule]) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in rules], indent=2))


def load_rules(path: str | Path) -> list[AlertRule]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return [AlertRule(**d) for d in json.loads(p.read_text())]
    except (json.JSONDecodeError, TypeError):
        return []
