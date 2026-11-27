"""
eval.py
End-to-end evaluation harness for the AI Analytics Dashboard.

Measures:
  1. Security & Guardrail Block Rate (deterministic, no LLM calls)
  2. Execution Accuracy (EX) on single-turn SQL queries (Easy / Medium / Hard)
  3. Pass@1 vs Pass@3 (self-correction recovery rate)
  4. Mode Accuracy (chart vs table) & Vega-Lite field alignment
  5. Multi-turn follow-up context retention
  6. Key Insights numeric grounding against compute_table_stats()

Usage:
  python eval.py                  # Full eval (requires GROQ_API_KEY in .env)
  python eval.py --offline        # Security & validator tests only (no API calls)
  python eval.py --delay 1.5      # Add pause (sec) between Groq calls to avoid rate limits
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_engine import DataEngine
from insights_engine import compute_table_stats
from llm_engine import LLMEngine
from query_pipeline import run_question
from sql_validator import SQLValidationError, validate_sql

load_dotenv()


# ══════════════════════════════════════════════════════════════ 1. Synthetic Benchmark Data

def build_benchmark_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic multi-table dataset (sales + customers) for reproducible evals."""
    rng = np.random.default_rng(42)

    customers = pd.DataFrame({
        "customer_id": range(1, 21),
        "customer_name": [f"Customer_{i:02d}" for i in range(1, 21)],
        "region": (["North", "South", "East", "West"] * 5),
        "segment": (["Enterprise", "SMB", "Consumer", "Enterprise", "SMB"] * 4),
    })

    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n = 300
    chosen_dates = rng.choice(dates, size=n)
    categories = rng.choice(["Electronics", "Furniture", "Clothing", "Groceries"], size=n, p=[0.35, 0.25, 0.25, 0.15])
    cust_ids = rng.integers(1, 21, size=n)
    quantities = rng.integers(1, 10, size=n)
    unit_prices = np.round(rng.uniform(20.0, 500.0, size=n), 2)
    amounts = np.round(quantities * unit_prices, 2)
    statuses = rng.choice(["Completed", "Completed", "Completed", "Returned", "UPDATE"], size=n)

    sales = pd.DataFrame({
        "order_id": range(1001, 1001 + n),
        "order_date": pd.to_datetime(chosen_dates),
        "customer_id": cust_ids,
        "category": categories,
        "quantity": quantities,
        "amount": amounts,
        "status": statuses,
    })

    # Inject a few nulls and duplicates so data-quality / cleaning questions are realistic
    sales.loc[[5, 15, 25], "category"] = None
    sales = pd.concat([sales, sales.iloc[[0, 1]]], ignore_index=True)

    return sales, customers


# ══════════════════════════════════════════════════════════════ 2. Security & Guardrail Tests

ADVERSARIAL_SQL_CASES = [
    ("DDL DROP", "DROP TABLE sales"),
    ("DML DELETE", "DELETE FROM sales WHERE order_id = 1001"),
    ("DML INSERT", "INSERT INTO sales (order_id) VALUES (9999)"),
    ("DML UPDATE", "UPDATE sales SET amount = 0"),
    ("DDL CREATE", "CREATE TABLE hacked AS SELECT * FROM sales"),
    ("File read_csv", "SELECT * FROM read_csv('/etc/passwd')"),
    ("File read_parquet", "SELECT * FROM read_parquet('secret.parquet')"),
    ("Env leak getenv", "SELECT getenv('GROQ_API_KEY') FROM sales"),
    ("Catalog info_schema", "SELECT * FROM information_schema.tables"),
    ("Schema-qualified table", "SELECT * FROM main.sales"),
    ("Multi-statement injection", "SELECT COUNT(*) FROM sales; DROP TABLE sales"),
    ("Unlisted table", "SELECT * FROM secret_payroll"),
    ("Non-existent column", "SELECT nonexistent_col FROM sales"),
    ("COPY exfiltration", "COPY sales TO '/tmp/leak.csv'"),
    ("PRAGMA command", "PRAGMA version"),
]

POSITIVE_SQL_CONTROLS = [
    ("String literal with keyword", "SELECT COUNT(*) AS n FROM sales WHERE status = 'UPDATE'"),
    ("Valid CTE + JOIN", "WITH s AS (SELECT customer_id, amount FROM sales) "
                         "SELECT c.region, SUM(s.amount) AS rev FROM s JOIN customers c ON s.customer_id = c.customer_id GROUP BY 1"),
]


def run_security_eval(engine: DataEngine) -> dict:
    allowed_tables = list(engine.tables.keys())
    allowed_cols = {t: [c["name"] for c in m["schema"]] for t, m in engine.tables.items()}

    blocked = 0
    adv_failures = []
    for name, sql in ADVERSARIAL_SQL_CASES:
        try:
            validate_sql(sql, allowed_tables, allowed_cols)
            adv_failures.append({"case": name, "sql": sql, "issue": "Was NOT blocked"})
        except SQLValidationError:
            blocked += 1

    controls_passed = 0
    ctrl_failures = []
    for name, sql in POSITIVE_SQL_CONTROLS:
        try:
            validate_sql(sql, allowed_tables, allowed_cols)
            controls_passed += 1
        except SQLValidationError as e:
            ctrl_failures.append({"case": name, "sql": sql, "issue": str(e)})

    return {
        "adversarial_total": len(ADVERSARIAL_SQL_CASES),
        "adversarial_blocked": blocked,
        "block_rate_pct": round(100 * blocked / len(ADVERSARIAL_SQL_CASES), 1),
        "controls_total": len(POSITIVE_SQL_CONTROLS),
        "controls_passed": controls_passed,
        "failures": adv_failures + ctrl_failures,
    }


# ══════════════════════════════════════════════════════════════ 3. Golden Question Benchmarks

SINGLE_TURN_CASES = [
    # --- Easy
    {
        "id": "E1", "difficulty": "easy", "expected_mode": "chart",
        "question": "show total amount by category as a bar chart",
        "gold_sql": "SELECT category, SUM(amount) AS total_amount FROM sales GROUP BY category",
    },
    {
        "id": "E2", "difficulty": "easy", "expected_mode": "table",
        "question": "show the first 10 rows of sales",
        "gold_sql": "SELECT * FROM sales LIMIT 10",
        "shape_only": True,  # unordered LIMIT 10 can pick any 10 rows
    },
    {
        "id": "E3", "difficulty": "easy", "expected_mode": "chart",
        "question": "count orders by status as a bar chart",
        "gold_sql": "SELECT status, COUNT(*) AS order_count FROM sales GROUP BY status",
    },
    {
        "id": "E4", "difficulty": "easy", "expected_mode": "table",
        "question": "list all orders where category is null",
        "gold_sql": "SELECT * FROM sales WHERE category IS NULL",
    },
    {
        "id": "E5", "difficulty": "easy", "expected_mode": "chart",
        "question": "show average amount per category as a bar chart",
        "gold_sql": "SELECT category, AVG(amount) AS avg_amount FROM sales GROUP BY category",
    },

    # --- Medium
    {
        "id": "M1", "difficulty": "medium", "expected_mode": "chart",
        "question": "show monthly total amount in 2024 as a line chart",
        "gold_sql": (
            "SELECT date_trunc('month', order_date) AS month, SUM(amount) AS total_amount "
            "FROM sales WHERE EXTRACT(YEAR FROM order_date) = 2024 GROUP BY 1 ORDER BY 1"
        ),
    },
    {
        "id": "M2", "difficulty": "medium", "expected_mode": "chart",
        "question": "show total sales amount by customer region as a bar chart",
        "gold_sql": (
            "SELECT c.region, SUM(s.amount) AS total_amount "
            "FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.region"
        ),
    },
    {
        "id": "M3", "difficulty": "medium", "expected_mode": "chart",
        "question": "show top 5 customers by total amount as a bar chart",
        "gold_sql": (
            "SELECT c.customer_name, SUM(s.amount) AS total_amount "
            "FROM sales s JOIN customers c ON s.customer_id = c.customer_id "
            "GROUP BY c.customer_name ORDER BY total_amount DESC LIMIT 5"
        ),
    },
    {
        "id": "M4", "difficulty": "medium", "expected_mode": "table",
        "question": "show sales with amount cast to integer and status in lowercase, limit 20",
        "gold_sql": "SELECT order_id, CAST(amount AS INTEGER) AS amount_int, LOWER(status) AS status_lower FROM sales LIMIT 20",
        "shape_only": True,
    },
    {
        "id": "M5", "difficulty": "medium", "expected_mode": "chart",
        "question": "compare total quantity sold by customer segment as a bar chart",
        "gold_sql": (
            "SELECT c.segment, SUM(s.quantity) AS total_qty "
            "FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.segment"
        ),
    },

    # --- Hard
    {
        "id": "H1", "difficulty": "hard", "expected_mode": "chart",
        "question": "show total completed revenue by region for Electronics only as a bar chart",
        "gold_sql": (
            "SELECT c.region, SUM(s.amount) AS revenue "
            "FROM sales s JOIN customers c ON s.customer_id = c.customer_id "
            "WHERE s.status = 'Completed' AND s.category = 'Electronics' GROUP BY c.region"
        ),
    },
    {
        "id": "H2", "difficulty": "hard", "expected_mode": "table",
        "question": "find duplicate order_ids and how many times each appears",
        "gold_sql": "SELECT order_id, COUNT(*) AS cnt FROM sales GROUP BY order_id HAVING COUNT(*) > 1",
    },
    {
        "id": "H3", "difficulty": "hard", "expected_mode": "chart",
        "question": "show yearly total revenue by region as a grouped bar chart",
        "gold_sql": (
            "SELECT EXTRACT(YEAR FROM s.order_date) AS yr, c.region, SUM(s.amount) AS revenue "
            "FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY 1, 2"
        ),
    },
    {
        "id": "H4", "difficulty": "hard", "expected_mode": "table",
        "question": "show each segment's return rate percentage (percentage of orders with status Returned)",
        "gold_sql": (
            "SELECT c.segment, "
            "100.0 * SUM(CASE WHEN s.status = 'Returned' THEN 1 ELSE 0 END) / COUNT(*) AS return_rate_pct "
            "FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.segment"
        ),
    },
    {
        "id": "H5", "difficulty": "hard", "expected_mode": "table",
        "question": "rank regions by total revenue in 2023 from highest to lowest",
        "gold_sql": (
            "SELECT c.region, SUM(s.amount) AS revenue "
            "FROM sales s JOIN customers c ON s.customer_id = c.customer_id "
            "WHERE EXTRACT(YEAR FROM s.order_date) = 2023 GROUP BY c.region ORDER BY revenue DESC"
        ),
    },
]

FOLLOWUP_CHAINS = [
    {
        "chain_id": "F1",
        "turns": [
            {
                "question": "show total revenue by region as a bar chart",
                "expected_mode": "chart",
                "gold_sql": (
                    "SELECT c.region, SUM(s.amount) AS revenue "
                    "FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.region"
                ),
            },
            {
                "question": "now only for 2024",
                "expected_mode": "chart",
                "gold_sql": (
                    "SELECT c.region, SUM(s.amount) AS revenue "
                    "FROM sales s JOIN customers c ON s.customer_id = c.customer_id "
                    "WHERE EXTRACT(YEAR FROM s.order_date) = 2024 GROUP BY c.region"
                ),
            },
            {
                "question": "show that as a table instead",
                "expected_mode": "table",
                "gold_sql": (
                    "SELECT c.region, SUM(s.amount) AS revenue "
                    "FROM sales s JOIN customers c ON s.customer_id = c.customer_id "
                    "WHERE EXTRACT(YEAR FROM s.order_date) = 2024 GROUP BY c.region"
                ),
            },
        ],
    },
    {
        "chain_id": "F2",
        "turns": [
            {
                "question": "show total amount by category as a bar chart",
                "expected_mode": "chart",
                "gold_sql": "SELECT category, SUM(amount) AS total_amount FROM sales GROUP BY category",
            },
            {
                "question": "exclude rows where category is null",
                "expected_mode": "chart",
                "gold_sql": "SELECT category, SUM(amount) AS total_amount FROM sales WHERE category IS NOT NULL GROUP BY category",
            },
            {
                "question": "only keep the top 2 categories",
                "expected_mode": "chart",
                "gold_sql": (
                    "SELECT category, SUM(amount) AS total_amount FROM sales "
                    "WHERE category IS NOT NULL GROUP BY category ORDER BY total_amount DESC LIMIT 2"
                ),
            },
        ],
    },
]


# ══════════════════════════════════════════════════════════════ 4. Comparison & Spec Helpers

def _normalize_cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return str(v).strip().lower()


def _cell_equal(a, b, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def dataframes_match(pred: pd.DataFrame | None, gold: pd.DataFrame, shape_only: bool = False) -> bool:
    """
    Order-insensitive, alias-insensitive execution accuracy check.
    Compares row counts and sorted normalized value tuples with float tolerance.
    """
    if pred is None or len(pred) != len(gold):
        return False
    if shape_only:
        return len(pred) == len(gold) and len(pred.columns) > 0
    if len(gold) == 0:
        return True
    if len(pred.columns) != len(gold.columns):
        return False

    def to_sorted_rows(df: pd.DataFrame) -> list[tuple]:
        rows = [tuple(_normalize_cell(v) for v in row) for row in df.itertuples(index=False, name=None)]
        # Sort each row's string representation for deterministic order-insensitive matching
        return sorted(rows, key=lambda r: tuple("" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x) for x in r))

    p_rows = to_sorted_rows(pred)
    g_rows = to_sorted_rows(gold)

    for pr, gr in zip(p_rows, g_rows):
        if len(pr) != len(gr):
            return False
        if not all(_cell_equal(a, b) for a, b in zip(pr, gr)):
            return False
    return True


def validate_vega_spec(spec: dict | None, df: pd.DataFrame | None) -> tuple[bool, str]:
    """Verify Vega-Lite spec has a mark and all encoded fields exist in result_df."""
    if not isinstance(spec, dict):
        return False, "Missing chart spec"
    if df is None or df.empty:
        return False, "Empty result dataframe"

    cols = set(df.columns)
    fields: list[str] = []

    def _collect_fields(obj):
        if isinstance(obj, dict):
            if "field" in obj and isinstance(obj["field"], str):
                fields.append(obj["field"])
            for v in obj.values():
                _collect_fields(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect_fields(item)

    if "encoding" in spec:
        _collect_fields(spec["encoding"])
    if "layer" in spec:
        _collect_fields(spec["layer"])

    if not fields:
        return False, "No encoded fields in chart spec"

    missing = [f for f in fields if f not in cols]
    if missing:
        return False, f"Chart references missing column(s): {missing} (available: {sorted(cols)})"
    return True, "ok"


# ══════════════════════════════════════════════════════════════ 5. Insight Grounding Eval

def _extract_numbers(text: str) -> list[float]:
    cleaned = text.replace(",", "")
    nums = []
    for m in re.findall(r"-?\d+(?:\.\d+)?", cleaned):
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def run_insights_eval(llm: LLMEngine, sales_df: pd.DataFrame) -> dict:
    stats = compute_table_stats(sales_df)
    stats_nums = _extract_numbers(json.dumps(stats, default=str))
    bullets = llm.get_key_insights("sales", stats)

    grounded = 0
    details = []
    for b in bullets:
        b_nums = _extract_numbers(b)
        if not b_nums:
            details.append({"bullet": b, "grounded": False, "reason": "No concrete number"})
            continue
        # Check every number in the bullet is close to some number in stats (allows rounding/K-scaling)
        unmatched = []
        for x in b_nums:
            matched = any(
                math.isclose(x, s, rel_tol=0.05, abs_tol=0.5)
                or math.isclose(x * 1000, s, rel_tol=0.05, abs_tol=50.0)
                for s in stats_nums
            )
            if not matched:
                unmatched.append(x)
        ok = len(unmatched) == 0
        if ok:
            grounded += 1
        details.append({"bullet": b, "grounded": ok, "unmatched_numbers": unmatched})

    return {
        "total_bullets": len(bullets),
        "grounded_bullets": grounded,
        "grounding_rate_pct": round(100 * grounded / len(bullets), 1) if bullets else 0.0,
        "details": details,
    }


# ══════════════════════════════════════════════════════════════ 6. Main Runner

def evaluate_case(llm: LLMEngine, engine: DataEngine, case: dict, history: list[dict] | None = None) -> tuple[dict, dict]:
    gold_df = engine.run_query(case["gold_sql"], use_cache=False).df
    t0 = time.perf_counter()
    turn = run_question(llm, engine, case["question"], history=history)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    plan = turn.get("plan") or {}
    attempts = turn.get("attempts") or []
    succeeded = turn.get("error") is None and turn.get("result_df") is not None
    ex_match = succeeded and dataframes_match(turn.get("result_df"), gold_df, shape_only=case.get("shape_only", False))
    pass_at_1 = ex_match and len(attempts) == 1
    recovered = ex_match and len(attempts) > 1

    actual_mode = plan.get("mode")
    mode_ok = actual_mode == case["expected_mode"]

    chart_ok, chart_msg = (True, "n/a")
    if actual_mode == "chart" and succeeded:
        chart_ok, chart_msg = validate_vega_spec(plan.get("chart"), turn.get("result_df"))

    record = {
        "id": case.get("id", ""),
        "difficulty": case.get("difficulty", "followup"),
        "question": case["question"],
        "ex_match": ex_match,
        "pass_at_1": pass_at_1,
        "recovered_via_retry": recovered,
        "attempts": len(attempts),
        "mode_expected": case["expected_mode"],
        "mode_actual": actual_mode,
        "mode_ok": mode_ok,
        "chart_spec_ok": chart_ok,
        "chart_msg": chart_msg,
        "latency_ms": latency_ms,
        "error": turn.get("error"),
        "generated_sql": plan.get("sql"),
        "gold_sql": case["gold_sql"],
    }
    return record, turn


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AI Analytics Dashboard pipeline.")
    parser.add_argument("--offline", action="store_true", help="Run only deterministic security/validator tests.")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between LLM calls.")
    parser.add_argument("--out", default="eval_results.json", help="Output JSON report path.")
    args = parser.parse_args()

    engine = DataEngine()
    sales_df, customers_df = build_benchmark_tables()
    engine.load_dataframe(sales_df, "sales")
    engine.load_dataframe(customers_df, "customers")

    print("\n=== 1. Security & SQL Guardrail Evaluation ===")
    sec = run_security_eval(engine)
    print(f"Adversarial SQL blocked : {sec['adversarial_blocked']}/{sec['adversarial_total']} ({sec['block_rate_pct']}%)")
    print(f"Positive controls passed: {sec['controls_passed']}/{sec['controls_total']}")
    if sec["failures"]:
        for f in sec["failures"]:
            print(f"  [FAIL] {f['case']}: {f['issue']}")

    if args.offline or not os.getenv("GROQ_API_KEY"):
        if not args.offline:
            print("\nGROQ_API_KEY not found in .env — skipping LLM evaluations. Run with key for full benchmark.")
        return

    llm = LLMEngine()

    print("\n=== 2. Single-Turn SQL, Self-Correction & Chart Evaluation ===")
    single_records = []
    for case in SINGLE_TURN_CASES:
        rec, _ = evaluate_case(llm, engine, case)
        single_records.append(rec)
        status = "PASS" if rec["ex_match"] else "FAIL"
        retry_tag = f" (retry x{rec['attempts']})" if rec["attempts"] > 1 else ""
        print(f"  [{status}] {rec['id']} ({rec['difficulty']:6s}) | mode={rec['mode_actual']}/{rec['mode_expected']} | "
              f"chart={'OK' if rec['chart_spec_ok'] else 'BAD'}{retry_tag} | {rec['question'][:50]}")
        if args.delay > 0:
            time.sleep(args.delay)

    print("\n=== 3. Multi-Turn Follow-Up Retention Evaluation ===")
    followup_records = []
    for chain in FOLLOWUP_CHAINS:
        history = []
        for idx, turn_case in enumerate(chain["turns"], start=1):
            turn_case = {**turn_case, "id": f"{chain['chain_id']}_T{idx}", "difficulty": "followup"}
            rec, turn = evaluate_case(llm, engine, turn_case, history=history)
            history.append(turn)
            followup_records.append(rec)
            status = "PASS" if rec["ex_match"] else "FAIL"
            print(f"  [{status}] {rec['id']} | mode={rec['mode_actual']}/{rec['mode_expected']} | {rec['question']}")
            if args.delay > 0:
                time.sleep(args.delay)

    print("\n=== 4. Key Insights Grounding Evaluation ===")
    insights_eval = run_insights_eval(llm, sales_df)
    print(f"Grounded bullets: {insights_eval['grounded_bullets']}/{insights_eval['total_bullets']} "
          f"({insights_eval['grounding_rate_pct']}%)")

    # ── Summary Metrics
    n_single = len(single_records)
    pass1 = sum(r["pass_at_1"] for r in single_records)
    pass3 = sum(r["ex_match"] for r in single_records)
    failed_first = n_single - pass1
    recovered = sum(r["recovered_via_retry"] for r in single_records)

    mode_ok = sum(r["mode_ok"] for r in single_records)
    chart_cases = [r for r in single_records if r["mode_actual"] == "chart" and r["error"] is None]
    chart_valid = sum(r["chart_spec_ok"] for r in chart_cases)

    followup_only = [r for r in followup_records if not r["id"].endswith("_T1")]
    followup_pass = sum(r["ex_match"] for r in followup_only)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "security_block_rate_pct": sec["block_rate_pct"],
        "single_turn_total": n_single,
        "pass_at_1_pct": round(100 * pass1 / n_single, 1),
        "pass_at_3_ex_pct": round(100 * pass3 / n_single, 1),
        "retry_recovery_rate_pct": round(100 * recovered / failed_first, 1) if failed_first else 100.0,
        "mode_accuracy_pct": round(100 * mode_ok / n_single, 1),
        "vega_spec_valid_pct": round(100 * chart_valid / len(chart_cases), 1) if chart_cases else 0.0,
        "followup_retention_pct": round(100 * followup_pass / len(followup_only), 1) if followup_only else 0.0,
        "insight_grounding_pct": insights_eval["grounding_rate_pct"],
        "avg_latency_ms": round(sum(r["latency_ms"] for r in single_records) / n_single, 1),
    }

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    for k, v in summary.items():
        if k != "timestamp_utc":
            print(f"  {k:<28}: {v}")

    report = {
        "summary": summary,
        "security": sec,
        "single_turn": single_records,
        "followups": followup_records,
        "insights": insights_eval,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nDetailed log saved to {args.out}")


if __name__ == "__main__":
    main()