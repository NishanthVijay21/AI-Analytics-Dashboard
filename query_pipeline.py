"""
query_pipeline.py
One function — run_question() — that the Streamlit app AND the eval harness both call,
so what you measure is exactly what users get.

  question (+ prior turns) -> LLM plan -> validate -> execute (timeout, row cap)
        └─ on any failure: show the LLM its SQL + the error, ask for a fix (up to max_attempts)
"""

from __future__ import annotations

import json

import duckdb

from data_engine import DataEngine, QueryTimeout
from sql_validator import SQLValidationError

DEFAULT_MAX_ATTEMPTS = 3
CONTEXT_TURNS = 3


def build_context(history: list[dict], n: int = CONTEXT_TURNS) -> list[dict]:
    """
    Turn the last `n` *successful* turns into chat messages for follow-up questions.
    Only the question and the plan we returned are included — never result rows.
    """
    good = [t for t in history if t.get("plan") and not t.get("error")][-n:]
    messages: list[dict] = []
    for t in good:
        p = t["plan"]
        messages.append({"role": "user", "content": t["question"]})
        messages.append({"role": "assistant", "content": json.dumps(
            {"sql": p.get("sql"), "mode": p.get("mode"), "chart": p.get("chart"), "note": p.get("note", "")}
        )})
    return messages


def run_question(
    llm,
    engine: DataEngine,
    question: str,
    history: list[dict] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict:
    """
    Returns a turn dict:
      question, plan, result_df, error,
      attempts   -> [{"attempt": 1, "sql": "...", "error": "..." | None}, ...]
      exec_ms, truncated, total_rows, cached
    """
    turn = {
        "question": question, "plan": None, "result_df": None, "error": None,
        "attempts": [], "exec_ms": None, "truncated": False, "total_rows": None, "cached": False,
    }
    context = build_context(history or [])
    schemas = engine.all_schemas_description()
    feedback: dict | None = None

    for attempt in range(1, max_attempts + 1):
        # ---- 1. plan
        try:
            plan = llm.get_query_plan(
                user_question=question,
                schemas_description=schemas,
                context=context,
                previous_attempt=feedback,
            )
        except Exception as e:  # network error, unparseable JSON, empty SQL...
            turn["attempts"].append({"attempt": attempt, "sql": None, "error": f"Planning failed: {e}"})
            turn["error"] = f"Couldn't get a query plan: {e}"
            feedback = {"plan": None, "error": f"Your last reply could not be used: {str(e)[:300]}"}
            continue

        turn["plan"] = plan

        # ---- 2. validate + execute
        try:
            res = engine.run_query(plan["sql"])
        except (SQLValidationError, QueryTimeout, duckdb.Error, ValueError) as e:
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            turn["attempts"].append({"attempt": attempt, "sql": plan["sql"], "error": msg})
            prefix = "SQL was rejected: " if isinstance(e, SQLValidationError) else "SQL execution failed: "
            turn["error"] = prefix + msg
            feedback = {"plan": plan, "error": msg}
            continue

        # ---- 3. success
        turn["attempts"].append({"attempt": attempt, "sql": plan["sql"], "error": None})
        turn.update(
            result_df=res.df, error=None, exec_ms=res.elapsed_ms,
            truncated=res.truncated, total_rows=res.total_rows, cached=res.cached,
        )
        return turn

    return turn  # error from the last attempt is already set
