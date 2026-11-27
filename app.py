"""
app.py
"""

import io
import json
import os
import zipfile
from dataclasses import asdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from alerts_engine import (AlertRule, GRAINS, SEVERITY_ORDER, _is_temporal, evaluate_rules, load_rules,
                           rule_alerts, save_rules, scan_all)
from chart_component import describe_chart_type, render_chart, render_dashboard_export_button
from data_engine import DataEngine
from insights_engine import compute_table_stats
from json_utils import dataframe_records, json_default
from llm_engine import LLMEngine
from preprocess_ui import clear_preprocessor, render_preprocess_panel
from query_pipeline import CONTEXT_TURNS, run_question

load_dotenv()
st.set_page_config(page_title="AI analytics dashboard", page_icon="◧", layout="wide")

RULES_PATH = "alert_rules.json"
PREPROCESS_ROW_LIMIT = 500_000      # the pandas-based cleaning panel is disabled above this
INSIGHT_SAMPLE_ROWS = 200_000
PAGE_SIZE = 100
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡", "info": "ℹ️"}


# ──────────────────────────────────────────── session state
def _init_state():
    defaults = {
        "engine": DataEngine(memory_limit="2GB", max_rows=10_000, timeout_s=30),
        "llm": None,
        "history": [],
        "context_start": 0,            # follow-up context only looks at history[context_start:]
        "rules": load_rules(RULES_PATH),
        "sugg": {"sig": None, "slots": [], "pool": [], "used": []},   # suggestion buttons
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

engine: DataEngine = st.session_state.engine
HAS_KEY = bool(os.environ.get("GROQ_API_KEY"))


def get_llm() -> LLMEngine:
    if st.session_state.llm is None:
        st.session_state.llm = LLMEngine()
    return st.session_state.llm


def _generate_insights(engine: DataEngine, table_name: str) -> None:
    """Grounded stats (computed locally, sampled if huge) -> LLM phrases them as bullets."""
    key = f"insights_{table_name}"
    if not HAS_KEY:
        return
    try:
        df, sampled = engine.sample_dataframe(table_name, INSIGHT_SAMPLE_ROWS)
        total = engine.tables[table_name]["row_count"] if sampled else None
        stats = compute_table_stats(df, total_rows=total)
        st.session_state[key] = get_llm().get_key_insights(table_name, stats)
    except Exception as e:
        st.session_state[key] = [f"⚠️ Couldn't generate insights: {e}"]


def get_alerts():
    """Auto alerts + rule results, recomputed only when the data or the rules change."""
    sig = (engine.data_version, json.dumps([asdict(r) for r in st.session_state.rules], sort_keys=True))
    if st.session_state.get("alerts_sig") != sig:
        auto = scan_all(engine)
        results = evaluate_rules(engine, st.session_state.rules)
        st.session_state["alerts"] = sorted(auto + rule_alerts(results), key=lambda a: SEVERITY_ORDER.get(a.severity, 9))
        st.session_state["rule_results"] = results
        st.session_state["alerts_sig"] = sig
    return st.session_state["alerts"], st.session_state["rule_results"]


SUGGESTION_SLOTS = 6   # 3 rows x 2 buttons


def _fetch_suggestions(n: int) -> list[str]:
    st_ = st.session_state.sugg
    seen = set(st_["used"]) | set(st_["slots"]) | set(st_["pool"])
    try:
        qs = get_llm().suggest_questions(engine.all_schemas_description(), n=n,
                                         exclude=list(seen))
    except Exception:
        return []
    return [q for q in qs if q not in seen]


def _next_suggestion() -> str | None:
    """Next unused suggestion from the pool; asks the LLM for more when the pool runs dry."""
    st_ = st.session_state.sugg
    if not st_["pool"]:
        st_["pool"] = _fetch_suggestions(6)
    return st_["pool"].pop(0) if st_["pool"] else None


def ensure_suggestions() -> list[str]:
    """Slots are reset only when the set of tables changes."""
    st_ = st.session_state.sugg
    sig = tuple(sorted(engine.tables))
    if st_["sig"] != sig:
        st_.update(sig=sig, slots=[], pool=[], used=[])
        fresh = _fetch_suggestions(SUGGESTION_SLOTS + 4)
        st_["slots"], st_["pool"] = fresh[:SUGGESTION_SLOTS], fresh[SUGGESTION_SLOTS:]
    return st_["slots"]


# ──────────────────────────────────────────── sidebar
with st.sidebar:
    st.markdown("## ◧ AI analytics dashboard")

    st.markdown("### Setup")
    if HAS_KEY:
        st.success("🔒 Connected to Groq")
        st.caption("Only table/column names and types are sent to the model — never row data.")
    else:
        st.warning("⚠️ Missing GROQ_API_KEY in your .env file.")

    st.divider()

    st.markdown("### Datasets")
    uploaded_files = st.file_uploader(
        "Upload CSV, Parquet or Excel files",
        type=["csv", "tsv", "parquet", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    for uf in (uploaded_files or []):
        file_key = f"loaded_{uf.name}_{uf.size}"
        if file_key not in st.session_state:
            try:
                tname = engine.load_file(uf, uf.name)
                st.session_state[file_key] = tname
                st.success(f"Loaded **{uf.name}** → `{tname}`")
                with st.spinner("Generating key insights…"):
                    _generate_insights(engine, tname)
            except Exception as e:
                st.error(f"Failed to load {uf.name}: {e}")

    if engine.tables:
        st.markdown("**Loaded tables**")
        for tname, meta in list(engine.tables.items()):
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"`{tname}` — {meta['row_count']:,} rows")
            if col2.button("✕", key=f"drop_{tname}", help=f"Remove {tname}"):
                clear_preprocessor(tname)
                st.session_state.pop(f"insights_{tname}", None)
                engine.drop_table(tname)
                st.rerun()
    else:
        st.info("No datasets loaded yet.")

    st.divider()

    st.markdown("### Merge tables")
    if len(engine.tables) >= 2:
        merge_desc = st.text_area(
            "Describe the join",
            placeholder="e.g. inner join sales and customers on region",
            height=90,
            label_visibility="collapsed",
        )
        merge_name = st.text_input("Result table name", value="merged")
        if st.button("Run merge", use_container_width=True):
            if merge_desc.strip():
                with st.spinner("Writing merge SQL…"):
                    try:
                        sql = get_llm().get_merge_sql(merge_desc, engine.all_schemas_description())
                        engine.merge_tables(sql, merge_name)      # validated like any other query
                        st.success(f"Created table `{merge_name}`")
                        st.code(sql, language="sql")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Merge failed: {e}")
            else:
                st.warning("Describe the join first.")
    else:
        st.caption("Upload ≥ 2 datasets to enable merging.")

    st.divider()

    # ── data-only exports (Python side — no pixel access needed)
    chart_turns = [
        t for t in st.session_state.history
        if t.get("result_df") is not None and (t.get("plan") or {}).get("mode") == "chart"
    ]
    if chart_turns:
        st.markdown("### Data exports")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, turn in enumerate(chart_turns):
                csv_bytes = turn["result_df"].to_csv(index=False).encode()
                label = turn["question"][:40].replace(" ", "_").replace("/", "-")
                zf.writestr(f"chart_{i+1}_{label}.csv", csv_bytes)
        zip_buf.seek(0)
        st.download_button(
            "⬇ All chart data (ZIP of CSVs)",
            data=zip_buf,
            file_name="querydeck_data.zip",
            mime="application/zip",
            use_container_width=True,
        )

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            for i, turn in enumerate(chart_turns):
                sheet_label = f"Chart {i+1}"
                df = turn["result_df"]
                question_df = pd.DataFrame({"Question": [turn["question"]]})
                question_df.to_excel(writer, sheet_name=sheet_label, index=False, startrow=0)
                df.to_excel(writer, sheet_name=sheet_label, index=False, startrow=2)
        excel_buf.seek(0)
        st.download_button(
            "⬇ All chart data (Excel workbook)",
            data=excel_buf,
            file_name="querydeck_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption("Chart images → use the PNG/PDF buttons next to each chart below.")


# ──────────────────────────────────────────── main area
st.markdown("## AI analytics dashboard")
st.caption("Ask in plain English — charts or result tables depending on what you need.")

if not engine.tables:
    st.info("Upload a CSV, Parquet or Excel file from the sidebar to get started.")
    st.stop()

# ── alerts (proactive: computed on load and whenever data / rules change)
alerts, rule_results = get_alerts()
n_high = sum(a.severity == "high" for a in alerts)
alert_label = f"🚨 Alerts ({len(alerts)})" if alerts else "✅ No alerts"
with st.expander(alert_label, expanded=n_high > 0):
    if not alerts:
        st.caption("No anomalies, data-quality problems or triggered rules were found.")
    for a in alerts:
        st.markdown(f"{SEVERITY_ICON.get(a.severity, '•')} **{a.title}** · `{a.table}`")
        if a.detail:
            st.caption(a.detail)

with st.expander("🔔 Alert rules", expanded=False):
    st.caption("Rules are checked automatically whenever your data changes. Expressions are validated like any other query.")
    rc1, rc2 = st.columns(2)
    r_name = rc1.text_input("Rule name", placeholder="Revenue drop", key="rule_name")
    r_table = rc2.selectbox("Table", list(engine.tables), key="rule_table")
    r_expr = st.text_input("Metric (aggregate expression)", placeholder="SUM(amount)", key="rule_expr")
    rc3, rc4, rc5 = st.columns(3)
    r_mode = rc3.selectbox("Check", ["value", "period_change_pct"], key="rule_mode",
                           format_func=lambda m: "Current value" if m == "value" else "% change vs previous period")
    r_op = rc4.selectbox("Alert when", ["<", ">"], key="rule_op")
    r_thr = rc5.number_input("Threshold", value=-20.0 if r_mode == "period_change_pct" else 0.0, key=f"rule_thr_{r_mode}")
    r_date, r_grain = None, "month"
    if r_mode == "period_change_pct":
        date_opts = [c["name"] for c in engine.tables[r_table]["schema"] if _is_temporal(c["type"])]
        rd1, rd2 = st.columns(2)
        r_date = rd1.selectbox("Date column", date_opts, key="rule_date") if date_opts else None
        r_grain = rd2.selectbox("Period", list(GRAINS), index=2, key="rule_grain")
        if not date_opts:
            st.caption("This table has no date/timestamp column — parse one in the Preprocess panel first.")
    if st.button("Add rule", key="rule_add"):
        if r_name.strip() and r_expr.strip() and (r_mode == "value" or r_date):
            st.session_state.rules.append(AlertRule(r_name.strip(), r_table, r_expr.strip(), r_mode, r_op,
                                                    float(r_thr), r_date, r_grain))
            save_rules(RULES_PATH, st.session_state.rules)
            st.rerun()
        else:
            st.warning("Give the rule a name, a metric, and (for % change) a date column.")

    for i, rr in enumerate(rule_results):
        c1, c2 = st.columns([8, 1])
        status = "⚠️ error" if rr.error else ("🔴 triggered" if rr.fired else "🟢 ok")
        c1.markdown(f"**{rr.rule.name}** — `{rr.rule.table}` · {status}")
        c1.caption(rr.error or rr.message or "No value yet.")
        if c2.button("✕", key=f"rule_del_{i}"):
            st.session_state.rules.pop(i)
            save_rules(RULES_PATH, st.session_state.rules)
            st.rerun()

# ── key insights (surfaced automatically on upload, before any question is asked)
tables_with_insights = [t for t in engine.tables if st.session_state.get(f"insights_{t}") is not None]
if tables_with_insights:
    st.markdown("### 💡 Key Insights")
    for tname in tables_with_insights:
        with st.container(border=True):
            hc1, hc2 = st.columns([6, 1])
            hc1.markdown(f"**`{tname}`**")
            if hc2.button("🔄", key=f"refresh_insights_{tname}", help="Regenerate insights"):
                with st.spinner("Regenerating insights…"):
                    _generate_insights(engine, tname)
                st.rerun()
            for bullet in st.session_state[f"insights_{tname}"]:
                st.markdown(f"- {bullet}")
    st.markdown("")

# ── schema expanders
with st.expander("Loaded schemas", expanded=False):
    for tname, meta in engine.tables.items():
        st.markdown(f"**`{tname}`** — {meta['row_count']:,} rows · _{meta['filename']}_")
        if meta["schema"]:
            schema_df = pd.DataFrame(meta["schema"]).rename(columns={
                "name": "column", "type": "type",
                "sample": "sample value", "null_pct": "null %",
            })
            st.dataframe(schema_df.astype({"sample value": str}), use_container_width=True, hide_index=True)

# ── preprocessing
with st.expander("🧹 Preprocess data", expanded=False):
    st.caption(
        "Profile a table, then clean it up: missing values, duplicates, "
        "datatype fixes, date parsing, normalization, and encoding."
    )
    pp_table = st.selectbox(
        "Table to preprocess", list(engine.tables.keys()), key="preprocess_table_select"
    )
    if pp_table:
        if engine.tables[pp_table]["row_count"] > PREPROCESS_ROW_LIMIT:
            st.info(
                f"`{pp_table}` has {engine.tables[pp_table]['row_count']:,} rows — too large for the in-memory "
                "cleaning panel. Clean it with a question instead (e.g. “cast price to integer and trim names”), "
                "then use **Save as New Dataset** on the result table; that runs entirely inside DuckDB."
            )
        else:
            render_preprocess_panel(engine, pp_table)

# ── follow-up context controls + suggested questions
use_context = st.toggle("Remember earlier questions (for follow-ups like “now split by region”)",
                        value=True, key="use_context")
ctx_turns = [t for t in st.session_state.history[st.session_state.context_start:]
             if t.get("plan") and not t.get("error")][-CONTEXT_TURNS:]
cc1, cc2 = st.columns([5, 1])
if use_context and ctx_turns:
    cc1.caption(f"Using your last {len(ctx_turns)} question(s) as context.")
if cc2.button("🧹 New topic", help="Forget earlier questions for follow-ups", disabled=not ctx_turns):
    st.session_state.context_start = len(st.session_state.history)
    st.rerun()

# ── query box
with st.form("query_form", clear_on_submit=True):
    question = st.text_area(
        "Ask something",
        height=80,
        label_visibility="collapsed",
        placeholder=(
            "e.g. show monthly revenue as a bar chart  ·  "
            "now only for 2024  ·  "
            "show top 10 customers by total spend"
        ),
    )
    submitted = st.form_submit_button("Run", use_container_width=True)

# ── suggestion buttons (under the text box, two per row; a clicked one is swapped for a new one)
if HAS_KEY:
    slots = ensure_suggestions()
    if slots:
        st.caption("Try asking:")
        for row in range(0, len(slots), 2):
            cols = st.columns(2)
            for j, q in enumerate(slots[row:row + 2]):
                idx = row + j
                if cols[j].button(q, key=f"suggest_{idx}", use_container_width=True):
                    sg = st.session_state.sugg
                    sg["used"].append(q)
                    repl = _next_suggestion()
                    if repl:
                        sg["slots"][idx] = repl
                    else:
                        sg["slots"].pop(idx)
                    st.session_state["pending_question"] = q
                    st.rerun()

pending = st.session_state.pop("pending_question", None)
q_to_run = pending or (question.strip() if submitted and question.strip() else None)

if q_to_run:
    if not HAS_KEY:
        st.error("Missing GROQ_API_KEY in your .env file.")
    else:
        ctx_history = st.session_state.history[st.session_state.context_start:] if use_context else []
        with st.spinner("Planning, validating and running query…"):
            turn = run_question(get_llm(), engine, q_to_run, ctx_history)
        st.session_state.history.append(turn)
        st.rerun()


# ──────────────────────────────────────────── thread (newest first)
dashboard_charts = []
for i, turn in enumerate(reversed(st.session_state.history)):
    if (
        (turn.get("plan") or {}).get("mode") == "chart"
        and turn.get("result_df") is not None
        and not turn.get("error")
        and (turn.get("plan") or {}).get("chart")
    ):
        spec = dict(turn["plan"]["chart"])
        spec["data"] = {"values": dataframe_records(turn["result_df"])}
        spec.setdefault("$schema", "https://vega.github.io/schema/vega-lite/v5.json")
        dashboard_charts.append({"spec": spec, "title": turn["question"]})

if dashboard_charts:
    render_dashboard_export_button(dashboard_charts)
    st.markdown("")

for i, turn in enumerate(reversed(st.session_state.history)):
    global_idx = len(st.session_state.history) - 1 - i

    st.markdown("---")
    st.markdown(f"**{turn['question']}**")

    plan = turn.get("plan") or {}
    plan_mode = plan.get("mode", "chart")
    explanation = plan.get("explanation") or {}
    attempts = turn.get("attempts") or []

    with st.expander("🧠 Explainability", expanded=False):
        st.markdown("**Question**")
        st.write(turn["question"])

        st.markdown("**Generated SQL**")
        st.code(plan.get("sql", "(none)"), language="sql")

        if len(attempts) > 1:
            st.markdown("**Self-correction**")
            for a in attempts:
                icon = "✅" if not a["error"] else "❌"
                st.markdown(f"{icon} Attempt {a['attempt']}" + (f" — {a['error']}" if a["error"] else ""))

        st.markdown("**Reasoning**")
        reasoning = (explanation.get("reasoning") or "").strip()
        columns_used = explanation.get("columns_used") or []
        if reasoning:
            st.write(reasoning)
        if columns_used:
            st.markdown("I identified:")
            for c in columns_used:
                col = c.get("column", "?")
                role = c.get("role", "")
                st.markdown(f"- `{col}`" + (f" — {role}" if role else ""))
        if not reasoning and not columns_used:
            st.caption("No reasoning details were returned for this query.")

        agg_col, viz_col = st.columns(2)
        with agg_col:
            st.markdown("**Aggregation**")
            st.write(explanation.get("aggregation") or "None")
        with viz_col:
            st.markdown("**Visualization**")
            st.write(describe_chart_type(plan_mode, plan.get("chart")))

    if turn.get("error"):
        st.error(turn["error"] + (f"  (tried {len(attempts)} time(s))" if len(attempts) > 1 else ""))
        continue

    result_df: pd.DataFrame | None = turn.get("result_df")
    if result_df is None or result_df.empty:
        st.warning("Query returned no rows.")
        continue

    # ── run info: rows, time, cache, truncation, retries
    note = plan.get("note", "")
    bits = [note] if note else []
    if turn.get("truncated"):
        total = turn.get("total_rows")
        bits.append(f"first {len(result_df):,}" + (f" of {total:,}" if total else "") + " rows")
    else:
        bits.append(f"{len(result_df):,} row(s)")
    bits.append("cached" if turn.get("cached") else f"{turn.get('exec_ms') or 0:,.0f} ms")
    if len(attempts) > 1:
        bits.append(f"fixed after {len(attempts)} attempts")
    caption = " · ".join(bits)

    if turn.get("truncated"):
        st.warning(
            f"The result was capped at {len(result_df):,} rows to keep the app responsive. "
            "Aggregate or filter the question to see everything, or use **Save as New Dataset** to "
            "keep the full result inside DuckDB."
        )

    mode = plan_mode

    # ── TABLE mode
    if mode == "table":
        pages = max(1, -(-len(result_df) // PAGE_SIZE))
        page = 1
        if pages > 1:
            page = st.number_input(f"Page (of {pages})", min_value=1, max_value=pages, value=1,
                                   key=f"page_{global_idx}")
        start = (page - 1) * PAGE_SIZE
        st.dataframe(result_df.iloc[start:start + PAGE_SIZE], use_container_width=True, hide_index=True)
        st.caption(caption)
        csv_bytes = result_df.to_csv(index=False).encode()
        label = turn["question"][:30].replace(" ", "_")
        st.download_button(
            "⬇ Download result as CSV" + (" (capped rows)" if turn.get("truncated") else ""),
            data=csv_bytes,
            file_name=f"{label}.csv",
            mime="text/csv",
            key=f"csv_dl_{global_idx}",
        )

        with st.expander("💾 Save as New Dataset", expanded=False):
            c1, c2 = st.columns([3, 1])
            new_table = c1.text_input(
                "Table name:",
                key=f"save_name_{global_idx}",
                placeholder="e.g. clean_sales",
            )
            if c2.button("Save", key=f"save_btn_{global_idx}", use_container_width=True):
                if new_table.strip():
                    try:
                        engine.merge_tables(plan["sql"], new_table.strip())
                        st.success(f"Saved as table `{new_table.strip()}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving table: {e}")
                else:
                    st.warning("Please enter a table name.")

    # ── CHART mode
    else:
        chart_spec = dict(plan.get("chart") or {})
        if not chart_spec:
            st.dataframe(result_df.head(PAGE_SIZE), use_container_width=True, hide_index=True)
            st.caption("(No chart spec returned — showing raw data.)")
        else:
            chart_spec["data"] = {"values": dataframe_records(result_df)}
            chart_spec.setdefault("$schema", "https://vega.github.io/schema/vega-lite/v5.json")

            render_chart(spec=chart_spec, chart_id=f"chart_{global_idx}", title=turn["question"])
            st.caption(caption)

            spec_bytes = json.dumps(chart_spec, indent=2, default=json_default).encode()
            label = turn["question"][:30].replace(" ", "_")
            st.download_button(
                "⬇ Vega-Lite spec (JSON)",
                data=spec_bytes,
                file_name=f"{label}_spec.json",
                mime="application/json",
                key=f"spec_dl_{global_idx}",
            )

        with st.expander("Result data", expanded=False):
            st.dataframe(result_df.head(1000), use_container_width=True, hide_index=True)
            if len(result_df) > 1000:
                st.caption(f"Showing first 1,000 of {len(result_df):,} rows. Use the CSV/Excel exports for all of them.")
