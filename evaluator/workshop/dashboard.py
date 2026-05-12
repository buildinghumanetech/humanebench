"""HumaneBench workshop dashboard - Streamlit one-pager.

Reads a JSONL of evaluation results produced by ``batch_evaluate.py`` and
renders the same view shape the Storytell production Grafana dashboard uses:
overall HumaneScore, per-principle averages, distribution, and a drill-down
table of the lowest-scoring conversations.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PRINCIPLES = [
    "respect_attention",
    "meaningful_choices",
    "enhance_capabilities",
    "dignity_safety",
    "healthy_relationships",
    "longterm_wellbeing",
    "transparency_honesty",
    "equity_inclusion",
]

PRINCIPLE_LABELS = {
    "respect_attention": "Respect Attention",
    "meaningful_choices": "Meaningful Choices",
    "enhance_capabilities": "Enhance Capabilities",
    "dignity_safety": "Dignity & Safety",
    "healthy_relationships": "Healthy Relationships",
    "longterm_wellbeing": "Long-term Wellbeing",
    "transparency_honesty": "Transparency & Honesty",
    "equity_inclusion": "Equity & Inclusion",
}


@st.cache_data(show_spinner=False)
def load_results(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        flat = {
            "id": r["id"],
            "timestamp": pd.to_datetime(r["timestamp"]),
            "user_prompt": r["user_prompt"],
            "response": r["response"],
            "principle_focus": r.get("principle_focus"),
            "expected_severity": r.get("expected_severity"),
            "humane_score": r["humane_score"],
            "confidence": r["confidence"],
            "global_violations": r.get("global_violations") or [],
            "rationales": r.get("rationales") or {},
            "model": r.get("model"),
        }
        for principle in PRINCIPLES:
            flat[principle] = r["scores"].get(principle)
        rows.append(flat)
    return pd.DataFrame(rows)


def score_color(score: float) -> str:
    if score is None:
        return "gray"
    if score >= 0.75:
        return "#0a8f3f"
    if score >= 0.25:
        return "#7eb238"
    if score >= -0.25:
        return "#d6b400"
    if score >= -0.75:
        return "#d97a00"
    return "#c0392b"


def main() -> None:
    st.set_page_config(page_title="HumaneBench Dashboard", layout="wide")
    st.title("HumaneBench eval dashboard")
    st.caption("Continuous humane-AI evaluation, workshop edition.")

    default_path = Path(__file__).parent / "results.jsonl"
    results_path = Path(
        st.sidebar.text_input("Results JSONL", value=str(default_path))
    )
    if not results_path.exists():
        st.error(f"File not found: {results_path}. Run `batch_evaluate.py` first.")
        return

    df = load_results(results_path)
    if df.empty:
        st.warning("No successful evaluations in this file.")
        return

    # ---- Filters ----
    with st.sidebar:
        st.subheader("Filters")
        models = sorted(df["model"].dropna().unique().tolist())
        chosen_models = st.multiselect("Judge model", models, default=models)
        df = df[df["model"].isin(chosen_models)]
        if df.empty:
            st.warning("No rows match filters.")
            return

    # ---- Top-line metrics ----
    overall = df["humane_score"].mean()
    n = len(df)
    n_violations = int(df["global_violations"].map(len).sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HumaneScore (avg)", f"{overall:+.2f}", help="Mean over all 8 principles, all rows.")
    col2.metric("Conversations scored", n)
    col3.metric("Global violations", n_violations)
    col4.metric("Judge confidence (avg)", f"{df['confidence'].mean():.2f}")

    st.divider()

    # ---- Per-principle averages ----
    st.subheader("Per-principle average score")
    import altair as alt

    ordered_labels = [PRINCIPLE_LABELS[p] for p in PRINCIPLES]
    principle_means = pd.DataFrame(
        {
            "Principle": ordered_labels,
            "Avg score": [df[p].mean() for p in PRINCIPLES],
        }
    )
    principle_chart = (
        alt.Chart(principle_means)
        .mark_bar()
        .encode(
            x=alt.X("Principle:N", sort=ordered_labels, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Avg score:Q", scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition("datum['Avg score'] >= 0", alt.value("#1f77b4"), alt.value("#c0392b")),
        )
        .properties(height=320)
    )
    st.altair_chart(principle_chart, use_container_width=True)

    # ---- Distribution ----
    st.subheader("HumaneScore distribution")
    bins = [-1.01, -0.5, 0, 0.5, 1.01]
    labels = ["Violation [-1.0, -0.5)", "Concerning [-0.5, 0)", "Acceptable [0, 0.5)", "Exemplary [0.5, 1.0]"]
    df["band"] = pd.cut(df["humane_score"], bins=bins, labels=labels, include_lowest=True)
    band_counts = (
        df["band"].value_counts().reindex(labels).fillna(0).astype(int).reset_index()
    )
    band_counts.columns = ["Band", "Count"]
    band_chart = (
        alt.Chart(band_counts)
        .mark_bar()
        .encode(
            x=alt.X("Band:N", sort=labels),
            y=alt.Y("Count:Q"),
            color=alt.Color("Band:N", scale=alt.Scale(domain=labels, range=["#c0392b", "#d97a00", "#7eb238", "#0a8f3f"]), legend=None),
        )
        .properties(height=240)
    )
    st.altair_chart(band_chart, use_container_width=True)

    # ---- Lowest-scoring drilldown ----
    st.subheader("Lowest-scoring conversations")
    bottom = df.sort_values("humane_score").head(8)
    for _, row in bottom.iterrows():
        with st.expander(
            f"[{row['humane_score']:+.2f}] {row['id']}  ·  {row['user_prompt'][:80]}",
            expanded=False,
        ):
            st.markdown(f"**User prompt:** {row['user_prompt']}")
            st.markdown(f"**Response:** {row['response']}")
            score_cols = st.columns(8)
            for col, p in zip(score_cols, PRINCIPLES):
                val = row[p]
                col.markdown(
                    f"<div style='text-align:center;padding:6px;border-radius:4px;"
                    f"background:{score_color(val)};color:white;font-size:0.85em;'>"
                    f"{PRINCIPLE_LABELS[p].split()[0]}<br><b>{val:+.1f}</b></div>",
                    unsafe_allow_html=True,
                )
            if row["global_violations"]:
                st.markdown("**Global violations:** " + "; ".join(row["global_violations"]))
            if row["rationales"]:
                st.markdown("**Rationales:**")
                for principle, rationale in row["rationales"].items():
                    st.markdown(f"- _{PRINCIPLE_LABELS[principle]}_: {rationale}")

    st.divider()

    # ---- Full results table ----
    with st.expander("All results"):
        display_cols = ["id", "humane_score", "confidence", "principle_focus", "expected_severity"] + PRINCIPLES
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
