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

import altair as alt
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
def load_results(path: Path, _mtime: float) -> pd.DataFrame:
    """Read a results JSONL into a DataFrame.

    ``_mtime`` is part of the cache key so re-running ``batch_evaluate.py``
    (which changes the file's mtime) invalidates the cache automatically.
    """
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


def score_color(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "#888"
    if score >= 0.75:
        return "#0a8f3f"
    if score >= 0.25:
        return "#7eb238"
    if score >= -0.25:
        return "#d6b400"
    if score >= -0.75:
        return "#d97a00"
    return "#c0392b"


def fmt_score(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "—"
    return f"{score:+.1f}"


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

    df = load_results(results_path, results_path.stat().st_mtime)
    if df.empty:
        st.warning("No successful evaluations in this file.")
        return

    # ---- Filters ----
    with st.sidebar:
        st.subheader("Filters")
        models = sorted(df["model"].dropna().unique().tolist())
        chosen_models = st.multiselect("Judge model", models, default=models)
        chosen_principles = st.multiselect(
            "Principle",
            PRINCIPLES,
            default=PRINCIPLES,
            format_func=lambda p: PRINCIPLE_LABELS[p],
            help="Narrow all charts and the drilldown to a subset of principles.",
        )
        df = df[df["model"].isin(chosen_models)]
        if df.empty or not chosen_principles:
            st.warning("No rows match filters.")
            return

    # Score across the *selected* principles, not always all 8.
    df = df.copy()
    df["filtered_score"] = df[chosen_principles].mean(axis=1)
    multi_model = len(chosen_models) > 1
    single_principle = len(chosen_principles) == 1

    # ---- Top-line metrics ----
    overall = df["filtered_score"].mean()
    n_rows = len(df)
    n_unique = df["id"].nunique()
    n_violations = int(df["global_violations"].map(len).sum())
    score_label = (
        f"{PRINCIPLE_LABELS[chosen_principles[0]]} score"
        if single_principle
        else "HumaneScore (avg)"
    )
    score_help = (
        f"Mean of `{chosen_principles[0]}` across selected rows."
        if single_principle
        else f"Mean over the {len(chosen_principles)} selected principles, all selected rows."
    )
    rows_label = (
        f"Scored rows ({n_unique} unique)" if multi_model else "Conversations scored"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(score_label, f"{overall:+.2f}", help=score_help)
    col2.metric(rows_label, n_rows)
    col3.metric("Global violations", n_violations)
    col4.metric("Judge confidence (avg)", f"{df['confidence'].mean():.2f}")

    st.divider()

    # ---- Per-principle averages ----
    st.subheader("Per-principle average score")
    ordered_principles = [p for p in PRINCIPLES if p in chosen_principles]
    ordered_labels = [PRINCIPLE_LABELS[p] for p in ordered_principles]

    if multi_model:
        # Grouped bars: one cluster per principle, one bar per judge model.
        rows = []
        for model in chosen_models:
            model_df = df[df["model"] == model]
            for principle in ordered_principles:
                rows.append(
                    {
                        "Principle": PRINCIPLE_LABELS[principle],
                        "Judge": model,
                        "Avg score": model_df[principle].mean(),
                    }
                )
        principle_means = pd.DataFrame(rows)
        principle_chart = (
            alt.Chart(principle_means)
            .mark_bar()
            .encode(
                x=alt.X("Principle:N", sort=ordered_labels, axis=alt.Axis(labelAngle=-30, title=None)),
                xOffset=alt.XOffset("Judge:N", sort=chosen_models),
                y=alt.Y("Avg score:Q", scale=alt.Scale(domain=[-1, 1])),
                color=alt.Color("Judge:N", sort=chosen_models),
                tooltip=["Principle", "Judge", alt.Tooltip("Avg score:Q", format="+.2f")],
            )
            .properties(height=360)
        )
    else:
        principle_means = pd.DataFrame(
            {
                "Principle": ordered_labels,
                "Avg score": [df[p].mean() for p in ordered_principles],
            }
        )
        principle_chart = (
            alt.Chart(principle_means)
            .mark_bar()
            .encode(
                x=alt.X("Principle:N", sort=ordered_labels, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("Avg score:Q", scale=alt.Scale(domain=[-1, 1])),
                color=alt.condition("datum['Avg score'] >= 0", alt.value("#1f77b4"), alt.value("#c0392b")),
                tooltip=["Principle", alt.Tooltip("Avg score:Q", format="+.2f")],
            )
            .properties(height=320)
        )
    st.altair_chart(principle_chart, use_container_width=True)

    # ---- Distribution ----
    dist_title = (
        f"{PRINCIPLE_LABELS[chosen_principles[0]]} score distribution"
        if single_principle
        else "HumaneScore distribution"
    )
    st.subheader(dist_title)
    # Midpoint bins so each rubric anchor (-1.0, -0.5, 0.5, 1.0) snaps to
    # its named band. Averaged scores fall into whichever anchor is nearest.
    bins = [-1.001, -0.75, 0, 0.75, 1.001]
    labels = ["Violation", "Concerning", "Acceptable", "Exemplary"]
    df["band"] = pd.cut(df["filtered_score"], bins=bins, labels=labels, include_lowest=True)
    palette = ["#c0392b", "#d97a00", "#7eb238", "#0a8f3f"]

    if multi_model:
        band_counts = (
            df.groupby(["model", "band"], observed=False)
            .size()
            .reset_index(name="Count")
            .rename(columns={"model": "Judge", "band": "Band"})
        )
        band_chart = (
            alt.Chart(band_counts)
            .mark_bar()
            .encode(
                x=alt.X("Band:N", sort=labels, axis=alt.Axis(labelAngle=0, title=None)),
                xOffset=alt.XOffset("Judge:N", sort=chosen_models),
                y=alt.Y("Count:Q"),
                color=alt.Color("Judge:N", sort=chosen_models),
                tooltip=["Band", "Judge", "Count"],
            )
            .properties(height=260)
        )
    else:
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
                color=alt.Color("Band:N", scale=alt.Scale(domain=labels, range=palette), legend=None),
                tooltip=["Band", "Count"],
            )
            .properties(height=240)
        )
    st.altair_chart(band_chart, use_container_width=True)

    # ---- Lowest-scoring drilldown ----
    drill_title = "Lowest-scoring conversations"
    if single_principle:
        drill_title += f" on {PRINCIPLE_LABELS[chosen_principles[0]]}"
    elif len(chosen_principles) < len(PRINCIPLES):
        drill_title += f" (mean of {len(chosen_principles)} selected principles)"
    st.subheader(drill_title)
    bottom = df.sort_values("filtered_score").head(8)
    for _, row in bottom.iterrows():
        score_pretty = f"{row['filtered_score']:+.2f}"
        suffix = f"  ·  judge: {row['model']}" if multi_model else ""
        with st.expander(
            f"[{score_pretty}] {row['id']}  ·  {row['user_prompt'][:80]}{suffix}",
            expanded=False,
        ):
            st.markdown(f"**User prompt:** {row['user_prompt']}")
            st.markdown(f"**Response:** {row['response']}")
            if multi_model:
                st.markdown(f"**Judge:** `{row['model']}`")
            score_cols = st.columns(8)
            for col, p in zip(score_cols, PRINCIPLES):
                val = row[p]
                dim = "" if p in chosen_principles else "opacity:0.35;"
                col.markdown(
                    f"<div style='text-align:center;padding:6px;border-radius:4px;"
                    f"background:{score_color(val)};color:white;font-size:0.85em;{dim}'>"
                    f"{PRINCIPLE_LABELS[p].split()[0]}<br><b>{fmt_score(val)}</b></div>",
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
        display_cols = ["id", "model", "filtered_score", "confidence", "principle_focus", "expected_severity"] + PRINCIPLES
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
