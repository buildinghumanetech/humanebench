#!/usr/bin/env python3
"""Compare a custom-system-prompt run against its no-system-prompt baseline.

Both logs come from src/custom_prompt_task.py and are scored under rubric v4.

    python scripts/compare_prompt.py                       # newest pair in ./logs
    python scripts/compare_prompt.py --log-dir some/logs
    python scripts/compare_prompt.py --baseline A.eval --custom B.eval

Per principle it prints the baseline mean, the custom-prompt mean, the delta and
the number of in-scope (scored) samples behind each mean. A sample the judges
returned not_applicable / insufficient_context / covered on, or scored only at
low confidence, is not in scope: it enters no mean and is counted separately. A
principle with no in-scope sample has no mean, and it is shown as "-" rather
than 0.

The prompt text is never printed. Only its sha256 is.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanebench.scorer_v4 import (  # noqa: E402
    PRINCIPLE_CODES,
    RUBRIC_VERSION,
    STATUS_QUOTE_UNVERIFIED,
    SAMPLE_JUDGE_FAILED,
    SAMPLE_SCORED,
)

def label(rubric_version: str = RUBRIC_VERSION) -> str:
    return f"HumaneBench rubric {rubric_version}, not comparable to the v1 leaderboard."
SCORER_NAME = "overseer_v4"
CONTEXT_BLOCKED_DIRECTIONAL = 0.15
# Below this many in-scope samples in either condition, a per-principle delta is
# shown but flagged: at that size it is too noisy to read as a finding.
NOISY_N = 10
PRINCIPLES = list(PRINCIPLE_CODES)  # dataset ids, kebab-case, canonical order


@dataclass
class SampleResult:
    id: str
    principle: str
    status: str  # scored | not_scored | judge_failed | errored
    value: float | None
    judge_statuses: list[str] = field(default_factory=list)


@dataclass
class PrincipleSummary:
    mean: float | None
    scored: int
    not_scored: int
    failed: int
    judge_votes: dict[str, int]

    @property
    def context_blocked_rate(self) -> float | None:
        """insufficient_context votes over votes where the principle was in scope."""
        in_scope = sum(n for s, n in self.judge_votes.items() if s != "not_applicable")
        if not in_scope:
            return None
        return self.judge_votes.get("insufficient_context", 0) / in_scope


@dataclass
class ConditionSummary:
    principles: dict[str, PrincipleSummary]

    @property
    def humane_score(self) -> float | None:
        means = [p.mean for p in self.principles.values() if p.mean is not None]
        return sum(means) / len(means) if means else None


def summarize(results: list[SampleResult]) -> ConditionSummary:
    out: dict[str, PrincipleSummary] = {}
    for principle in PRINCIPLES:
        rows = [r for r in results if r.principle == principle]
        values = [
            r.value for r in rows
            if r.status == SAMPLE_SCORED and r.value is not None and not math.isnan(r.value)
        ]
        votes: dict[str, int] = {}
        for r in rows:
            for s in r.judge_statuses:
                votes[s] = votes.get(s, 0) + 1
        out[principle] = PrincipleSummary(
            mean=sum(values) / len(values) if values else None,
            scored=len(values),
            not_scored=sum(1 for r in rows if r.status not in (SAMPLE_SCORED, SAMPLE_JUDGE_FAILED, "errored")),
            failed=sum(1 for r in rows if r.status in (SAMPLE_JUDGE_FAILED, "errored")),
            judge_votes=votes,
        )
    return ConditionSummary(out)


def delta(baseline: float | None, custom: float | None) -> float | None:
    if baseline is None or custom is None:
        return None
    return custom - baseline


@dataclass
class Comparison:
    baseline: ConditionSummary
    custom: ConditionSummary

    def principle_delta(self, principle: str) -> float | None:
        return delta(self.baseline.principles[principle].mean, self.custom.principles[principle].mean)

    @property
    def overall_delta(self) -> float | None:
        return delta(self.baseline.humane_score, self.custom.humane_score)

    @property
    def worse(self) -> list[str]:
        return [p for p in PRINCIPLES if (d := self.principle_delta(p)) is not None and d < 0]


def compare(baseline: list[SampleResult], custom: list[SampleResult]) -> Comparison:
    return Comparison(summarize(baseline), summarize(custom))


# --- reading logs -------------------------------------------------------------

def results_from_log(log) -> list[SampleResult]:
    results = []
    for s in log.samples or []:
        principle = str(s.target if isinstance(s.target, str) else s.target[0])
        score = (s.scores or {}).get(SCORER_NAME)
        if score is None:
            results.append(SampleResult(str(s.id), principle, "errored", None))
            continue
        md = score.metadata or {}
        value = score.value if isinstance(score.value, (int, float)) else None
        results.append(SampleResult(
            str(s.id), principle, md.get("status", "errored"),
            None if value is None or math.isnan(value) else float(value),
            list(md.get("judge_statuses") or []),
        ))
    return results


def _task_name(log) -> str:
    return log.eval.task.split("/")[-1]


def find_pair(log_dir: Path):
    from inspect_ai.log import list_eval_logs, read_eval_log

    infos = list_eval_logs(str(log_dir))  # newest first
    headers = [read_eval_log(i, header_only=True) for i in infos]
    customs = [h for h in headers if _task_name(h) == "custom_prompt_eval" and h.status == "success"]
    if not customs:
        raise SystemExit(f"No successful custom_prompt_eval log in {log_dir}")
    custom = customs[0]
    for h in headers:
        if (_task_name(h) == "baseline_v4_eval" and h.status == "success"
                and h.eval.model == custom.eval.model
                and (h.eval.metadata or {}).get("per_principle") == (custom.eval.metadata or {}).get("per_principle")
                and (h.eval.metadata or {}).get("seed") == (custom.eval.metadata or {}).get("seed")
                and (h.eval.metadata or {}).get("rubric_version")
                == (custom.eval.metadata or {}).get("rubric_version")):
            return h.location, custom.location
    raise SystemExit(
        f"No successful baseline_v4_eval log in {log_dir} for model {custom.eval.model} "
        "with the same per_principle, seed and rubric version"
    )


def check_pair(baseline_log, custom_log) -> list[str]:
    """Hard errors for a pair that cannot be compared; warnings are returned."""
    for log, want in ((baseline_log, "baseline_v4_eval"), (custom_log, "custom_prompt_eval")):
        if _task_name(log) != want:
            raise SystemExit(f"{log.location} is {_task_name(log)}, expected {want}")
        if not str((log.eval.metadata or {}).get("rubric_version", "")).startswith("v4"):
            raise SystemExit(f"{log.location} was not scored under rubric v4")
    b_ver = (baseline_log.eval.metadata or {}).get("rubric_version")
    c_ver = (custom_log.eval.metadata or {}).get("rubric_version")
    if b_ver != c_ver:
        raise SystemExit(f"Different rubric versions: baseline {b_ver} vs custom {c_ver}. Re-run the baseline.")
    if baseline_log.eval.model != custom_log.eval.model:
        raise SystemExit(
            f"Different models: baseline {baseline_log.eval.model} vs custom {custom_log.eval.model}"
        )
    warnings = []
    b_md, c_md = baseline_log.eval.metadata or {}, custom_log.eval.metadata or {}
    if b_md.get("judge_prompt_sha256") != c_md.get("judge_prompt_sha256"):
        raise SystemExit("The two logs used different versions of rubrics/judge_prompt_v4.md")
    b_ids = {str(s.id) for s in baseline_log.samples or []}
    c_ids = {str(s.id) for s in custom_log.samples or []}
    if b_ids != c_ids:
        raise SystemExit(
            f"Different samples: {len(b_ids - c_ids)} only in baseline, {len(c_ids - b_ids)} only in custom. "
            "Run both with the same per_principle, seed and --limit."
        )
    if b_md.get("system_prompt_sha256") not in (None, c_md.get("system_prompt_sha256")):
        warnings.append("The baseline was run alongside a different system prompt (sha256 differs). "
                        "That is fine, a baseline is prompt-independent, but check it is the one you meant.")
    return warnings


# --- output -------------------------------------------------------------------

def _fmt(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def format_report(cmp: Comparison, model: str = "", prompt_sha: str | None = None,
                  warnings: list[str] | None = None, rubric_version: str = RUBRIC_VERSION) -> str:
    lines = [label(rubric_version), ""]
    if model:
        lines.append(f"Model:          {model}")
    if prompt_sha:
        lines.append(f"Prompt sha256:  {prompt_sha}")
    if model or prompt_sha:
        lines.append("")
    header = f"{'Principle':<34}{'Baseline':>9}{'Custom':>9}{'Delta':>8}{'n base':>8}{'n cust':>8}"
    lines += [header, "-" * len(header)]
    directional = []
    noisy = set()
    for p in PRINCIPLES:
        b, c = cmp.baseline.principles[p], cmp.custom.principles[p]
        flag = ""
        if min(b.scored, c.scored) < NOISY_N:
            noisy.add(p)
            flag = "  noisy"
        lines.append(
            f"{p:<34}{_fmt(b.mean):>9}{_fmt(c.mean):>9}{_fmt(cmp.principle_delta(p), True):>8}"
            f"{b.scored:>8}{c.scored:>8}{flag}"
        )
        for name, s in (("baseline", b), ("custom", c)):
            rate = s.context_blocked_rate
            if rate is not None and rate > CONTEXT_BLOCKED_DIRECTIONAL:
                directional.append(f"{p} ({name}, {rate:.0%} context-blocked)")
    lines.append("-" * len(header))
    lines.append(
        f"{'HumaneScore (mean of principles)':<34}{_fmt(cmp.baseline.humane_score):>9}"
        f"{_fmt(cmp.custom.humane_score):>9}{_fmt(cmp.overall_delta, True):>8}"
    )
    lines.append("")

    d = cmp.overall_delta
    if d is None:
        lines.append("Overall: not enough in-scope samples to compare.")
    elif d > 0:
        lines.append(f"Overall: the prompt made this model MORE humane than no prompt ({d:+.2f}).")
    elif d < 0:
        lines.append(f"Overall: the prompt made this model LESS humane than no prompt ({d:+.2f}).")
    else:
        lines.append("Overall: no change from no prompt.")
    worse = cmp.worse
    lines.append("Got worse: " + (", ".join(
        f"{p} ({_fmt(cmp.principle_delta(p), True)}{', noisy' if p in noisy else ''})" for p in worse
    ) if worse else "none"))
    if noisy:
        lines.append(
            f"Rows marked noisy have fewer than {NOISY_N} in-scope samples in a condition. Their deltas are "
            "too noisy to interpret; judge the prompt on the overall delta, or rerun with more samples."
        )

    not_scored_b = sum(s.not_scored for s in cmp.baseline.principles.values())
    not_scored_c = sum(s.not_scored for s in cmp.custom.principles.values())
    failed_b = sum(s.failed for s in cmp.baseline.principles.values())
    failed_c = sum(s.failed for s in cmp.custom.principles.values())
    lines.append("")
    lines.append(
        f"Out of scope (not_applicable / insufficient_context / covered / low confidence / "
        f"unverified quote on every judge), "
        f"excluded from means: baseline {not_scored_b}, custom {not_scored_c}."
    )
    if failed_b or failed_c:
        lines.append(f"Judge or sample failures, excluded: baseline {failed_b}, custom {failed_c}.")
    unverified_b = sum(s.judge_votes.get(STATUS_QUOTE_UNVERIFIED, 0) for s in cmp.baseline.principles.values())
    unverified_c = sum(s.judge_votes.get(STATUS_QUOTE_UNVERIFIED, 0) for s in cmp.custom.principles.values())
    if unverified_b or unverified_c:
        lines.append(
            f"Negative judge scores dropped because no quote matched the response: "
            f"baseline {unverified_b}, custom {unverified_c}."
        )
    if directional:
        lines.append("Directional only (over 15% of in-scope judge votes context-blocked): " + "; ".join(directional))
    lines.append("n = in-scope samples behind each mean. Small n means a noisy per-principle delta.")
    for w in warnings or []:
        lines.append(f"Warning: {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="logs", help="where to look for the newest pair (default: logs)")
    ap.add_argument("--baseline", help="baseline_v4_eval log")
    ap.add_argument("--custom", help="custom_prompt_eval log")
    args = ap.parse_args(argv)

    from inspect_ai.log import read_eval_log

    if bool(args.baseline) != bool(args.custom):
        ap.error("pass both --baseline and --custom, or neither")
    if args.baseline:
        b_path, c_path = args.baseline, args.custom
    else:
        b_path, c_path = find_pair(Path(args.log_dir))
    baseline_log, custom_log = read_eval_log(b_path), read_eval_log(c_path)
    warnings = check_pair(baseline_log, custom_log)
    cmp = compare(results_from_log(baseline_log), results_from_log(custom_log))
    print(format_report(
        cmp,
        model=custom_log.eval.model,
        prompt_sha=(custom_log.eval.metadata or {}).get("system_prompt_sha256"),
        warnings=warnings,
        rubric_version=(custom_log.eval.metadata or {}).get("rubric_version", RUBRIC_VERSION),
    ))
    print(f"\nBaseline log: {b_path}\nCustom log:   {c_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
