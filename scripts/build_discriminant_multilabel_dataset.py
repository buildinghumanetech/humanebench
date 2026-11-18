#!/usr/bin/env python3
"""Build the multi-label scoring datasets for the designed x measured matrix.

Every scenario was *designed* to probe one principle. This analysis asks which
principle it actually *measures*, by scoring the same archived baseline response
against all eight. Rows of the resulting matrix are the designed principle,
columns the scored one; the claim rides on the diagonal being distinct from its
own row, which is a *within-row* contrast and therefore survives a general
model-quality factor and the judge factor collapse documented by Feuer et al.
(arXiv:2509.20293). See `docs/discriminant_validity_design.md`.

This script generates no model output and makes no API call. It reads the
archived baseline runs and emits one Inspect dataset per source model, each
carrying the same 96 responses repeated across the 8 principles.

WHERE THE RESPONSE TEXT COMES FROM, AND WHY IT MATTERS
------------------------------------------------------
The obvious source is ``sample.output.choices[0].message.content``. For the
reasoning models that is a *block list* of `reasoning` and `text` blocks, and
whatever join you pick to flatten it is a guess about what Inspect rendered into
the judge prompt. A guess is not good enough here: the diagonal of this matrix is
supposed to be the main-run procedure re-executed, so the judge must receive the
same bytes it received in November.

So the response is recovered from the archived judge prompt itself. The scoring
template is a pure function

    judge_prompt = OVERSEER_TEMPLATE.format(issue, prompt, response).strip()

with exactly one variable slot for the response, so rendering the template with a
sentinel in that slot splits it into a known prefix and suffix. The archived
prompt must start with that prefix and end with that suffix, and what lies
between them *is* the response, exactly. That is a solve, not a parse: no
whitespace convention is assumed, and it fails loudly rather than silently
returning something close.

The same step doubles as the round-trip proof the analysis needs. Matching the
prefix proves the rubric scaffold and the scenario prompt are byte-identical to
the main run; matching the suffix proves nothing was appended. It runs against
all 288 (96 scenarios x 3 models) archived prompts and a mismatch is a hard stop.

The manifest records the expected judge-prompt sha256 for every one of the 2,304
(scenario, model, scored principle) calls, so `verify_discriminant_provenance.py`
can check the whole run afterwards rather than a diagonal sample of it.

Inputs (read-only):
  - data/decomposition/discriminant_96_ids.txt   (frozen draw, nested in the 200)
  - data/decomposition/humane_bench_discriminant_96.jsonl
  - logs/baseline/<model>/*.eval                 (archived responses)

Outputs (written to --output-dir, default data/discriminant/):
  - multilabel_<model>.jsonl    768 rows each: 96 scenarios x 8 principles
  - expected_prompt_hashes.csv  2,304 rows
  - manifest.json

Run from repo root:
    python scripts/build_discriminant_multilabel_dataset.py
"""
# Paper: produces data/discriminant/multilabel_<model>.jsonl, expected_prompt_hashes.csv and
#        manifest.json - the 96 scenarios x 8 principles multi-label scoring datasets behind the
#        designed x scored matrix (main paper, "Principle Separability").
# Paper: implements recovery of each archived baseline response from its stored judge prompt by
#        splitting the scoring template on its single response slot, so the matrix diagonal
#        re-executes the main scoring run on byte-identical text.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench.bootstrap import PRINCIPLES  # noqa: E402
from humanebench.discriminant import (  # noqa: E402
    CONDITIONS,
    FRAME_JSONL,
    GLOBAL_RULES,
    IDS_PATH,
    JUDGE_MODEL,
    JUDGE_SCORE_ATTEMPTS,
    JUDGE_TEMPERATURE,
    PARENT_IDS_PATH,
    PER_PRINCIPLE,
    SOURCE_MODELS,
    SUMMARY_PATH,
    render_judge_prompt,
    scoring_template,
)
from humanebench.excluded import load_excluded_ids  # noqa: E402
from humanebench.provenance import DATASET_PATH, file_sha256  # noqa: E402

# Must not occur in the rendered scaffold or in any response. Checked, not hoped.
SENTINEL = "\x00__HUMANEBENCH_RESPONSE_SLOT_9f3c__\x00"


def _rel(path: Path) -> str:
    """Repo-relative where possible; absolute otherwise.

    ``--logs-dir`` may point outside the checkout -- a git worktree reading the
    main clone's archived logs, for instance -- and a bare ``relative_to`` raises
    there, aborting the build after the expensive extraction work is done.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _rel_log(path: Path, logs_dir: Path) -> str:
    """A log path as ``logs/<persona>/<model>/<file>``, wherever the logs live.

    This goes into the committed manifest, so it must not depend on the
    operator's directory layout. Falling back to an absolute path would write
    a home directory into a published file -- the exact local-path leak
    ``anonymization_redaction_list.txt`` exists to catch -- and would also make
    the manifest differ between a checkout and a worktree that reads the same
    archives.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(Path("logs") / path.relative_to(logs_dir))


class ExtractionError(RuntimeError):
    """The archived judge prompt did not match the re-rendered template."""


def scaffold_split(principle_id: str, prompt: str) -> tuple[str, str]:
    """Return ``(prefix, suffix)`` around the response slot for one scaffold."""
    rendered = render_judge_prompt(principle_id, prompt, SENTINEL)
    if rendered.count(SENTINEL) != 1:
        raise ExtractionError(
            f"sentinel appears {rendered.count(SENTINEL)} times in the rendered "
            f"template for {principle_id}; it must appear exactly once"
        )
    prefix, suffix = rendered.split(SENTINEL)
    return prefix, suffix


def _resolve(content, attachments: dict) -> str:
    """Inspect stores long message bodies as ``attachment://`` refs."""
    if not isinstance(content, str):
        return ""
    if content.startswith("attachment://"):
        return attachments.get(content.split("://", 1)[1], "")
    return content


def archived_judge_prompts(sample: dict) -> list[str]:
    """Every distinct judge prompt sent for this sample (one per ensemble judge)."""
    att = sample.get("attachments") or {}
    found: list[str] = []
    for ev in sample.get("events") or []:
        if ev.get("event") != "model":
            continue
        for msg in ev.get("input") or []:
            if msg.get("role") != "user":
                continue
            text = _resolve(msg.get("content"), att)
            if GLOBAL_RULES in text:
                found.append(text)
    return found


def extract_response(sample: dict, prompt: str, designed_principle: str) -> tuple[str, int]:
    """Solve the archived judge prompt for the response slot.

    Returns ``(response, n_archived_prompts)``. Raises `ExtractionError` unless
    every archived prompt for the sample is identical and brackets the response
    with the byte-exact scaffold this repo would render today.
    """
    prompts = archived_judge_prompts(sample)
    if not prompts:
        raise ExtractionError(f"{sample.get('id')}: no judge prompt found in events")
    if len(set(prompts)) != 1:
        raise ExtractionError(
            f"{sample.get('id')}: {len(set(prompts))} distinct judge prompts; the "
            "ensemble must have been sent one identical prompt"
        )
    archived = prompts[0]

    prefix, suffix = scaffold_split(designed_principle, prompt)
    if not archived.startswith(prefix):
        raise ExtractionError(
            f"{sample.get('id')}: archived judge prompt does not start with the "
            f"re-rendered scaffold for {designed_principle}. The rubric or the "
            "scenario prompt has changed since the run."
        )
    if not archived.endswith(suffix):
        raise ExtractionError(
            f"{sample.get('id')}: archived judge prompt does not end with the "
            "re-rendered response contract."
        )
    response = archived[len(prefix): len(archived) - len(suffix)]
    if SENTINEL in response:
        raise ExtractionError(f"{sample.get('id')}: response contains the sentinel")

    # Belt and braces: re-render with the recovered response and demand equality.
    if render_judge_prompt(designed_principle, prompt, response) != archived:
        raise ExtractionError(f"{sample.get('id')}: round-trip re-render mismatch")
    return response, len(prompts)


def load_frame(cond=None) -> dict[str, dict]:
    """Return ``{id: row}`` for the condition's frame, validated."""
    if cond is None:
        cond = CONDITIONS["discriminant"]
    ids = [ln.strip() for ln in cond.ids_path.read_text().splitlines() if ln.strip()]
    rows = {}
    with cond.frame_jsonl.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    if set(ids) != set(rows):
        raise SystemExit("ids file and frame jsonl disagree")
    n_expected = cond.per_principle * len(PRINCIPLES)
    if len(ids) != n_expected:
        raise SystemExit(f"expected {n_expected} frozen ids, got {len(ids)}")

    if cond.parent_ids_path is not None:
        parent = {ln.strip() for ln in cond.parent_ids_path.read_text().splitlines()
                  if ln.strip()}
        if not set(ids) <= parent:
            raise SystemExit(
                f"the {len(ids)} are not a subset of {cond.parent_ids_path.name}; "
                "the parent frame was redrawn after this draw."
            )
        summary = json.loads(cond.summary_path.read_text())
        if summary.get("parent_ids_sha256") and \
                summary["parent_ids_sha256"] != file_sha256(cond.parent_ids_path):
            raise SystemExit(
                f"{cond.summary_path.name} records a different parent hash than "
                f"{cond.parent_ids_path.name} has now."
            )
    else:
        original_ids = {ln.strip()
                        for ln in IDS_PATH.read_text().splitlines() if ln.strip()}
        if set(ids) & original_ids:
            raise SystemExit(
                f"expansion frame overlaps with original 96: "
                f"{sorted(set(ids) & original_ids)[:5]}"
            )

    excluded = load_excluded_ids(DATASET_PATH)
    if set(ids) & excluded:
        raise SystemExit(f"frame contains excluded ids: {sorted(set(ids) & excluded)}")

    per_principle = {p: sum(1 for i in ids if rows[i]["target"] == p) for p in PRINCIPLES}
    if set(per_principle.values()) != {cond.per_principle}:
        raise SystemExit(
            f"frame is not {cond.per_principle} per principle: {per_principle}"
        )
    return rows


def eval_path_for(model: str, logs_dir: Path) -> Path:
    paths = sorted((logs_dir / "baseline" / model).glob("*.eval"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one baseline .eval for {model}, found {len(paths)}")
    return paths[0]


def build_model(
    model: str, frame: dict[str, dict], logs_dir: Path,
    hash_template: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Return ``(dataset_rows, hash_rows, provenance)`` for one source model.

    ``hash_template`` is the template used for expected-hash computation. It
    defaults to None (= the deployed OVERSEER_TEMPLATE), which is correct for
    as-deployed conditions. For rules27, pass the variant template so the
    expected hashes match what the scorer will actually send — but extraction
    ALWAYS uses the default template, since responses are solved from November
    archives rendered with the deployed prompt.
    """
    path = eval_path_for(model, logs_dir)
    wanted = set(frame)
    responses: dict[str, str] = {}

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.startswith("samples/"):
                continue
            sample = json.loads(z.read(name))
            sid = sample.get("id")
            if sid not in wanted:
                continue
            row = frame[sid]
            response, _n = extract_response(sample, row["input"], row["target"])
            if not response:
                raise ExtractionError(
                    f"{sid}: extracted response is empty; "
                    "use_pregenerated_output_strict would raise on falsy ai_output"
                )
            responses[sid] = response

    missing = sorted(wanted - set(responses))
    if missing:
        raise SystemExit(f"{model}: {len(missing)} of 96 responses missing: {missing[:5]}")

    dataset_rows: list[dict] = []
    hash_rows: list[dict] = []
    for sid in sorted(responses):
        row = frame[sid]
        meta = row.get("metadata") or {}
        for scored in PRINCIPLES:
            dataset_rows.append({
                "id": f"{sid}|{model}|{scored}",
                "input": row["input"],
                "target": scored,  # selects the rubric in scorer.overseer
                "metadata": {
                    "ai_output": responses[sid],
                    "scenario_id": sid,
                    "source_model": model,
                    "designed_principle": row["target"],
                    "scored_principle": scored,
                    "is_diagonal": scored == row["target"],
                    "domain": meta.get("domain", ""),
                    "vulnerable_population": meta.get("vulnerable-population", ""),
                },
            })
            expected = render_judge_prompt(scored, row["input"], responses[sid],
                                          template=hash_template)
            hash_rows.append({
                "scenario_id": sid,
                "source_model": model,
                "designed_principle": row["target"],
                "scored_principle": scored,
                "is_diagonal": scored == row["target"],
                "expected_judge_prompt_sha256": hashlib.sha256(expected.encode()).hexdigest(),
                # Carried so the launch runner can price the run against live
                # OpenRouter rates instead of a remembered per-call figure.
                "expected_judge_prompt_chars": len(expected),
            })

    provenance = {
        "source_eval": _rel_log(path, logs_dir),
        "source_eval_sha256": file_sha256(path),
        "n_responses": len(responses),
        "response_sha256": hashlib.sha256(
            "".join(responses[s] for s in sorted(responses)).encode()
        ).hexdigest(),
    }
    return dataset_rows, hash_rows, provenance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=list(SOURCE_MODELS))
    ap.add_argument("--condition", default="discriminant",
                    choices=sorted(CONDITIONS),
                    help="which condition to build datasets for")
    args = ap.parse_args()

    cond = CONDITIONS[args.condition]
    if args.output_dir is None:
        args.output_dir = cond.data_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hash_template = scoring_template(cond)

    frame = load_frame(cond)
    print(f"frame: {len(frame)} scenarios, {cond.per_principle} per principle")

    all_hashes: list[dict] = []
    per_model: dict[str, dict] = {}
    for model in args.models:
        rows, hashes, prov = build_model(model, frame, args.logs_dir, hash_template)
        out = args.output_dir / f"multilabel_{model}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        all_hashes.extend(hashes)
        prov["dataset_file"] = _rel(out)
        prov["dataset_file_sha256"] = file_sha256(out)
        prov["n_rows"] = len(rows)
        per_model[model] = prov
        print(f"  {model:22s} {len(rows):4d} rows -> {_rel(out)}")

    hashes_path = args.output_dir / "expected_prompt_hashes.csv"
    with hashes_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_hashes[0].keys()))
        w.writeheader()
        w.writerows(all_hashes)

    n_calls = len(all_hashes)
    total_prompt_chars = sum(r["expected_judge_prompt_chars"] for r in all_hashes)
    summary = json.loads(cond.summary_path.read_text())
    manifest = {
        "schema": "humanebench-discriminant-multilabel/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "design": "designed x measured matrix; baseline responses only",
        "n_scenarios": len(frame),
        "n_source_models": len(args.models),
        "n_principles": len(PRINCIPLES),
        "n_judge_calls": n_calls,
        "one_call_per_principle": True,
        "total_judge_prompt_chars": total_prompt_chars,
        "pricing_calibration": {
            "chars_per_input_token": 4.483,
            "output_tokens_mean": 315,
            "measured_on_n_calls": 287,
            "source": "logs/baseline/<model>/*.eval, gpt-5.1 judge events",
        },
        "frame_ids_file": _rel(cond.ids_path),
        "frame_ids_sha256": file_sha256(cond.ids_path),
        "frame_subset_prompt_hash": summary.get("subset_prompt_hash"),
        "parent_ids_file": summary.get("parent_ids_file"),
        "parent_ids_sha256": summary.get("parent_ids_sha256"),
        "source_dataset_sha256": file_sha256(DATASET_PATH),
        "judge": {
            "models": [JUDGE_MODEL],
            "temperature": JUDGE_TEMPERATURE,
            "score_attempts": JUDGE_SCORE_ATTEMPTS,
            "rationale": (
                "the only main-run ensemble judge that is not among the three "
                "scored models, so the diagonal is a same-judge replication"
            ),
        },
        "expected_prompt_hashes_file": _rel(hashes_path),
        "expected_prompt_hashes_sha256": file_sha256(hashes_path),
        "per_model": per_model,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{n_calls:,} judge calls will be made "
          f"({len(frame)} scenarios x {len(PRINCIPLES)} principles x {len(args.models)} models)")
    print(f"all {len(frame) * len(args.models)} archived judge prompts round-tripped byte-exactly")
    print(f"wrote {_rel(hashes_path)} and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
