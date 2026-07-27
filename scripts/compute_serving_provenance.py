#!/usr/bin/env python3
"""Which serving stack actually produced each response?

A model slug is not a system. OpenRouter is a router: a single slug can be
fulfilled by several upstream providers, and for open-weight models those
providers differ in quantisation and serving configuration in ways that
measurably change behaviour. The eval logs record the slug in
``eval.model``, but the *provider that answered* is only in the raw API
response, which is where this script reads it from.

Two things it recovers that nothing else in the pipeline reports:

1. **Provider mixture per run.** If one run was served by four providers, that
   cell is a mixture of four serving stacks, not one system. The paper should
   say so rather than let a reviewer discover it in the released logs.
2. **``system_fingerprint``**, where the upstream exposes it. This is the only
   per-response version identifier available, and it is null for every provider
   except OpenAI.

Generation calls are separated from judge calls by POSITION -- the first model
event of a sample is the generation, every later one is a judge. A slug
comparison would be wrong: all three ensemble judges are themselves evaluated
models, so on those runs the model's self-judge call looks identical to its
generation call.

Run from repo root:
    python scripts/compute_serving_provenance.py
    python scripts/compute_serving_provenance.py --personas decomp_b_xml_objective
    python scripts/compute_serving_provenance.py --csv-out tables/

``--csv-out`` additionally exports the per-call rows behind the report's
percentages, so the provider attribution survives outside the .eval logs --
which are 584 MB of already-compressed archives and are not distributed with
the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench import provenance as prov  # noqa: E402


def scan(
    path: Path,
    eval_model: str,
    gen_rows: list | None = None,
    judge_pairs: Counter | None = None,
    persona: str | None = None,
    model_dir_name: str | None = None,
) -> tuple[Counter, Counter, Counter, Counter]:
    """Return (gen providers, gen fingerprints, judge providers, judge fps).

    Generation is identified by POSITION, not by slug. All three ensemble judges
    are themselves evaluated models (claude-sonnet-4.5, gpt-5.1,
    gemini-2.5-pro), so on those three runs a slug comparison books the model's
    self-judge call as a generation call -- inflating the generation provider
    mixture and understating the judge one. The generation call is the first
    model event of each sample; every later one is a judge.

    When `gen_rows` / `judge_pairs` are supplied they are filled in as a side
    effect, for the per-call CSV export. They do not touch the counters the
    markdown report is built from, so passing them cannot move a published
    number.
    """
    gen_p, gen_f, jud_p, jud_f = Counter(), Counter(), Counter(), Counter()
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.startswith("samples/"):
                continue
            sample = json.loads(z.read(name))
            seen_gen = False
            for ev in sample.get("events") or []:
                if ev.get("event") != "model":
                    continue
                resp = (ev.get("call") or {}).get("response")
                if not isinstance(resp, dict):
                    continue
                is_gen = (not seen_gen) and ev.get("model") == eval_model
                if is_gen:
                    seen_gen = True
                p, f = resp.get("provider"), resp.get("system_fingerprint")
                if is_gen:
                    gen_p[p] += 1
                    gen_f[f] += 1
                    if gen_rows is not None:
                        gen_rows.append({
                            "persona": persona,
                            "model": model_dir_name,
                            "sample_id": sample.get("id"),
                            "provider": p,
                            "system_fingerprint": f,
                        })
                else:
                    jud_p[f"{ev.get('model')} <- {p}"] += 1
                    jud_f[f] += 1
                    if judge_pairs is not None:
                        judge_pairs[(ev.get("model"), p)] += 1
    return gen_p, gen_f, jud_p, jud_f


def fmt(counter: Counter, top: int = 6) -> str:
    total = sum(counter.values()) or 1
    parts = [f"{k if k is not None else '(none)'} {100*v/total:.0f}%"
             for k, v in counter.most_common(top)]
    extra = len(counter) - top
    return ", ".join(parts) + (f", +{extra} more" if extra > 0 else "")


def write_call_tables(out_dir: Path, gen_rows: list, judge_pairs: Counter) -> None:
    """Write the two per-call CSVs the markdown report summarises.

    The report gives percentages; these give the rows behind them, so a reader
    can check any claim about provider mixture without the .eval logs. Sorted
    deterministically -- the zip that ships them must be reproducible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    responses = out_dir / "serving_provenance_responses.csv"
    with open(responses, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["persona", "model", "sample_id", "provider",
                        "system_fingerprint"],
        )
        w.writeheader()
        for row in sorted(
            gen_rows,
            key=lambda r: (r["persona"] or "", r["model"] or "",
                           str(r["sample_id"])),
        ):
            w.writerow(row)

    judges = out_dir / "serving_provenance_judges.csv"
    with open(judges, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["judge_model", "provider", "n"])
        for (judge_model, provider), n in sorted(
            judge_pairs.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
        ):
            w.writerow([judge_model, provider, n])

    print(f"Wrote {responses} ({len(gen_rows):,} generation calls)")
    print(f"Wrote {judges} ({len(judge_pairs):,} judge x provider pairs, "
          f"{sum(judge_pairs.values()):,} calls)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=prov.LOGS_DIR)
    ap.add_argument("--personas", nargs="*", default=None,
                    help="condition dirs to scan (default: the 3 reported personas)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "tables" / "serving_provenance.md")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="Directory to additionally write two per-call tables "
                         "into: serving_provenance_responses.csv (one row per "
                         "generation call) and serving_provenance_judges.csv "
                         "(judge model x provider counts). These carry the "
                         "provider attribution that would otherwise only be "
                         "readable inside the undistributable .eval logs.")
    args = ap.parse_args()

    personas = args.personas or list(prov.PERSONAS)
    per_model_gen: dict[str, Counter] = defaultdict(Counter)
    per_model_fp: dict[str, Counter] = defaultdict(Counter)
    judge_gen: Counter = Counter()
    judge_fp: Counter = Counter()
    n_runs = 0
    gen_rows: list | None = [] if args.csv_out else None
    judge_pairs: Counter | None = Counter() if args.csv_out else None

    for persona in personas:
        pdir = args.logs_dir / persona
        if not pdir.is_dir():
            print(f"[warn] missing {pdir}")
            continue
        for model_dir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            evals = sorted(model_dir.glob("*.eval"))
            if not evals:
                continue
            # Newest-by-name is the wrong criterion: a re-run that died early
            # leaves a fresh but truncated log beside a complete one. Take the
            # log with the most samples.
            path = max(evals, key=lambda q: sum(1 for _ in prov.iter_eval_samples(q)))
            eval_model = prov.read_eval_header(path)["eval"].get("model")
            print(f"  scanning {persona}/{model_dir.name} ...", flush=True)
            g, gf, j, jf = scan(
                path,
                eval_model,
                gen_rows=gen_rows,
                judge_pairs=judge_pairs,
                persona=persona,
                model_dir_name=model_dir.name,
            )
            per_model_gen[model_dir.name] += g
            per_model_fp[model_dir.name] += gf
            judge_gen += j
            judge_fp += jf
            n_runs += 1

    L = ["# Serving provenance: which stack answered each call\n"]
    L.append(
        "A model slug is not a system. OpenRouter routes a slug to one of several "
        "upstream providers, and for open-weight models those providers differ in "
        "quantisation and serving configuration. The logs record the slug; the "
        "provider that actually answered is recoverable only from the raw API "
        "response, which is what this table reads.\n"
    )
    L.append(f"Scanned **{n_runs} runs** across conditions: {', '.join(personas)}.\n")

    L.append("## Generation calls, by evaluated model\n")
    L.append("| model | providers that served it | distinct | `system_fingerprint` |")
    L.append("| --- | --- | ---: | --- |")
    for m in sorted(per_model_gen):
        L.append(f"| {m} | {fmt(per_model_gen[m])} | {len(per_model_gen[m])} | "
                 f"{fmt(per_model_fp[m], 3)} |")
    L.append("")

    multi = {m: c for m, c in per_model_gen.items() if len(c) > 1}
    if multi:
        L.append(
            f"**{len(multi)} of {len(per_model_gen)} models were served by more than "
            "one provider.** Those cells are a mixture of serving stacks rather than "
            "a single system. This is a property of the reported runs, not something "
            "introduced later, and it is visible to anyone who opens the released "
            "logs.\n"
        )
        for m, c in sorted(multi.items(), key=lambda kv: -len(kv[1])):
            L.append(f"- `{m}`: {len(c)} providers — {fmt(c)}")
        L.append("")

    L.append("## Judge calls\n")
    L.append("The judge ensemble is routed the same way. This exposure is identical "
             "across conditions, so it does not bias a persona contrast, but it does "
             "bound how exactly any single judge is reproducible.\n")
    L.append("| judge <- provider | share |")
    L.append("| --- | ---: |")
    tot = sum(judge_gen.values()) or 1
    for k, v in judge_gen.most_common(12):
        L.append(f"| {k} | {100*v/tot:.1f}% |")
    L.append("")
    L.append(f"Judge `system_fingerprint`: {fmt(judge_fp, 4)}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L))
    print(f"\nWrote {args.out}")
    print(f"models served by >1 provider: {len(multi)}/{len(per_model_gen)}")

    if args.csv_out is not None:
        write_call_tables(args.csv_out, gen_rows or [], judge_pairs or Counter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
