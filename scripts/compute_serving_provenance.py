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
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanebench import provenance as prov  # noqa: E402


def scan(path: Path, eval_model: str) -> tuple[Counter, Counter, Counter, Counter]:
    """Return (gen providers, gen fingerprints, judge providers, judge fps).

    Generation is identified by POSITION, not by slug. All three ensemble judges
    are themselves evaluated models (claude-sonnet-4.5, gpt-5.1,
    gemini-2.5-pro), so on those three runs a slug comparison books the model's
    self-judge call as a generation call -- inflating the generation provider
    mixture and understating the judge one. The generation call is the first
    model event of each sample; every later one is a judge.
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
                else:
                    jud_p[f"{ev.get('model')} <- {p}"] += 1
                    jud_f[f] += 1
    return gen_p, gen_f, jud_p, jud_f


def fmt(counter: Counter, top: int = 6) -> str:
    total = sum(counter.values()) or 1
    parts = [f"{k if k is not None else '(none)'} {100*v/total:.0f}%"
             for k, v in counter.most_common(top)]
    extra = len(counter) - top
    return ", ".join(parts) + (f", +{extra} more" if extra > 0 else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", type=Path, default=prov.LOGS_DIR)
    ap.add_argument("--personas", nargs="*", default=None,
                    help="condition dirs to scan (default: the 3 reported personas)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "tables" / "serving_provenance.md")
    args = ap.parse_args()

    personas = args.personas or list(prov.PERSONAS)
    per_model_gen: dict[str, Counter] = defaultdict(Counter)
    per_model_fp: dict[str, Counter] = defaultdict(Counter)
    judge_gen: Counter = Counter()
    judge_fp: Counter = Counter()
    n_runs = 0

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
            g, gf, j, jf = scan(path, eval_model)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
