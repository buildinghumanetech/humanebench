#!/usr/bin/env python3
"""Build the anonymous supplementary package for AAAI-27.

The submission may not link to anything outside itself, so the code, the data
appendix and every derived table travel as one zip, under 100 MB, with nothing
in it that identifies an author or their institution.

Pipeline: **stage -> scrub -> verify -> zip.**

  stage    copy in exactly the files scripts/supplementary/manifest.json names.
           Inclusion, not exclusion: a file added to the repo tomorrow cannot
           reach the package by accident.
  scrub    apply scripts/supplementary/scrub_rules.json to the STAGED COPIES.
           The repository is never edited. Every rule declares how many times
           it must match, and a mismatch fails the build -- a rule that stops
           matching means the repo moved and the rule list is stale, which is
           the moment a leak would slip through.
  verify   six gates, every one of them, every build. A gate that fails is a
           bug to fix, not a threshold to lower.
  zip      deterministic: sorted entries, fixed timestamps and modes, so two
           builds of the same tree are byte-identical and the package can be
           checked against a recorded hash.

Usage:
    python scripts/build_supplementary.py
    python scripts/build_supplementary.py --aux-root /path/to/main/checkout

`--aux-root` says where to find artifacts that are gitignored in the repo but
belong in the package (the decomposition tables, the discriminant JSONLs, the
two results/ documents). It defaults to the repo this script lives in, which is
what the final build uses -- aux == repo removes any chance of pulling a stale
copy from somewhere else.

build_report.txt is written NEXT TO the zip and never inside it: it records
local paths, which are exactly what the package must not contain.
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "scripts" / "supplementary"

# Terms every staged byte is checked against, on top of the repo's redaction
# list. The list covers people and the organisation; these cover the artifacts
# that would let a reviewer look them up.
EXTRA_SCAN_TERMS = [
    "zenodo",
    # The DOI prefix carries its slash on purpose. Bare "10.5281" is a
    # substring of PDF coordinates like "10.528125" and fires on figures that
    # contain no DOI at all; a DOI is always 10.NNNN/suffix. "zenodo" above
    # still catches the real one independently, so nothing is given up.
    "10.5281/",
    "github.com",
    "4open.science",
    "/Users/",
    "buildinghumanetech",
    "humanetech",
]

# Deliberately empty. An entry here would let identifying text through, so
# adding one is a decision for the authors, not for this script.
SCAN_WHITELIST: list[str] = []

MAX_FILE_BYTES = 25 * 1024 * 1024
ZIP_HARD_LIMIT = 100 * 1024 * 1024
ZIP_WARN_LIMIT = 50 * 1024 * 1024

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class BuildError(RuntimeError):
    """A gate failed or the configuration no longer matches the repository."""


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------

def is_excluded(rel: str, patterns: list[str]) -> bool:
    """Match a staged-relative POSIX path against the manifest's exclude globs.

    Each pattern is tried against the whole path and against every path
    suffix, so `scripts/deprecated/**` excludes the directory itself as well
    as its contents, and a bare pattern like `.*` or `*.pyc` matches at any
    depth without needing a `**/` prefix.
    """
    for pat in patterns:
        if pat.startswith("COMMENT:"):
            continue
        if fnmatch.fnmatch(rel, pat):
            return True
        # `a/b/**` should also drop `a/b` itself.
        if pat.endswith("/**") and fnmatch.fnmatch(rel, pat[:-3]):
            return True
        # Match a pattern anchored at any directory level, e.g. `**/*.pyc`.
        parts = rel.split("/")
        for i in range(len(parts)):
            if fnmatch.fnmatch("/".join(parts[i:]), pat):
                return True
    return False


def iter_source_files(src: Path, exclude: list[str], rel_base: str) -> list[tuple[Path, str]]:
    """Yield (absolute source, staged-relative path) for one manifest entry.

    Symlinks are rejected here, at the source. Checking for them in the stage
    was pointless: shutil.copyfile dereferences a link and copies the target's
    bytes, so by then there is no link left to find and the target's contents
    are already inside the package. This is the only point where the question
    "does this path leave the tree we meant to publish?" can still be asked.
    """
    def check_link(p: Path, rel: str) -> None:
        if p.is_symlink():
            raise BuildError(
                f"{rel} is a symlink to {os.readlink(p)}. Copying it would "
                f"silently pull that target's contents into the package; "
                f"resolve it deliberately or exclude it."
            )

    if src.is_file():
        check_link(src, rel_base)
        return [] if is_excluded(rel_base, exclude) else [(src, rel_base)]
    out = []
    for p in sorted(src.rglob("*")):
        rel = f"{rel_base}/{p.relative_to(src).as_posix()}"
        if is_excluded(rel, exclude):
            continue
        check_link(p, rel)
        if not p.is_file():
            continue
        out.append((p, rel))
    return out


def stage(manifest: dict, repo: Path, aux: Path, stage_dir: Path) -> list[str]:
    """Copy every included file into a clean staging tree. Returns staged paths."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    exclude = manifest["exclude"]
    staged: list[str] = []
    skipped: list[str] = []

    for entry in manifest["include"]:
        root = aux if entry.get("root") == "aux" else repo
        src = root / entry["src"]
        if not src.exists():
            raise BuildError(
                f"manifest entry {entry['src']!r} (root={entry.get('root', 'repo')}) "
                f"does not exist at {src}. The package is built by inclusion, so a "
                f"missing entry means a paper artifact would silently not ship."
            )
        rel_base = entry.get("dst", entry["src"])

        if entry.get("transform") == "gzip":
            expect = entry.get("expect_sha256_prefix")
            if expect:
                actual = sha256_of(src)[:len(expect)]
                if actual != expect:
                    raise BuildError(
                        f"{entry['src']} hashes to {actual}, manifest expects "
                        f"{expect}. The table was regenerated; confirm the new "
                        f"one is the published version before updating the "
                        f"manifest."
                    )
            dst = stage_dir / rel_base
            dst.parent.mkdir(parents=True, exist_ok=True)
            # mtime=0: the gzip header otherwise embeds the build time and two
            # builds of the same tree would differ.
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                with gzip.GzipFile(fileobj=fout, mode="wb",
                                   compresslevel=9, mtime=0) as gz:
                    shutil.copyfileobj(fin, gz)
            staged.append(rel_base)
            continue

        found = iter_source_files(src, exclude, rel_base)
        if not found and not entry.get("allow_empty"):
            # A directory that exists but yields nothing is the failure mode
            # the inclusion design is supposed to rule out: the aux checkout
            # has the folder, its contents moved, and the package ships without
            # a cited artifact while every gate passes. Only gate_freshness
            # covers one such entry; the rest had nothing behind them.
            raise BuildError(
                f"manifest entry {entry['src']!r} (root={entry.get('root', 'repo')}) "
                f"exists at {src} but contributes no files after exclusions. A "
                f"cited artifact would silently not ship. Fix the source, or set "
                f'"allow_empty": true on the entry if emptiness is legitimate.'
            )
        if not found:
            skipped.append(entry["src"])
        for abs_src, rel in found:
            dst = stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(abs_src, dst)
            staged.append(rel)

    readme_src = repo / manifest["zip_root_readme"]
    if not readme_src.is_file():
        raise BuildError(f"zip_root_readme missing: {readme_src}")
    shutil.copyfile(readme_src, stage_dir / "README.md")
    staged.append("README.md")

    if skipped:
        print(f"  {YELLOW}note{RESET}  entries declared allow_empty that "
              f"contributed nothing: {', '.join(skipped)}")

    # Later entries may overwrite earlier ones; report the distinct set.
    return sorted(set(staged))


# --------------------------------------------------------------------------
# scrub
# --------------------------------------------------------------------------

def scrub_manifest_json(path: Path) -> int:
    """Rewrite provenance/MANIFEST.json's identifying values in place.

    Walks the parsed JSON rather than the text. A regex over 47 KB of hashes
    is one bad character class away from corrupting the record this package
    exists to make checkable.

    Returns the number of values changed.
    """
    data = load_json(path)
    changed = 0

    for run in data.get("reported_runs", []) + data.get("decomposition_runs", []):
        loc = run.get("dataset_location")
        if isinstance(loc, str) and loc.startswith("/"):
            # Keep the path from the repo directory down; that is the part
            # that identifies the file, and drop the machine-specific prefix.
            marker = "/humanebench/"
            idx = loc.find(marker)
            run["dataset_location"] = (
                loc[idx + len(marker):] if idx >= 0 else Path(loc).name
            )
            changed += 1

    if "zenodo_doi" in data:
        # Renamed to match the scrubbed build_provenance.py, so regenerating
        # the manifest from inside the package produces the same key.
        data.pop("zenodo_doi")
        data["archive_doi"] = ""
        changed += 1

    note = (data.get("summary") or {}).get("decomposition_note")
    if isinstance(note, str) and "Zenodo" in note:
        data["summary"]["decomposition_note"] = note.replace(
            "The Zenodo deposit", "The archived deposit"
        )
        changed += 1

    # Assert the outcome, not the arithmetic. A fixed hit count here is wrong
    # by construction: the count is 45 + one per decomposition run + 2, so it
    # moves every time provenance is rebuilt -- and rebuilding provenance is
    # exactly what the freshness gate requires of the final build. Pinning the
    # number made the two requirements mutually exclusive and would have failed
    # the deadline build outright. What actually matters is that nothing
    # identifying survives, so that is what gets checked.
    residue = []

    def walk(node, where="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                if "zenodo" in k.lower():
                    residue.append(f"{where}.{k} (key)")
                walk(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")
        elif isinstance(node, str):
            low = node.lower()
            if node.startswith("/") and "/users/" in low:
                residue.append(f"{where} = {node[:60]}")
            elif "zenodo" in low or "10.5281/" in low:
                residue.append(f"{where} = {node[:60]}")

    walk(data)
    if residue:
        raise BuildError(
            "provenance/MANIFEST.json still carries identifying values after "
            "scrubbing:\n  " + "\n  ".join(residue[:10])
        )

    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return changed


def scrub_golden_rater_names(path: Path) -> int:
    """Pseudonymise metadata.human_rater_names in the golden-questions JSONL.

    Eleven of the 24 rows name their human raters by first name; two of those
    names are authors and a third is on no redaction list, so a keyword scan
    would not have caught it. Rows that already use the file's own `rater-N`
    convention are left alone, and the rewritten names get a disjoint
    `rater-a/b/c` series: they are a different set of raters, and folding them
    into the numbered series would assert an identity nobody has checked.

    Returns the number of rows changed.
    """
    already_pseudonymous = re.compile(r"^rater[-_ ]?\w+$", re.IGNORECASE)
    alias: dict[str, str] = {}
    letters = "abcdefghijklmnopqrstuvwxyz"
    changed = 0
    out_lines = []

    for line in path.read_text().splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        row = json.loads(line)
        md = row.get("metadata") or {}
        inner = md.get("metadata") if isinstance(md.get("metadata"), dict) else md
        names = inner.get("human_rater_names")
        if isinstance(names, str):
            tokens = [t.strip() for t in names.split(",")]
            if any(not already_pseudonymous.match(t) for t in tokens):
                new = []
                for t in tokens:
                    if already_pseudonymous.match(t):
                        new.append(t)
                        continue
                    key = t.lower()
                    if key not in alias:
                        alias[key] = f"rater-{letters[len(alias)]}"
                    new.append(alias[key])
                inner["human_rater_names"] = ", ".join(new)
                changed += 1
        out_lines.append(json.dumps(row, ensure_ascii=False))

    path.write_text("\n".join(out_lines) + "\n")
    return changed


def scrub(rules: dict, stage_dir: Path) -> list[str]:
    """Apply every rule to the staged tree. Raises on any hit-count mismatch."""
    log = []
    for rule in rules["rules"]:
        target = stage_dir / rule["path"]
        if not target.is_file():
            raise BuildError(
                f"scrub rule targets {rule['path']}, which is not in the stage. "
                f"Either the manifest stopped including it (then drop the rule) "
                f"or the path changed (then fix the rule)."
            )
        op = rule["op"]
        expect = rule.get("expect_hits")

        if op == "replace":
            text = target.read_text()
            hits = text.count(rule["find"])
            if hits != expect:
                raise BuildError(
                    f"{rule['path']}: expected {expect} occurrence(s) of the "
                    f"scrub pattern, found {hits}. The file changed under the "
                    f"rule; re-read it and update scrub_rules.json rather than "
                    f"relaxing the count.\n  pattern: {rule['find'][:120]!r}"
                )
            target.write_text(text.replace(rule["find"], rule["replace"]))

        elif op == "write":
            target.write_text(rule["content"])
            hits = 1

        elif op == "manifest_json":
            # No expect_hits: the count scales with the number of runs in the
            # manifest, so pinning it guarantees a false failure the next time
            # provenance is rebuilt. scrub_manifest_json asserts the outcome.
            hits = scrub_manifest_json(target)
            if expect is not None:
                raise BuildError(
                    f"{rule['path']}: this rule must not declare expect_hits. "
                    f"Its hit count is a function of how many runs the manifest "
                    f"holds and changes whenever provenance is regenerated; it "
                    f"verifies its own outcome instead."
                )

        elif op == "golden_rater_names":
            hits = scrub_golden_rater_names(target)
            if hits != expect:
                raise BuildError(
                    f"{rule['path']}: pseudonymised {hits} rows, expected "
                    f"{expect}. Rows were added or already fixed upstream; "
                    f"confirm which before changing this number."
                )

        else:
            raise BuildError(f"unknown scrub op: {op!r}")

        log.append(f"{rule['path']}: {op} x{hits}")
    return log


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def load_scan_terms(redaction_list: Path) -> list[str]:
    terms = []
    if redaction_list.is_file():
        for line in redaction_list.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                terms.append(line)
    else:
        raise BuildError(
            f"redaction list not found at {redaction_list}; the keyword gate "
            f"would run with only the built-in terms, which is weaker than it "
            f"looks. Point --redaction-list at the real file."
        )
    seen = {t.lower() for t in terms}
    for extra in EXTRA_SCAN_TERMS:
        if extra.lower() not in seen:
            terms.append(extra)
    return terms


def scannable_bytes(p: Path) -> bytes:
    """The bytes a content gate should actually inspect.

    A gzipped entry read raw is DEFLATE noise: no term can match it, so the
    keyword gate reported PASS on the single largest data file in the package
    without ever seeing a byte of it. Compressed entries are inflated here so
    they are scanned on their real contents. Everything else is returned as-is.
    """
    if p.suffix == ".gz":
        try:
            with gzip.open(p, "rb") as fh:
                return fh.read()
        except OSError as exc:
            raise BuildError(f"{p.name} is not readable as gzip: {exc}") from None
    return p.read_bytes()


def gate_keywords(stage_dir: Path, terms: list[str]) -> list[str]:
    """Every staged byte, binaries and compressed entries included."""
    failures = []
    lowered = [(t, t.lower().encode()) for t in terms]
    for p in sorted(stage_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(stage_dir).as_posix()
        if any(fnmatch.fnmatch(rel, w) for w in SCAN_WHITELIST):
            continue
        blob = scannable_bytes(p).lower()
        hits = sorted({t for t, needle in lowered if needle in blob})
        if hits:
            failures.append(f"{rel}: {', '.join(hits)}")
    return failures


def _png_text_chunks(raw: bytes) -> list[bytes]:
    out, i = [], 8
    while i + 8 <= len(raw):
        try:
            length = struct.unpack(">I", raw[i:i + 4])[0]
        except struct.error:
            break
        kind = raw[i + 4:i + 8]
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            payload = raw[i + 8:i + 8 + length]
            out.append(payload)
            if kind == b"zTXt":
                # zTXt is keyword \0 method <deflate>; try to read the text.
                try:
                    out.append(zlib.decompress(payload.split(b"\x00", 2)[-1]))
                except Exception:
                    pass
        if kind == b"IEND":
            break
        i += 12 + length
    return out


def _pdf_blobs(raw: bytes) -> list[bytes]:
    """Raw bytes plus every FlateDecode stream we can inflate."""
    blobs = [raw]
    for m in re.finditer(rb"stream\r?\n", raw):
        start = m.end()
        end = raw.find(b"endstream", start)
        if end < 0:
            continue
        try:
            blobs.append(zlib.decompress(raw[start:end]))
        except Exception:
            pass
    return blobs


def gate_figure_metadata(stage_dir: Path, terms: list[str]) -> list[str]:
    """PDF Info dicts and PNG text chunks carry OS-level author tags."""
    failures = []
    lowered = [(t, t.lower().encode()) for t in terms]
    for p in sorted(stage_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".pdf", ".png"):
            continue
        raw = p.read_bytes()
        blobs = _pdf_blobs(raw) if p.suffix.lower() == ".pdf" else _png_text_chunks(raw)
        hits = set()
        for blob in blobs:
            low = blob.lower()
            hits |= {t for t, needle in lowered if needle in low}
        if hits:
            failures.append(
                f"{p.relative_to(stage_dir).as_posix()}: {', '.join(sorted(hits))} "
                f"(regenerate the figure with metadata={{}}; do not whitelist)"
            )
    return failures


# Dotfiles that may appear in the package, named one at a time. Anything else
# beginning with a dot fails the build. The exclude glob already drops them;
# this is the independent check, because the file that motivated it
# (data_generation/.env, live API credentials) is invisible in the worktree
# where development builds run and present in the checkout where the final
# build runs -- so no amount of dev-build evidence could have caught it.
ALLOWED_DOTFILES: set[str] = set()


def gate_structure(stage_dir: Path) -> list[str]:
    failures = []
    for p in sorted(stage_dir.rglob("*")):
        rel = p.relative_to(stage_dir).as_posix()
        if p.is_symlink():
            resolved = p.resolve()
            if not str(resolved).startswith(str(stage_dir.resolve())):
                failures.append(f"{rel}: symlink escaping the stage -> {resolved}")
            continue
        if not p.is_file():
            continue
        name = p.name
        dotted = [seg for seg in rel.split("/") if seg.startswith(".")]
        if dotted and rel not in ALLOWED_DOTFILES:
            failures.append(
                f"{rel}: hidden file or directory ({', '.join(dotted)}). Dotfiles "
                f"are excluded as a class because credentials hide among them; "
                f"add it to ALLOWED_DOTFILES only if it is genuinely publishable"
            )
        if "__pycache__" in rel.split("/"):
            failures.append(f"{rel}: build junk")
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(f"{rel}: {size / 1e6:.1f} MB exceeds the "
                            f"{MAX_FILE_BYTES / 1e6:.0f} MB per-file cap")
    return failures


# Gates 4 and 5 run Python with the stage as its working directory, which
# writes __pycache__ into it and lets any script drop an output there. Both
# would then be zipped, and the __pycache__ check in gate 3 has already run by
# then, so it would not catch them -- an earlier build shipped three .pyc files
# this way. These gates therefore operate on a throwaway copy: whatever they
# leave behind is deleted with it, and the artifact cannot be touched by the
# act of testing it. PYTHONDONTWRITEBYTECODE is belt to that braces.
SUBPROC_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def _run_in(sandbox: Path, python: str, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([python, *argv], cwd=sandbox, env=SUBPROC_ENV,
                          capture_output=True, text=True)


def gate_importable(sandbox: Path, python: str) -> list[str]:
    proc = _run_in(
        sandbox, python,
        ["-c", "import humanebench.provenance, humanebench.decomposition"],
    )
    if proc.returncode != 0:
        return [f"import failed after scrubbing:\n{proc.stderr.strip()}"]
    return []


def gate_smoke(sandbox: Path, python: str) -> list[str]:
    """Run the package the way a reviewer will, and check it agrees with itself."""
    failures = []

    proc = _run_in(sandbox, python, ["scripts/verify_provenance.py"])
    if proc.returncode != 0:
        failures.append(
            "verify_provenance.py exited "
            f"{proc.returncode}\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}"
        )

    proc = _run_in(sandbox, python, ["scripts/compute_helm_power.py"])
    if proc.returncode != 0:
        failures.append(
            f"compute_helm_power.py exited {proc.returncode}\n{proc.stderr[-2000:]}"
        )

    # The agreement script from the shipped .gz, against the shipped table.
    # n_bootstrap=0 on purpose: the CIs take minutes and the point estimates
    # are the published numbers, so this checks the claim without making the
    # build unusably slow.
    smoke_out = sandbox / "_smoke_tables"
    proc = _run_in(sandbox, python, [
        "scripts/compute_inter_judge_agreement.py",
        "--raw-csv", "tables/inter_judge_raw_regenerated.csv.gz",
        "--tables-dir", str(smoke_out), "--n-bootstrap", "0",
    ])
    if proc.returncode != 0:
        failures.append(
            f"compute_inter_judge_agreement.py --raw-csv exited "
            f"{proc.returncode}\n{proc.stderr[-2000:]}"
        )
    else:
        import csv as _csv
        published = sandbox / "tables" / "inter_judge_agreement.csv"
        recomputed = smoke_out / "inter_judge_agreement.csv"
        try:
            a = next(iter(_csv.DictReader(published.open())))
            b = next(iter(_csv.DictReader(recomputed.open())))
            for field in ("alpha_ord", "alpha_bin", "sign_disagreement_rate",
                          "n_items_included"):
                if abs(float(a[field]) - float(b[field])) > 1e-9:
                    failures.append(
                        f"shipped table and shipped code disagree on {field}: "
                        f"{a[field]} vs {b[field]}"
                    )
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            failures.append(f"could not compare agreement outputs: {exc}")
    return failures


def gate_freshness(stage_dir: Path, allow_empty_decomp: bool) -> list[str]:
    failures = []
    summary = stage_dir / "tables" / "decomposition" / "decomposition_summary.md"
    if not summary.is_file():
        failures.append(
            "tables/decomposition/decomposition_summary.md is missing; the "
            "decomposition analysis has not been regenerated into --aux-root"
        )
    manifest = stage_dir / "provenance" / "MANIFEST.json"
    if manifest.is_file():
        runs = load_json(manifest).get("decomposition_runs") or []
        if not runs and not allow_empty_decomp:
            failures.append(
                "provenance/MANIFEST.json has no decomposition_runs. The "
                "manifest predates the decomposition runs; rebuild provenance "
                "before the final build (--allow-empty-decomp-manifest is for "
                "development builds only)"
            )
    return failures


# --------------------------------------------------------------------------
# zip
# --------------------------------------------------------------------------

def write_zip(stage_dir: Path, out: Path) -> None:
    """Deterministic zip: sorted entries, fixed timestamps, fixed modes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p.relative_to(stage_dir).as_posix()
        for p in stage_dir.rglob("*") if p.is_file()
    )
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            src = stage_dir / rel
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, src.read_bytes())


def write_report(report_path: Path, stage_dir: Path, zip_path: Path,
                 repo: Path, aux: Path, scrub_log: list[str]) -> None:
    """Local-path-bearing build record. Never goes inside the zip."""
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True)
    lines = [
        "HumaneBench supplementary build report",
        "",
        f"repo:        {repo}",
        f"aux root:    {aux}",
        f"git HEAD:    {head.stdout.strip() or '(unavailable)'}",
        f"zip:         {zip_path}",
        f"zip bytes:   {zip_path.stat().st_size:,}",
        f"zip sha256:  {sha256_of(zip_path)}",
        "",
        "uncommitted at build time:",
    ]
    lines += [f"  {ln}" for ln in (dirty.stdout.strip().splitlines() or ["  (clean)"])]
    lines += ["", "scrub rules applied:"]
    lines += [f"  {ln}" for ln in scrub_log]
    lines += ["", "staged contents (sha256, bytes, path):"]
    for p in sorted(stage_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(stage_dir).as_posix()
            lines.append(f"  {sha256_of(p)}  {p.stat().st_size:>10,}  {rel}")
    report_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aux-root", type=Path, default=REPO_ROOT,
                    help="Where to find artifacts gitignored in the repo but "
                         "included in the package (default: this repo).")
    ap.add_argument("--stage-dir", type=Path,
                    default=REPO_ROOT / "dist" / "supplementary_stage")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "dist" / "humanebench_aaai27_supplementary.zip")
    ap.add_argument("--redaction-list", type=Path, default=None,
                    help="Redaction term list (default: <aux-root>/"
                         "anonymization_redaction_list.txt).")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter used for the smoke gates.")
    ap.add_argument("--allow-empty-decomp-manifest", action="store_true",
                    help="Development builds only: permit a provenance manifest "
                         "with no decomposition runs. The final build must not "
                         "need this.")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="Skip gate 5 only. For iterating on the manifest; the "
                         "final build must run every gate.")
    args = ap.parse_args()

    repo = REPO_ROOT
    aux = args.aux_root.expanduser().resolve()
    stage_dir = args.stage_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    redaction_list = (args.redaction_list or aux / "anonymization_redaction_list.txt")

    manifest = load_json(CONFIG_DIR / "manifest.json")
    rules = load_json(CONFIG_DIR / "scrub_rules.json")
    terms = load_scan_terms(redaction_list.expanduser().resolve())

    print(f"repo {repo}\naux  {aux}\n")

    print("staging ...")
    staged = stage(manifest, repo, aux, stage_dir)
    total = sum(p.stat().st_size for p in stage_dir.rglob("*") if p.is_file())
    print(f"  {len(staged):,} files, {total / 1e6:.1f} MB\n")

    print("scrubbing ...")
    scrub_log = scrub(rules, stage_dir)
    for line in scrub_log:
        print(f"  {line}")
    print()

    print(f"verifying ({len(terms)} scan terms) ...")
    gates: list[tuple[str, list[str]]] = [
        ("1 keyword scan", gate_keywords(stage_dir, terms)),
        ("2 figure metadata", gate_figure_metadata(stage_dir, terms)),
        ("3 structure", gate_structure(stage_dir)),
    ]

    sandbox = stage_dir.with_name(stage_dir.name + "_run")
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(stage_dir, sandbox)
    try:
        gates.append(("4 importability", gate_importable(sandbox, args.python)))
        if args.skip_smoke:
            print(f"  {YELLOW}SKIP{RESET}  5 smoke (--skip-smoke)")
        else:
            gates.append(("5 smoke", gate_smoke(sandbox, args.python)))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    gates.append(
        ("6 freshness", gate_freshness(stage_dir, args.allow_empty_decomp_manifest))
    )
    # The executing gates ran on a copy, so this should be a no-op. It is here
    # because "nothing wrote to the stage" is the assumption the zip's
    # reproducibility rests on, and assumptions that are never checked are the
    # ones that stop being true.
    gates.append(("7 stage unchanged by testing", gate_structure(stage_dir)))

    failed = False
    for name, failures in gates:
        if failures:
            failed = True
            print(f"  {RED}FAIL{RESET}  {name}")
            for f in failures:
                print(f"          {f}")
        else:
            print(f"  {GREEN}PASS{RESET}  {name}")
    if failed:
        print(f"\n{RED}Build stopped.{RESET} Fix the cause; do not whitelist, "
              f"loosen a limit, or delete a gate to get a zip out.")
        return 1

    print("\nzipping ...")
    write_zip(stage_dir, out)
    size = out.stat().st_size
    if size > ZIP_HARD_LIMIT:
        print(f"  {RED}FAIL{RESET}  {size / 1e6:.1f} MB exceeds the 100 MB limit")
        return 1
    if size > ZIP_WARN_LIMIT:
        print(f"  {YELLOW}warn{RESET}  {size / 1e6:.1f} MB is over half the "
              f"100 MB limit")

    report = out.with_name("build_report.txt")
    write_report(report, stage_dir, out, repo, aux, scrub_log)

    print(f"  {out}  ({size / 1e6:.1f} MB)")
    print(f"  sha256 {sha256_of(out)}")
    print(f"  report {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"\n{RED}BUILD ERROR{RESET} {exc}", file=sys.stderr)
        raise SystemExit(2) from None
