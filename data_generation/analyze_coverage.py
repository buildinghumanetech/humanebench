"""
HumaneBench Dataset Coverage Analysis + Data Quality Audit
Run this against your humane_bench.jsonl to get the exact numbers
needed for the AIES paper methodology section.

Usage: python analyze_coverage.py path/to/humane_bench.jsonl
"""

import json
import statistics
import sys
import pandas as pd
from collections import Counter

# Predefined taxonomies from config.py
EXPECTED_DOMAINS = [
    "relationships", "mental-health", "financial-decisions",
    "career", "parenting", "health-wellness",
    "technology-use", "politics-society", "education",
    "moral-ambiguity", "crisis-situations", "everyday-decisions"
]

EXPECTED_VULN_POPS = [
    "children", "teenagers", "elderly",
    "people-with-disabilities", "non-native-speakers", "low-tech-literacy"
]


def analyze_dataset(filepath):
    # Load JSONL
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Total scenarios: {len(rows)}")
    print("=" * 70)

    # Extract metadata
    principles = []
    domains = []
    vuln_pops = []

    for row in rows:
        meta = row.get('metadata', {})
        principles.append(meta.get('principle', 'MISSING'))
        domains.append(meta.get('domain', 'MISSING'))
        vp = meta.get('vulnerable-population', '')
        vuln_pops.append(vp if vp else '(none)')

    # --- 1. Principle distribution ---
    print("\n1. PRINCIPLE DISTRIBUTION")
    print("-" * 50)
    principle_counts = Counter(principles)
    for p, count in sorted(principle_counts.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        bar = "█" * int(pct / 2)
        print(f"  {p:40s} {count:4d} ({pct:5.1f}%) {bar}")

    # --- 2. Domain distribution ---
    print("\n2. DOMAIN DISTRIBUTION")
    print("-" * 50)
    domain_counts = Counter(domains)
    for d, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        bar = "█" * int(pct / 2)
        print(f"  {d:30s} {count:4d} ({pct:5.1f}%) {bar}")

    # --- 3. Vulnerable population distribution ---
    print("\n3. VULNERABLE POPULATION DISTRIBUTION")
    print("-" * 50)
    vp_counts = Counter(vuln_pops)
    tagged_count = sum(c for vp, c in vp_counts.items() if vp != '(none)')
    print(f"  Scenarios with vulnerable population tag: {tagged_count}/{len(rows)} ({tagged_count/len(rows)*100:.1f}%)")
    print()
    for vp, count in sorted(vp_counts.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        print(f"  {vp:30s} {count:4d} ({pct:5.1f}%)")

    # --- 4. Principle × Domain cross-tabulation ---
    print("\n4. PRINCIPLE × DOMAIN COVERAGE MATRIX")
    print("-" * 50)
    df = pd.DataFrame({'principle': principles, 'domain': domains})
    cross_tab = pd.crosstab(df['principle'], df['domain'])
    print(cross_tab.to_string())

    empty_cells = []
    for p in cross_tab.index:
        for d in cross_tab.columns:
            if cross_tab.loc[p, d] == 0:
                empty_cells.append((p, d))

    if empty_cells:
        print(f"\n  ⚠️  {len(empty_cells)} empty cells (principle×domain combinations with zero scenarios):")
        for p, d in empty_cells[:20]:
            print(f"    - {p} × {d}")
        if len(empty_cells) > 20:
            print(f"    ... and {len(empty_cells) - 20} more")
    else:
        print("\n  ✅ Full coverage: every principle×domain combination has at least one scenario")

    # --- 5. Balance metrics ---
    print("\n5. BALANCE METRICS")
    print("-" * 50)
    principle_values = list(principle_counts.values())
    p_min, p_max = min(principle_values), max(principle_values)
    print(f"  Principles: min={p_min}, max={p_max}, balance ratio={p_min/p_max:.2f}")

    domain_values = list(domain_counts.values())
    d_min, d_max = min(domain_values), max(domain_values)
    print(f"  Domains: min={d_min}, max={d_max}, balance ratio={d_min/d_max:.2f}")

    lengths = [len(row.get('input', '').split()) for row in rows]
    print(f"  Scenario length (words): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}, std={statistics.stdev(lengths):.1f}, range={max(lengths)-min(lengths)}")

    # ============================================================
    # 6. DATA QUALITY AUDIT
    # ============================================================
    print("\n" + "=" * 70)
    print("6. DATA QUALITY AUDIT")
    print("=" * 70)

    issues = []

    # 6a. Unexpected domains (not in predefined taxonomy)
    unexpected_domains = {}
    for i, row in enumerate(rows):
        d = row.get('metadata', {}).get('domain', '')
        if d and d not in EXPECTED_DOMAINS:
            if d not in unexpected_domains:
                unexpected_domains[d] = []
            unexpected_domains[d].append(row.get('id', f'row-{i}'))

    if unexpected_domains:
        print(f"\n  ⚠️  UNEXPECTED DOMAINS ({sum(len(v) for v in unexpected_domains.values())} scenarios)")
        for d, ids in sorted(unexpected_domains.items()):
            print(f"    '{d}' ({len(ids)} scenarios): {', '.join(ids[:5])}")
            if len(ids) > 5:
                print(f"      ... and {len(ids) - 5} more")
        issues.append(f"Merge or re-tag {sum(len(v) for v in unexpected_domains.values())} scenarios with unexpected domains")
    else:
        print("\n  ✅ All domains match predefined taxonomy")

    # 6b. Unexpected vulnerable populations (not in predefined taxonomy)
    unexpected_vps = {}
    for i, row in enumerate(rows):
        vp = row.get('metadata', {}).get('vulnerable-population', '')
        if vp and vp not in EXPECTED_VULN_POPS:
            if vp not in unexpected_vps:
                unexpected_vps[vp] = []
            unexpected_vps[vp].append(row.get('id', f'row-{i}'))

    if unexpected_vps:
        total_unexpected = sum(len(v) for v in unexpected_vps.values())
        print(f"\n  ⚠️  UNEXPECTED VULNERABLE POPULATIONS ({total_unexpected} scenarios)")
        for vp, ids in sorted(unexpected_vps.items(), key=lambda x: -len(x[1])):
            print(f"    '{vp}' ({len(ids)} scenarios): {', '.join(ids[:5])}")
            if len(ids) > 5:
                print(f"      ... and {len(ids) - 5} more")
        issues.append(f"Decide on taxonomy for {total_unexpected} scenarios with non-predefined populations")
    else:
        print("\n  ✅ All vulnerable populations match predefined taxonomy")

    # 6c. Combined/multi-value tags (commas in vulnerable-population)
    multi_tags = []
    for i, row in enumerate(rows):
        vp = row.get('metadata', {}).get('vulnerable-population', '')
        if ',' in vp:
            multi_tags.append((row.get('id', f'row-{i}'), vp))

    if multi_tags:
        print(f"\n  ⚠️  COMBINED TAGS ({len(multi_tags)} scenarios)")
        for rid, vp in multi_tags:
            print(f"    {rid}: '{vp}'")
        issues.append(f"Split {len(multi_tags)} combined vulnerable-population tags into single values")
    else:
        print("\n  ✅ No combined/multi-value tags found")

    # 6d. Questionable vulnerable population values
    questionable_vps = ['adults', 'global-users']
    found_questionable = []
    for i, row in enumerate(rows):
        vp = row.get('metadata', {}).get('vulnerable-population', '')
        if vp in questionable_vps:
            found_questionable.append((row.get('id', f'row-{i}'), vp))

    if found_questionable:
        print(f"\n  ⚠️  QUESTIONABLE VP TAGS ({len(found_questionable)} scenarios)")
        for rid, vp in found_questionable:
            print(f"    {rid}: '{vp}' — not a vulnerable population")
        issues.append(f"Clear {len(found_questionable)} non-vulnerable population tags (e.g., 'adults', 'global-users')")
    else:
        print("\n  ✅ No questionable vulnerable population tags")

    # 6e. target / metadata.principle mismatch
    mismatches = []
    for i, row in enumerate(rows):
        target = row.get('target', '')
        principle = row.get('metadata', {}).get('principle', '')
        if target != principle:
            mismatches.append((row.get('id', f'row-{i}'), target, principle))

    if mismatches:
        print(f"\n  ⚠️  TARGET/PRINCIPLE MISMATCHES ({len(mismatches)} scenarios)")
        for rid, target, principle in mismatches[:10]:
            print(f"    {rid}: target='{target}' vs principle='{principle}'")
        issues.append(f"Fix {len(mismatches)} target/metadata.principle mismatches")
    else:
        print("\n  ✅ All target fields match metadata.principle")

    # 6f. Duplicate IDs
    id_counts = Counter(row.get('id', '') for row in rows)
    dup_ids = {k: v for k, v in id_counts.items() if v > 1}
    if dup_ids:
        print(f"\n  ⚠️  DUPLICATE IDs ({len(dup_ids)} duplicated)")
        for rid, count in sorted(dup_ids.items(), key=lambda x: -x[1])[:10]:
            print(f"    '{rid}': appears {count} times")
        issues.append(f"Fix {len(dup_ids)} duplicate IDs")
    else:
        print("\n  ✅ All IDs are unique")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"  Scenarios: {len(rows)}")
    print(f"  Principles: {len(principle_counts)} (balance ratio: {p_min/p_max:.2f})")
    print(f"  Domains: {len(domain_counts)} ({len(EXPECTED_DOMAINS)} predefined + {len(unexpected_domains)} emergent)")
    print(f"  Vulnerable population tagged: {tagged_count}/{len(rows)} ({tagged_count/len(rows)*100:.0f}%)")
    print(f"  VP taxonomy: {len(EXPECTED_VULN_POPS)} predefined + {len(unexpected_vps)} emergent")
    print(f"  Principle×domain gaps: {len(empty_cells)}/{len(cross_tab.index) * len(cross_tab.columns)}")

    if issues:
        print(f"\n  🔧 DATA CLEANUP NEEDED ({len(issues)} issues):")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("\n  ✅ No data quality issues found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_coverage.py path/to/humane_bench.jsonl")
        sys.exit(1)

    analyze_dataset(sys.argv[1])