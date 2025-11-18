#!/usr/bin/env python3
"""
Reorder rater CSV files to match template ordering while preserving completed ratings.
"""

import csv
from pathlib import Path
from collections import defaultdict

def normalize_rating_column(fieldnames):
    """
    Normalize rating column name to 'rating'.
    Handles both 'rating (rubric)' and 'rating'.
    """
    normalized = []
    for field in fieldnames:
        if field in ['rating (rubric)', 'rating']:
            normalized.append('rating')
        else:
            normalized.append(field)
    return normalized

def read_template(template_path):
    """
    Read template file and return list of rows with correct ordering.
    Returns (rows, fieldnames).
    """
    rows = []
    with open(template_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames
        normalized_fieldnames = normalize_rating_column(original_fieldnames)

        for row in reader:
            # Normalize keys
            normalized_row = {}
            for orig_key, norm_key in zip(original_fieldnames, normalized_fieldnames):
                normalized_row[norm_key] = row[orig_key]
            rows.append(normalized_row)

    return rows, normalized_fieldnames

def read_rater_file(rater_path):
    """
    Read rater file and extract completed ratings.
    Returns dict: {(input_id, ai_model, ai_persona): {rating, reasoning, misc_comments}}
    """
    completed_ratings = {}

    with open(rater_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames
        normalized_fieldnames = normalize_rating_column(original_fieldnames)

        for row in reader:
            # Normalize keys
            normalized_row = {}
            for orig_key, norm_key in zip(original_fieldnames, normalized_fieldnames):
                normalized_row[norm_key] = row[orig_key]

            # Create unique key for this row
            key = (
                normalized_row['input_id'],
                normalized_row['ai_model'],
                normalized_row['ai_persona']
            )

            # Extract rating data (only if at least one field is non-empty)
            rating = normalized_row.get('rating', '').strip()
            reasoning = normalized_row.get('reasoning', '').strip()
            misc_comments = normalized_row.get('misc_comments', '').strip()

            if rating or reasoning or misc_comments:
                completed_ratings[key] = {
                    'rating': rating,
                    'reasoning': reasoning,
                    'misc_comments': misc_comments
                }

    return completed_ratings

def merge_template_with_ratings(template_rows, completed_ratings):
    """
    Merge template rows (correct order) with completed ratings from rater.
    Returns list of merged rows.
    """
    merged_rows = []
    matched_count = 0

    for template_row in template_rows:
        # Create key for this template row
        key = (
            template_row['input_id'],
            template_row['ai_model'],
            template_row['ai_persona']
        )

        # Start with template row
        merged_row = template_row.copy()

        # If rater has completed this row, fill in the rating data
        if key in completed_ratings:
            rating_data = completed_ratings[key]
            merged_row['rating'] = rating_data['rating']
            merged_row['reasoning'] = rating_data['reasoning']
            merged_row['misc_comments'] = rating_data['misc_comments']
            matched_count += 1
        else:
            # Ensure rating fields are empty if not completed
            merged_row['rating'] = ''
            merged_row['reasoning'] = ''
            merged_row['misc_comments'] = ''

        merged_rows.append(merged_row)

    return merged_rows, matched_count

def write_csv(rows, output_path, fieldnames):
    """
    Write rows to CSV file.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

def collect_all_rater_ratings(rater, base_dir):
    """
    Collect ALL completed ratings from all three sets for a single rater.
    Returns dict: {(input_id, ai_model, ai_persona): {rating, reasoning, misc_comments}}
    """
    all_ratings = {}

    for set_num in [1, 2, 3]:
        rater_filename = f"{rater} - Golden Question Human Ratings - set_{set_num}.csv"
        rater_path = base_dir / rater_filename

        if rater_path.exists():
            ratings = read_rater_file(rater_path)
            all_ratings.update(ratings)

    return all_ratings

def process_set(set_number, raters, rater_all_ratings, base_dir):
    """
    Process one set (set_1, set_2, or set_3) for all raters.
    Uses pre-collected ratings from all sets for each rater.
    """
    print(f"\n{'='*60}")
    print(f"Processing Set {set_number}")
    print(f"{'='*60}")

    # Read template
    template_filename = f"__TEMPLATE__ - Golden Question Human Ratings - set_{set_number}.csv"
    template_path = base_dir / template_filename

    print(f"\nReading template: {template_filename}")
    template_rows, fieldnames = read_template(template_path)
    print(f"  Template has {len(template_rows)} rows")

    # Process each rater
    results = []

    for rater in raters:
        print(f"\nProcessing rater: {rater}")

        # Use all ratings collected from this rater across all sets
        completed_ratings = rater_all_ratings[rater]
        print(f"    Using {len(completed_ratings)} total completed ratings from rater")

        # Merge template with ratings
        merged_rows, matched_count = merge_template_with_ratings(template_rows, completed_ratings)
        print(f"    Matched {matched_count} ratings to this template")

        # Write output file
        output_filename = f"{rater} - Golden Question Human Ratings - set_{set_number}_reordered.csv"
        output_path = base_dir / output_filename

        write_csv(merged_rows, output_path, fieldnames)
        print(f"  ✓ Wrote: {output_filename}")

        results.append({
            'rater': rater,
            'set': set_number,
            'matched': matched_count
        })

    return results

def main():
    base_dir = Path('human_rating_sets')
    raters = ['Andalib', 'Erika', 'Sarah']
    sets = [1, 2, 3]

    print("="*60)
    print("Reordering Human Rating CSV Files")
    print("="*60)
    print(f"\nBase directory: {base_dir}")
    print(f"Raters: {', '.join(raters)}")
    print(f"Sets: {', '.join(map(str, sets))}")

    # Step 1: Collect ALL ratings from ALL sets for each rater
    print("\n" + "="*60)
    print("Step 1: Collecting all completed ratings from each rater")
    print("="*60)

    rater_all_ratings = {}
    for rater in raters:
        print(f"\nCollecting all ratings from {rater}...")
        all_ratings = collect_all_rater_ratings(rater, base_dir)
        rater_all_ratings[rater] = all_ratings
        print(f"  Total completed ratings: {len(all_ratings)}")

    # Step 2: Process each set, redistributing ratings to match templates
    print("\n" + "="*60)
    print("Step 2: Redistributing ratings to match template sets")
    print("="*60)

    all_results = []

    for set_num in sets:
        results = process_set(set_num, raters, rater_all_ratings, base_dir)
        all_results.extend(results)

    # Print summary
    print("\n" + "="*60)
    print("Summary of Reordering")
    print("="*60)
    print("\n{:<10} {:<8} {:<10}".format("Rater", "Set", "Matched"))
    print("-"*60)

    rater_totals = {rater: 0 for rater in raters}

    for result in all_results:
        print("{:<10} {:<8} {:<10}".format(
            result['rater'],
            f"set_{result['set']}",
            result['matched']
        ))
        rater_totals[result['rater']] += result['matched']

    print("-"*60)

    # Show rater totals
    print("\nTotal ratings preserved per rater:")
    for rater in raters:
        original_count = len(rater_all_ratings[rater])
        matched_count = rater_totals[rater]
        print(f"  {rater}: {matched_count}/{original_count} ratings matched")

    print("\n" + "="*60)
    print("✓ Successfully created 9 reordered CSV files")
    print("="*60)
    print("\nOriginal files preserved.")
    print("Review the '_reordered.csv' files before replacing originals.")

if __name__ == '__main__':
    main()
