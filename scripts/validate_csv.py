"""
Validates a newly scraped ns_results_new.csv against the committed ns_results.csv.

Extracted from the inline Python heredoc in weekly-scrape.yml so the logic
can be linted, unit-tested, and run locally without copying it out of YAML.

Usage:
    python scripts/validate_csv.py <old_csv> <new_csv>
"""

import argparse
import csv
import os
import sys


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return rows, cols


def main():
    parser = argparse.ArgumentParser(
        description="Validate a new NS results CSV against the existing one."
    )
    parser.add_argument("old_csv", help="Path to the existing (committed) CSV")
    parser.add_argument("new_csv", help="Path to the newly scraped CSV")
    args = parser.parse_args()

    if not os.path.exists(args.old_csv):
        print(f"No existing {args.old_csv} found -- accepting new file unconditionally.")
        sys.exit(0)

    old_rows, old_cols = load(args.old_csv)
    new_rows, new_cols = load(args.new_csv)

    errors = []

    # Row count: new must be within 10% of old
    old_n, new_n = len(old_rows), len(new_rows)
    ratio = new_n / old_n if old_n else 1
    print(f"Row count   -- old: {old_n:,}  new: {new_n:,}  ratio: {ratio:.3f}")
    if ratio < 0.90:
        errors.append(
            f"Row count dropped by more than 10% ({old_n} -> {new_n}, ratio={ratio:.3f})"
        )

    # Column count: new must have at least as many columns as old
    old_c, new_c = len(old_cols), len(new_cols)
    print(f"Column count -- old: {old_c}  new: {new_c}")
    if new_c < old_c:
        missing = set(old_cols) - set(new_cols)
        print(f"  Missing columns ({len(missing)}): {sorted(missing)[:10]}")
        if len(missing) > 5:
            errors.append(
                f"New CSV is missing {len(missing)} columns that existed before."
            )

    # Issue count: unique issue numbers in new must be >= 90% of old
    old_issues = {r["issue_num"] for r in old_rows if r.get("issue_num")}
    new_issues = {r["issue_num"] for r in new_rows if r.get("issue_num")}
    old_i, new_i = len(old_issues), len(new_issues)
    issue_ratio = new_i / old_i if old_i else 1
    print(f"Issue count  -- old: {old_i}  new: {new_i}  ratio: {issue_ratio:.3f}")
    if issue_ratio < 0.90:
        errors.append(
            f"Unique issue count dropped by more than 10% "
            f"({old_i} -> {new_i}, ratio={issue_ratio:.3f})"
        )

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"   * {e}")
        sys.exit(1)
    else:
        print("\nValidation passed -- new CSV looks healthy.")


if __name__ == "__main__":
    main()
