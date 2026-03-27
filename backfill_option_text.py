"""
backfill_option_text.py

Rewrites choices.ndjson so every entry's option_text comes from the
ns_results.csv column of the same name, matched on (issue_num, option_num).

Entries that have no CSV match (e.g. dismiss option_id=0, or issues not in
the CSV) are left exactly as-is.

Usage:
    python backfill_option_text.py                    # uses default paths
    python backfill_option_text.py --dry-run          # prints diffs, no write
"""

import argparse
import csv
import json
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(SCRIPT_DIR, "ns_results.csv")
NDJSON_PATH = os.path.join(SCRIPT_DIR, "choices.ndjson")


def build_lookup(csv_path: str) -> dict:
    """Return {(issue_num, option_num): option_text} from the CSV."""
    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["issue_num"]), int(row["option_num"]))
                lookup[key] = row["option_text"].strip()
            except (KeyError, ValueError):
                continue
    return lookup


def backfill(ndjson_path: str, lookup: dict, dry_run: bool = False) -> None:
    changed = unchanged = 0

    new_lines = []
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                new_lines.append(line)
                continue

            entry = json.loads(line)

            try:
                issue_num  = int(entry["issue_num"])
                option_num = int(entry["option_id"])
            except (KeyError, ValueError, TypeError):
                new_lines.append(line)
                unchanged += 1
                continue

            csv_text = lookup.get((issue_num, option_num))

            if csv_text is None:
                # Not in CSV — dismiss option or unknown issue; leave alone.
                new_lines.append(line)
                unchanged += 1
                continue

            old_text = entry.get("option_text", "")
            if old_text == csv_text:
                new_lines.append(line)
                unchanged += 1
                continue

            if dry_run:
                print(f"Line {lineno} | issue {issue_num} opt {option_num}")
                print(f"  OLD: {old_text[:120]}")
                print(f"  NEW: {csv_text[:120]}")
                print()

            entry["option_text"] = csv_text
            new_lines.append(json.dumps(entry, ensure_ascii=False))
            changed += 1

    if not dry_run:
        with open(ndjson_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")

    print(f"{'[DRY RUN] ' if dry_run else ''}Done — {changed} updated, {unchanged} unchanged.")


def main():
    parser = argparse.ArgumentParser(description="Backfill choices.ndjson option_text from CSV.")
    parser.add_argument("--csv",     default=CSV_PATH,    help="Path to ns_results.csv")
    parser.add_argument("--ndjson",  default=NDJSON_PATH, help="Path to choices.ndjson")
    parser.add_argument("--dry-run", action="store_true",  help="Print diffs without writing")
    args = parser.parse_args()

    lookup = build_lookup(args.csv)
    print(f"Loaded {len(lookup):,} CSV rows.")
    backfill(args.ndjson, lookup, dry_run=args.dry_run)


if __name__ == "__main__":
    main()