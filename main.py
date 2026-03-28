"""
NationStates Issue Bot
----------------------
Fetches open issues, scores each option using:
  • Weighted census stat deltas from ns_results.csv
  • Optional policy / notability preferences from policy_priority.cfg

Issues not found in the CSV are skipped entirely.

Setup:
  1. Add NATION, PASSWORD, USER_AGENT to a .env file (or GitHub Secrets)
  2. Edit priority.cfg with your stat priorities
  3. Optionally edit policy_priority.cfg with policy preferences
  4. Place ns_results.csv in the same directory

policy_priority.cfg format (one entry per line):
  Marriage Equality — 80      # positive  → want this policy ADDED
  Capital Punishment — -90    # negative  → want this policy REMOVED
  Lines starting with # are comments.

Policy scoring rules:
  • "adds" / "sometimes_adds"   → treated as a gain  (half-weight for "sometimes")
  • "removes" / "sometimes_removes" → treated as a loss (half-weight for "sometimes")
  • "sometimes_*" columns apply 50 % of the stated weight.
  • Notability columns are scored identically to policy columns.
"""

import os
import time
import json
import argparse
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE          = "https://www.nationstates.net/cgi-bin/api.cgi"
NATION        = os.getenv("NATION")
PASSWORD      = os.getenv("PASSWORD")
USER_AGENT    = os.getenv("USER_AGENT", "NS-Math-Bot/1.0")
SLEEP_BETWEEN = int(os.getenv("SLEEP_BETWEEN_REQUESTS", 10))
TEST_MODE     = os.getenv("TEST_MODE", "false").lower() == "true"
CSV_PATH      = os.path.join(os.path.dirname(__file__), "ns_results.csv")
PRIORITY_PATH = os.path.join(os.path.dirname(__file__), "priority.cfg")
POLICY_PRIORITY_PATH = os.path.join(os.path.dirname(__file__), "policy_priority.cfg")
LOG_PATH      = os.path.join(os.path.dirname(__file__), "choices.ndjson")
SUMMARY_PATH  = os.path.join(os.path.dirname(__file__), "choices_summary.md")

# Pipe-delimited string columns added by the new scraper.
# These must never be treated as numeric stat columns.
EFFECT_COLS = [
    "policy_adds",
    "policy_removes",
    "policy_sometimes_adds",
    "policy_sometimes_removes",
    "notability_adds",
    "notability_removes",
    "notability_sometimes_adds",
    "notability_sometimes_removes",
]

# Session PIN — obtained on first login, reused to avoid 409 errors
_pin = None


# ---------------------------------------------------------------------------
# Load and precompute scoring data
# ---------------------------------------------------------------------------

def load_scoring_data():
    df = pd.read_csv(CSV_PATH)

    # Stat columns: everything after the 4 base columns, minus effect columns
    base_cols = {"issue_num", "issue_title", "option_num", "option_text"}
    stat_cols = [
        c for c in df.columns[4:]
        if c not in base_cols and c not in EFFECT_COLS
    ]

    stat_stds = df[stat_cols].apply(pd.to_numeric, errors="coerce").std()

    col_map = {}
    for col in stat_cols:
        clean = col.replace("Industry: ", "").replace("Sector: ", "").lower()
        col_map[clean] = col

    return df, stat_stds, col_map, stat_cols


def load_priority(col_map):
    priority = {}
    if not os.path.exists(PRIORITY_PATH):
        return priority
    with open(PRIORITY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "—" in line:
                name, score_str = line.rsplit("—", 1)
            elif "-" in line.rsplit(None, 1)[-1]:
                parts = line.rsplit(None, 1)
                name, score_str = parts[0].rstrip(" -"), parts[1]
            else:
                continue
            name = name.strip()
            try:
                score = float(score_str.strip())
            except ValueError:
                continue
            col = col_map.get(name.lower())
            if col:
                priority[col] = score
    return priority


def load_policy_priority():
    """
    Load policy / notability preferences from policy_priority.cfg.
    Returns a dict { policy_name_lower: score } where:
      positive score → want this policy PRESENT (added)
      negative score → want this policy ABSENT  (removed)
    """
    policy_pref = {}
    if not os.path.exists(POLICY_PRIORITY_PATH):
        return policy_pref
    with open(POLICY_PRIORITY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "—" in line:
                name, score_str = line.rsplit("—", 1)
            elif line.count("-") >= 1:
                # Allow "Policy Name - 80" or "Policy Name — -90"
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    name, score_str = parts[0].rstrip(" -"), parts[1]
                else:
                    continue
            else:
                continue
            name = name.strip()
            try:
                score = float(score_str.strip())
            except ValueError:
                continue
            if name:
                policy_pref[name.lower()] = score
    return policy_pref


def build_weights(priority):
    return {col: abs(score) for col, score in priority.items()}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_stat_option(row, weights, priority, stat_stds):
    """Score an option from numeric census-stat deltas."""
    total = 0.0
    for col, weight in weights.items():
        delta = row.get(col, np.nan)
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            delta = np.nan
        if pd.isna(delta):
            continue
        std = stat_stds.get(col, 0)
        if std == 0:
            continue
        norm    = delta / std
        directed = norm if priority[col] >= 0 else -norm
        total   += weight * directed
    return total


def score_policy_option(row, policy_pref):
    """
    Score an option from its policy / notability effect columns.

    For each effect bucket:
      adds / removes          → full weight
      sometimes_adds / _removes → half weight

    Direction:
      "adds"    bucket + positive preference  →  +ve contribution
      "removes" bucket + positive preference  →  -ve contribution
      (and vice versa for negative preferences)
    """
    if not policy_pref:
        return 0.0

    # (column, multiplier) pairs
    buckets = [
        ("policy_adds",                  +1.0),
        ("policy_removes",               -1.0),
        ("policy_sometimes_adds",        +0.5),
        ("policy_sometimes_removes",     -0.5),
        ("notability_adds",              +1.0),
        ("notability_removes",           -1.0),
        ("notability_sometimes_adds",    +0.5),
        ("notability_sometimes_removes", -0.5),
    ]

    total = 0.0
    for col, direction in buckets:
        cell = row.get(col, "")
        if not cell or (isinstance(cell, float) and pd.isna(cell)):
            continue
        for name in str(cell).split("|"):
            name = name.strip()
            if not name:
                continue
            pref = policy_pref.get(name.lower())
            if pref is None:
                continue
            total += direction * pref

    return total


def format_effects(row):
    """Return a human-readable summary of policy/notability effects for a row."""
    parts = []
    labels = {
        "policy_adds":                  "adds policy",
        "policy_removes":               "removes policy",
        "notability_adds":              "adds notability",
        "notability_removes":           "removes notability",
        "policy_sometimes_adds":        "sometimes adds policy",
        "policy_sometimes_removes":     "sometimes removes policy",
        "notability_sometimes_adds":    "sometimes adds notability",
        "notability_sometimes_removes": "sometimes removes notability",
    }
    for col, label in labels.items():
        cell = row.get(col, "")
        if not cell or (isinstance(cell, float) and pd.isna(cell)):
            continue
        names = [n.strip() for n in str(cell).split("|") if n.strip()]
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts) if parts else ""


def score_option(row, weights, priority, stat_stds, policy_pref):
    """Combined stat + policy score for an option row."""
    return (
        score_stat_option(row, weights, priority, stat_stds)
        + score_policy_option(row, policy_pref)
    )


# ---------------------------------------------------------------------------
# NationStates API — PIN session management
# ---------------------------------------------------------------------------

def ns_request(params=None, headers=None, data=None):
    global _pin
    headers = headers or {}
    headers["User-Agent"] = USER_AGENT

    if _pin:
        headers["X-Pin"] = _pin
    else:
        headers["X-Password"] = PASSWORD

    r = requests.request(
        "POST" if data else "GET",
        BASE,
        params=params if not data else None,
        data=data,
        headers=headers,
    )

    if "X-Pin" in r.headers:
        _pin = r.headers["X-Pin"]

    if r.status_code != 200:
        print(f"[!] HTTP {r.status_code} – {r.text[:120]}")

    return r


def fetch_issues():
    r = ns_request(params={"nation": NATION, "q": "issues"})
    return BeautifulSoup(r.text, "xml")


def answer_issue(issue_id, option_id):
    r = ns_request(data={
        "nation": NATION,
        "c": "issue",
        "issue": issue_id,
        "option": option_id,
    })
    return r.text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_once(df, stat_stds, col_map, priority, weights, policy_pref):
    soup   = fetch_issues()
    issues = soup.find_all("ISSUE")
    print(f"Found {len(issues)} open issue(s).")

    for issue_xml in issues:
        issue_id = issue_xml["id"]
        title_el = issue_xml.find("TITLE")
        title    = title_el.text if title_el else ""
        options  = issue_xml.find_all("OPTION")

        print(f"\nIssue #{issue_id}: {title}")

        try:
            issue_num = int(issue_id)
        except ValueError:
            issue_num = None

        if issue_num is not None:
            match = df[df["issue_num"] == issue_num]
        else:
            match = df[df["issue_title"].str.lower() == title.lower()]

        if match.empty:
            print(f"  → Not in CSV, skipping.")
            continue

        best_option_id   = None
        best_score       = float("-inf")
        best_option_text = ""
        best_effects     = ""

        # Sort API options by numeric ID; pair positionally with sorted CSV rows.
        sorted_options = sorted(
            options,
            key=lambda o: (int(o["id"]) if o["id"].lstrip("-").isdigit() else 999),
        )
        csv_rows = match.sort_values("option_num").reset_index(drop=True)

        for pos, opt_xml in enumerate(sorted_options):
            opt_id   = opt_xml["id"]
            opt_text = opt_xml.text.strip()

            if pos >= len(csv_rows):
                print(f"  Option {opt_id}: no CSV row at position {pos}, skipping")
                continue

            row          = csv_rows.iloc[pos].to_dict()
            stat_score   = score_stat_option(row, weights, priority, stat_stds)
            policy_score = score_policy_option(row, policy_pref)
            total_score  = stat_score + policy_score
            effects      = format_effects(row)

            score_str = f"{total_score:+.1f}"
            if policy_pref and policy_score != 0.0:
                score_str += f" (stats {stat_score:+.1f} / policy {policy_score:+.1f})"

            effects_str = f"  [{effects}]" if effects else ""
            print(f"  Option {opt_id}: {score_str}  {opt_text[:60]}{effects_str}")

            if total_score > best_score:
                best_score       = total_score
                best_option_id   = opt_id
                best_option_text = str(row.get("option_text", opt_text)).strip()
                best_effects     = effects

        if best_option_id is None:
            print(f"  → No scoreable options found, skipping.")
            continue

        print(f"  ✓ Choosing option {best_option_id}: {best_option_text[:60]}")
        if best_effects:
            print(f"     Effects: {best_effects}")

        log_entry = {
            "timestamp":    time.time(),
            "issue_id":     issue_id,
            "issue_num":    issue_num,
            "title":        title,
            "option_id":    best_option_id,
            "option_text":  best_option_text,
            "score":        round(best_score, 2),
            "effects":      best_effects,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if TEST_MODE:
            print(f"  [TEST MODE] Would submit option {best_option_id} — not submitting.")
        else:
            result = answer_issue(issue_id, best_option_id)
            print(f"  → Submitted. Response: {result[:80]}")

        time.sleep(SLEEP_BETWEEN)


def write_summary():
    """Rewrite choices_summary.md from the full choices.ndjson log."""
    entries = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    entries.sort(key=lambda e: e["timestamp"], reverse=True)

    lines = [
        "# NationStates Choices Log",
        "",
        "| Date (UTC) | Issue | Option | Score | Effects |",
        "|------------|-------|--------|-------|---------|",
    ]
    if not entries:
        lines.append("| — | *No issues answered yet* | — | — | — |")
    for e in entries:
        date    = time.strftime("%Y-%m-%d %H:%M", time.gmtime(e["timestamp"]))
        issue   = f"#{e['issue_num']} {e['title']}"
        text    = e["option_text"]
        effects = e.get("effects", "")
        if len(text) > 2 and text[1] == "." and text[0].isdigit():
            text = text[2:].strip()
        
        text    = text.replace("|", "\\|")
        issue   = issue.replace("|", "\\|")
        effects = effects.replace("|", "\\|") if effects else "—"
        lines.append(f"| {date} | {issue} | {text} | {e['score']:.2f} | {effects} |")

    lines.append("")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary written to {SUMMARY_PATH} ({len(entries)} entries).")


def run_check(issue_num, df, stat_stds, col_map, priority, weights, policy_pref):
    """Score all options for a given issue number from the CSV without submitting."""
    match = df[df["issue_num"] == issue_num]
    if match.empty:
        print(f"Issue #{issue_num} not found in ns_results.csv.")
        return

    title = match.iloc[0]["issue_title"] if "issue_title" in match.columns else f"Issue #{issue_num}"
    print(f"\nIssue #{issue_num}: {title}")
    print(f"{'Option':<8} {'Total':>8}  {'Stats':>8}  {'Policy':>8}  Text")
    print("-" * 90)

    csv_rows   = match.sort_values("option_num").reset_index(drop=True)
    best_score = float("-inf")
    best_opt   = None
    scores     = []

    for _, row in csv_rows.iterrows():
        stat_score   = score_stat_option(row.to_dict(), weights, priority, stat_stds)
        policy_score = score_policy_option(row.to_dict(), policy_pref)
        total_score  = stat_score + policy_score
        opt_num      = int(row["option_num"])
        text         = str(row.get("option_text", "")).strip()
        if len(text) > 2 and text[1] == "." and text[0].isdigit():
            text = text[2:].strip()
        effects = format_effects(row.to_dict())
        scores.append((opt_num, total_score, stat_score, policy_score, text, effects))
        if total_score > best_score:
            best_score = total_score
            best_opt   = (opt_num, text)

    for opt_num, total, stats, policy, text, effects in scores:
        marker    = " ◄" if (best_opt and opt_num == best_opt[0]) else ""
        truncated = (text[:50] + "…") if len(text) > 50 else text
        print(f"  [{opt_num}]    {total:>+8.2f}  {stats:>+8.2f}  {policy:>+8.2f}  {truncated}{marker}")
        if effects:
            print(f"           {effects}")

    print("-" * 90)
    if best_opt:
        print(f"  Best option: [{best_opt[0]}]  score {best_score:+.2f}")
        print(f"  → {best_opt[1][:120]}")

    log_path = os.path.join(os.path.dirname(__file__), "manual_checks.ndjson")
    entry = {
        "timestamp":        time.time(),
        "issue_num":        issue_num,
        "title":            title,
        "best_option_num":  best_opt[0] if best_opt else None,
        "best_option_text": best_opt[1] if best_opt else None,
        "best_score":       round(best_score, 2) if best_opt else None,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nLogged to {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NationStates Issue Bot")
    parser.add_argument(
        "--check", type=int, metavar="ISSUE_NUM",
        help="Score all options for a given issue number from the CSV without submitting.",
    )
    args = parser.parse_args()

    print("Loading scoring data...")
    df, stat_stds, col_map, stat_cols = load_scoring_data()
    priority     = load_priority(col_map)
    weights      = build_weights(priority)
    policy_pref  = load_policy_priority()

    print(f"Loaded {len(priority)} stat priorities from priority.cfg.")
    if policy_pref:
        print(f"Loaded {len(policy_pref)} policy/notability preferences from policy_priority.cfg.")
    else:
        print("No policy_priority.cfg found — policy/notability effects will be shown but not scored.")

    if args.check:
        run_check(args.check, df, stat_stds, col_map, priority, weights, policy_pref)
    else:
        if TEST_MODE:
            print("TEST MODE — decisions will be logged but not submitted.\n")

        print("Checking for issues...")
        run_once(df, stat_stds, col_map, priority, weights, policy_pref)
        print("\nWriting summary...")
        write_summary()
        print("Done.")
