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

from __future__ import annotations

import os
import time
import json
import argparse
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE          = "https://www.nationstates.net/cgi-bin/api.cgi"
NATION        = os.getenv("NATION")
PASSWORD      = os.getenv("PASSWORD")
USER_AGENT    = os.getenv("USER_AGENT", "NS-Math-Bot/1.0")
SLEEP_BETWEEN = int(os.getenv("SLEEP_BETWEEN_REQUESTS", 10))
TEST_MODE     = os.getenv("TEST_MODE", "false").lower() == "true"

HERE          = os.path.dirname(__file__)
CSV_PATH      = os.path.join(HERE, "ns_results.csv")
PRIORITY_PATH = os.path.join(HERE, "priority.cfg")
POLICY_PRIORITY_PATH = os.path.join(HERE, "policy_priority.cfg")
LOG_PATH      = os.path.join(HERE, "choices.ndjson")
SUMMARY_PATH  = os.path.join(HERE, "choices_summary.md")

# Pipe-delimited string columns added by the new scraper.
# These must never be treated as numeric stat columns.
_EFFECT_COLS: frozenset[str] = frozenset({
    "policy_adds",
    "policy_removes",
    "policy_sometimes_adds",
    "policy_sometimes_removes",
    "notability_adds",
    "notability_removes",
    "notability_sometimes_adds",
    "notability_sometimes_removes",
})

# Sort key used when an option id cannot be parsed as an integer.
_UNKNOWN_OPTION_SORT_KEY = float("inf")

# (effect_column, score_direction_multiplier) pairs used when scoring policy effects.
_POLICY_BUCKETS: tuple[tuple[str, float], ...] = (
    ("policy_adds",                  +1.0),
    ("policy_removes",               -1.0),
    ("policy_sometimes_adds",        +0.5),
    ("policy_sometimes_removes",     -0.5),
    ("notability_adds",              +1.0),
    ("notability_removes",           -1.0),
    ("notability_sometimes_adds",    +0.5),
    ("notability_sometimes_removes", -0.5),
)

# Session PIN — obtained on first login, reused to avoid 409 errors
_pin: str | None = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ScoredOption:
    """All scoring information for a single issue option."""
    option_id:    str
    option_num:   int
    text:         str
    stat_score:   float
    policy_score: float
    effects:      str

    @property
    def total_score(self) -> float:
        return self.stat_score + self.policy_score


@dataclass
class ScoringData:
    """Pre-computed data loaded once at startup."""
    df:        pd.DataFrame
    stat_stds: pd.Series
    col_map:   dict[str, str]
    stat_cols: list[str]
    priority:  dict[str, float]  = field(default_factory=dict)
    weights:   dict[str, float]  = field(default_factory=dict)
    policy_pref: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_env() -> None:
    """Raise clearly if required environment variables are missing."""
    missing = [v for v in ("NATION", "PASSWORD") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Required environment variable(s) not set: {', '.join(missing)}. "
            "Add them to your .env file or GitHub Secrets."
        )


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

def strip_option_prefix(text: str) -> str:
    """Remove a leading '1. ' / '2. ' style prefix from option text."""
    if len(text) > 2 and text[1] == "." and text[0].isdigit():
        return text[2:].strip()
    return text


def parse_priority_line(line: str) -> tuple[str, float] | None:
    """
    Parse a single line from a priority config file.

    Accepted formats:
      Some Stat Name — 80
      Some Stat Name — -90
      Some Stat Name - 80

    Returns (name, score) or None if the line cannot be parsed.
    Emits a warning for lines that look like entries but fail to parse.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Prefer em-dash split; fall back to splitting on the last whitespace-
    # surrounded hyphen so that hyphenated names are handled correctly.
    if "—" in line:
        parts = line.rsplit("—", 1)
    else:
        # Split on the last run of whitespace + optional minus sign.
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            parts[0] = parts[0].rstrip(" -")
        else:
            print(f"  [cfg] Unrecognised line (skipped): {line!r}")
            return None

    name = parts[0].strip()
    score_str = parts[1].strip()
    if not name:
        print(f"  [cfg] Empty name (skipped): {line!r}")
        return None
    try:
        score = float(score_str)
    except ValueError:
        print(f"  [cfg] Could not parse score {score_str!r} (skipped): {line!r}")
        return None
    return name, score


# ---------------------------------------------------------------------------
# Load and precompute scoring data
# ---------------------------------------------------------------------------

def load_scoring_data() -> ScoringData:
    df = pd.read_csv(CSV_PATH)

    # Stat columns: everything after the 4 base columns, minus effect columns.
    base_cols = {"issue_num", "issue_title", "option_num", "option_text"}
    stat_cols = [
        c for c in df.columns[4:]
        if c not in base_cols and c not in _EFFECT_COLS
    ]

    stat_stds = df[stat_cols].apply(pd.to_numeric, errors="coerce").std()

    col_map: dict[str, str] = {}
    for col in stat_cols:
        clean = col.replace("Industry: ", "").replace("Sector: ", "").lower()
        col_map[clean] = col

    data = ScoringData(df=df, stat_stds=stat_stds, col_map=col_map, stat_cols=stat_cols)
    data.priority    = load_priority(col_map)
    data.weights     = {col: abs(score) for col, score in data.priority.items()}
    data.policy_pref = load_policy_priority()
    return data


def load_priority(col_map: dict[str, str]) -> dict[str, float]:
    priority: dict[str, float] = {}
    if not os.path.exists(PRIORITY_PATH):
        return priority
    with open(PRIORITY_PATH, encoding="utf-8") as f:
        for line in f:
            parsed = parse_priority_line(line)
            if parsed is None:
                continue
            name, score = parsed
            col = col_map.get(name.lower())
            if col:
                priority[col] = score
            else:
                print(f"  [cfg] Stat not found in CSV (skipped): {name!r}")
    return priority


def load_policy_priority() -> dict[str, float]:
    """
    Load policy / notability preferences from policy_priority.cfg.

    Returns a dict { policy_name_lower: score } where:
      positive score → want this policy PRESENT (added)
      negative score → want this policy ABSENT  (removed)
    """
    policy_pref: dict[str, float] = {}
    if not os.path.exists(POLICY_PRIORITY_PATH):
        return policy_pref
    with open(POLICY_PRIORITY_PATH, encoding="utf-8") as f:
        for line in f:
            parsed = parse_priority_line(line)
            if parsed is None:
                continue
            name, score = parsed
            policy_pref[name.lower()] = score
    return policy_pref


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_stat_option(
    row: dict,
    weights: dict[str, float],
    priority: dict[str, float],
    stat_stds: pd.Series,
) -> float:
    """Score an option from numeric census-stat deltas."""
    total = 0.0
    for col, weight in weights.items():
        delta = row.get(col)
        if delta is None or pd.isna(delta):
            continue
        std = stat_stds.get(col, 0)
        if std == 0:
            continue
        norm     = float(delta) / std
        directed = norm if priority[col] >= 0 else -norm
        total   += weight * directed
    return total


def score_policy_option(
    row: dict,
    policy_pref: dict[str, float],
) -> float:
    """
    Score an option from its policy / notability effect columns.

    For each effect bucket:
      adds / removes            → full weight
      sometimes_adds / _removes → half weight

    Direction:
      "adds"    + positive preference  →  +ve contribution
      "removes" + positive preference  →  -ve contribution
      (and vice versa for negative preferences)
    """
    if not policy_pref:
        return 0.0

    total = 0.0
    for col, direction in _POLICY_BUCKETS:
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


def format_effects(row: dict) -> str:
    """Return a human-readable summary of policy/notability effects for a row."""
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
    parts = []
    for col, label in labels.items():
        cell = row.get(col, "")
        if not cell or (isinstance(cell, float) and pd.isna(cell)):
            continue
        names = [n.strip() for n in str(cell).split("|") if n.strip()]
        if names:
            parts.append(f"{label}: {', '.join(names)}")
    return "; ".join(parts) if parts else ""


def score_all_options(
    issue_num: int,
    api_options: list | None,
    sd: ScoringData,
) -> list[ScoredOption] | None:
    """
    Score every option for the given issue number.

    ``api_options`` is the list of <OPTION> XML elements from the NS API,
    or None when calling from --check mode (CSV-only).

    Returns a sorted list of ScoredOption (highest score first), or None if
    the issue is not in the CSV.

    Options are matched by numeric ID rather than position, so a missing or
    re-ordered API option cannot corrupt the scoring of others.
    """
    match = sd.df[sd.df["issue_num"] == issue_num]
    if match.empty:
        return None

    csv_rows = match.sort_values("option_num").reset_index(drop=True)

    # Build a map from option_num → API option id (only needed in live mode).
    api_id_by_num: dict[int, str] = {}
    if api_options is not None:
        for opt_xml in api_options:
            raw_id = opt_xml["id"]
            try:
                api_id_by_num[int(raw_id)] = raw_id
            except ValueError:
                pass  # Non-integer IDs are ignored; they won't match CSV rows.

    scored: list[ScoredOption] = []
    for _, row in csv_rows.iterrows():
        row_dict   = row.to_dict()
        opt_num    = int(row["option_num"])
        text       = strip_option_prefix(str(row_dict.get("option_text", "")).strip())
        stat_score = score_stat_option(row_dict, sd.weights, sd.priority, sd.stat_stds)
        pol_score  = score_policy_option(row_dict, sd.policy_pref)
        effects    = format_effects(row_dict)

        # In live mode, look up the API option id by number.
        # Fall back to the CSV option_num cast to a string when not found.
        opt_id = api_id_by_num.get(opt_num, str(opt_num))

        scored.append(ScoredOption(
            option_id=opt_id,
            option_num=opt_num,
            text=text,
            stat_score=stat_score,
            policy_score=pol_score,
            effects=effects,
        ))

    scored.sort(key=lambda o: o.total_score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# NationStates API — PIN session management
# ---------------------------------------------------------------------------

def ns_request(
    params: dict | None = None,
    headers: dict | None = None,
    data:   dict | None = None,
    *,
    retries: int = 3,
    backoff: float = 5.0,
) -> requests.Response:
    """
    Make a GET or POST request to the NS API with PIN session management.

    Retries up to ``retries`` times on non-200 responses, with exponential
    backoff starting at ``backoff`` seconds.
    """
    global _pin
    headers = dict(headers or {})
    headers["User-Agent"] = USER_AGENT

    if _pin:
        headers["X-Pin"] = _pin
    else:
        headers["X-Password"] = PASSWORD

    last_response: requests.Response | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.request(
                "POST" if data else "GET",
                BASE,
                params=params if not data else None,
                data=data,
                headers=headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"  [!] Network error on attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(backoff * attempt)
            continue

        if "X-Pin" in r.headers:
            _pin = r.headers["X-Pin"]

        if r.status_code == 200:
            return r

        print(f"  [!] HTTP {r.status_code} on attempt {attempt}/{retries} – {r.text[:120]}")
        last_response = r
        if attempt < retries:
            time.sleep(backoff * attempt)

    # All retries exhausted — return the last response so callers can inspect it.
    if last_response is not None:
        return last_response
    raise RuntimeError("All NS API retries exhausted with no response received.")


def fetch_issues() -> BeautifulSoup:
    r = ns_request(params={"nation": NATION, "q": "issues"})
    return BeautifulSoup(r.text, "xml")


def answer_issue(issue_id: str, option_id: str) -> str:
    r = ns_request(data={
        "nation":  NATION,
        "c":       "issue",
        "issue":   issue_id,
        "option":  option_id,
    })
    return r.text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_once(sd: ScoringData) -> None:
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
            print(f"  → Non-integer issue id {issue_id!r}, skipping.")
            continue

        scored = score_all_options(issue_num, options, sd)
        if scored is None:
            print("  → Not in CSV, skipping.")
            continue

        if not scored:
            print("  → No scoreable options found, skipping.")
            continue

        for opt in sorted(scored, key=lambda o: o.option_num):
            score_str = f"{opt.total_score:+.1f}"
            if sd.policy_pref and opt.policy_score != 0.0:
                score_str += (
                    f" (stats {opt.stat_score:+.1f} / policy {opt.policy_score:+.1f})"
                )
            effects_str = f"  [{opt.effects}]" if opt.effects else ""
            print(f"  Option {opt.option_id}: {score_str}  {opt.text[:60]}{effects_str}")

        best = scored[0]
        print(f"  ✓ Choosing option {best.option_id}: {best.text[:60]}")
        if best.effects:
            print(f"     Effects: {best.effects}")

        log_entry = {
            "timestamp":   time.time(),
            "issue_id":    issue_id,
            "issue_num":   issue_num,
            "title":       title,
            "option_id":   best.option_id,
            "option_text": best.text,
            "score":       round(best.total_score, 2),
            "effects":     best.effects,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        append_summary_row(log_entry)

        if TEST_MODE:
            print(f"  [TEST MODE] Would submit option {best.option_id} — not submitting.")
        else:
            result = answer_issue(issue_id, best.option_id)
            print(f"  → Submitted. Response: {result[:80]}")

        time.sleep(SLEEP_BETWEEN)


def run_check(issue_num: int, sd: ScoringData) -> None:
    """Score all options for a given issue number from the CSV without submitting."""
    scored = score_all_options(issue_num, None, sd)
    if scored is None:
        print(f"Issue #{issue_num} not found in ns_results.csv.")
        return

    title = sd.df[sd.df["issue_num"] == issue_num].iloc[0].get("issue_title", f"Issue #{issue_num}")
    print(f"\nIssue #{issue_num}: {title}")
    print(f"{'Option':<8} {'Total':>8}  {'Stats':>8}  {'Policy':>8}  Text")
    print("-" * 90)

    best = scored[0]
    for opt in sorted(scored, key=lambda o: o.option_num):
        marker    = " ◄" if opt.option_num == best.option_num else ""
        truncated = (opt.text[:50] + "…") if len(opt.text) > 50 else opt.text
        print(
            f"  [{opt.option_num}]    {opt.total_score:>+8.2f}  "
            f"{opt.stat_score:>+8.2f}  {opt.policy_score:>+8.2f}  {truncated}{marker}"
        )
        if opt.effects:
            print(f"           {opt.effects}")

    print("-" * 90)
    print(f"  Best option: [{best.option_num}]  score {best.total_score:+.2f}")
    print(f"  → {best.text[:120]}")

    log_path = os.path.join(HERE, "manual_checks.ndjson")
    entry = {
        "timestamp":        time.time(),
        "issue_num":        issue_num,
        "title":            str(title),
        "best_option_num":  best.option_num,
        "best_option_text": best.text,
        "best_score":       round(best.total_score, 2),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nLogged to {log_path}")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def append_summary_row(entry: dict) -> None:
    """
    Append a single new row to choices_summary.md.

    Writes the header block if the file does not yet exist; otherwise appends
    one table row, avoiding the need to rewrite the entire file on every run.
    """
    write_header = not os.path.exists(SUMMARY_PATH)

    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        if write_header:
            f.write("# NationStates Choices Log\n\n")
            f.write("| Date (UTC) | Issue | Option | Score | Effects |\n")
            f.write("|------------|-------|--------|-------|--------|\n")

        date    = time.strftime("%Y-%m-%d %H:%M", time.gmtime(entry["timestamp"]))
        issue   = f"#{entry['issue_num']} {entry['title']}".replace("|", "\\|")
        text    = strip_option_prefix(entry["option_text"]).replace("|", "\\|")
        effects = (entry.get("effects") or "—").replace("|", "\\|")
        f.write(f"| {date} | {issue} | {text} | {entry['score']:.2f} | {effects} |\n")


def rebuild_summary() -> None:
    """
    Rebuild choices_summary.md entirely from choices.ndjson (newest-first).

    Use this when you want a fully re-sorted file, e.g. after manual edits to
    the log or when recovering from a partial write.
    """
    entries: list[dict] = []
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
        issue   = f"#{e['issue_num']} {e['title']}".replace("|", "\\|")
        text    = strip_option_prefix(e["option_text"]).replace("|", "\\|")
        effects = (e.get("effects") or "—").replace("|", "\\|")
        lines.append(f"| {date} | {issue} | {text} | {e['score']:.2f} | {effects} |")

    lines.append("")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary rebuilt at {SUMMARY_PATH} ({len(entries)} entries).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NationStates Issue Bot")
    parser.add_argument(
        "--check", type=int, metavar="ISSUE_NUM",
        help="Score all options for a given issue number from the CSV without submitting.",
    )
    parser.add_argument(
        "--rebuild-summary", action="store_true",
        help="Rebuild choices_summary.md from the full choices.ndjson log and exit.",
    )
    args = parser.parse_args()

    validate_env()

    print("Loading scoring data...")
    sd = load_scoring_data()

    print(f"Loaded {len(sd.priority)} stat priorities from priority.cfg.")
    if sd.policy_pref:
        print(f"Loaded {len(sd.policy_pref)} policy/notability preferences from policy_priority.cfg.")
    else:
        print("No policy_priority.cfg found — policy/notability effects will be shown but not scored.")

    if args.rebuild_summary:
        rebuild_summary()
    elif args.check:
        run_check(args.check, sd)
    else:
        if TEST_MODE:
            print("TEST MODE — decisions will be logged but not submitted.\n")
        print("Checking for issues...")
        run_once(sd)
        print("Done.")
