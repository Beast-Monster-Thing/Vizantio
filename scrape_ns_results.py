"""
NationStates Issue Results Scraper -- async concurrent version
Output: ns_results.csv  (one row per option, each stat as its own column,
        plus pipe-delimited policy/notability effect columns)

Usage:
  python scrape_ns_results.py                          # all issues
  python scrape_ns_results.py 0 49                     # issues 0-49
  python scrape_ns_results.py 0 49 20                  # issues 0-49, 20 concurrent
  python scrape_ns_results.py --output path/to/out.csv # custom output path
  python scrape_ns_results.py 0 49 20 --output out.csv # combined

Requirements: Python 3.7+  |  pip install aiohttp
"""

import argparse
import asyncio
import csv
import logging
import re
import sys
import time
from collections import OrderedDict
from html import unescape

try:
    import aiohttp
except ImportError:
    print("Missing dependency -- run:  pip install aiohttp")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)

# NOTE: This site has no valid HTTPS certificate. Fetching over plain HTTP is
# an accepted, documented risk: the scraped data could theoretically be
# tampered with in transit. If the site ever gains a valid cert, switch to
# HTTPS. Until then this comment records the conscious decision.
BASE_URL    = "http://www.mwq.dds.nl/ns/results/"
DEFAULT_OUT = "ns_results.csv"
CONCURRENCY = 15
HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Matches stat delta lines: "min to max Stat Name (mean delta)"
STAT_RE = re.compile(
    r'([+-]?[\d.]+)\s+to\s+([+-]?[\d.]+)\s+'
    r'([\w][\w\s:\'&@()-]*?)\s*\(mean\s+([+-]?[\d.]+)\)',
    re.IGNORECASE
)

# Matches policy/notability effect lines.
# Captures: (1) full relationship phrase, (2) kind ("policy"/"notability"), (3) name
# The anchor text has already been stripped of HTML tags before this runs,
# so we just match on plain text.
EFFECT_RE = re.compile(
    r'(sometimes\s+(?:adds|removes)|adds|removes)\s+(policy|notability):\s*(.+?)(?=\s*(?:sometimes\s+(?:adds|removes)|adds|removes)\s+(?:policy|notability):|$)',
    re.IGNORECASE | re.DOTALL
)

# Column keys for the eight effect buckets
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


def _effect_key(relationship: str, kind: str) -> str:
    """Map a parsed relationship + kind to a column name."""
    rel = re.sub(r'\s+', '_', relationship.strip().lower())  # e.g. "sometimes_adds"
    return f"{kind.lower()}_{rel}"                            # e.g. "policy_sometimes_adds"


def strip_tags(s):
    return re.sub(r"<[^>]+>", " ", s)


def get_all_issue_nums(html):
    nums = re.findall(r'href="(?:[^"]+/)?(\d+)\.html"', html)
    return sorted(set(int(n) for n in nums))


def get_issue_title(html, num):
    m = re.search(r'<h2[^>]*>(.*?)</h2>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return unescape(strip_tags(m.group(1))).strip()
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return unescape(strip_tags(m.group(1))).strip()
    return f"Issue #{num}"


def parse_effects(text: str) -> dict:
    """
    Extract policy and notability effects from the plain-text cell content.
    Returns a dict with keys from EFFECT_COLS; values are pipe-joined name strings
    (empty string when nothing matched for that bucket).
    """
    buckets: dict[str, list] = {k: [] for k in EFFECT_COLS}

    for m in EFFECT_RE.finditer(text):
        relationship = m.group(1).strip()   # e.g. "sometimes adds"
        kind         = m.group(2).strip()   # "policy" or "notability"
        name         = re.sub(r'\s+', ' ', m.group(3)).strip()
        if not name:
            continue
        key = _effect_key(relationship, kind)
        if key in buckets:
            buckets[key].append(name)

    return {k: "|".join(v) for k, v in buckets.items()}


def parse_issue(html, num, title):
    options = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
        if len(tds) < 2:
            continue
        opt_text = re.sub(r"\s+", " ", unescape(strip_tags(tds[0]))).strip()
        if not opt_text or opt_text.lower() == "option":
            continue

        row = {
            "issue_num":   num,
            "issue_title": title,
            "option_num":  len(options) + 1,
            "option_text": opt_text,
        }

        # Decode the result cell once so both parsers work on the same text
        cell_text = unescape(strip_tags(tds[1]))

        # Stat deltas
        for mn, mx, name, mean in STAT_RE.findall(cell_text):
            name = re.sub(r"\s+", " ", name).strip()
            if name:
                row[name] = float(mean)

        # Policy / notability effects
        row.update(parse_effects(cell_text))

        options.append(row)
    return options


async def fetch_issue(session, sem, num):
    url = f"{BASE_URL}{num}.html"
    async with sem:
        try:
            async with session.get(
                url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                html  = await resp.text()
                title = get_issue_title(html, num)
                rows  = parse_issue(html, num, title)
                return num, title, rows
        except Exception as e:
            logging.warning("#{%d} failed - %s", num, e)
            return num, f"Issue #{num}", []


async def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("positional", nargs="*")
    parser.add_argument("--output", default=DEFAULT_OUT)
    args = parser.parse_args()

    pos         = args.positional
    output_path = args.output
    start_n = end_n = None
    concurrency = CONCURRENCY

    if len(pos) >= 2:
        start_n, end_n = int(pos[0]), int(pos[1])
    if len(pos) >= 3:
        concurrency = int(pos[2])

    if start_n is not None:
        issue_nums = list(range(start_n, end_n + 1))
        logging.info(
            "Range mode: issues %d-%d  |  concurrency=%d  |  output=%s",
            start_n, end_n, concurrency, output_path,
        )
    else:
        logging.info("Fetching index...")
        async with aiohttp.ClientSession() as s:
            async with s.get(
                BASE_URL, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                logging.info("Index fetch: HTTP %s  url=%s", r.status, r.url)
                index_html = await r.text()
                logging.info("Response length: %d chars", len(index_html))
                logging.info("First 500 chars of response:\n%s", index_html[:500])
                issue_nums = get_all_issue_nums(index_html)

        if len(issue_nums) == 0:
            logging.error(
                "Index returned 0 issues. The site may have redirected to an "
                "error/challenge page instead of the real index. "
                "URL fetched: %s. Aborting -- the existing CSV will not be overwritten.",
                BASE_URL,
            )
            sys.exit(1)

        logging.info(
            "Found %d issues  |  concurrency=%d  |  output=%s",
            len(issue_nums), concurrency, output_path,
        )

    sem     = asyncio.Semaphore(concurrency)
    t0      = time.perf_counter()
    done    = 0
    results = {}

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_issue(session, sem, n) for n in issue_nums]
        for coro in asyncio.as_completed(tasks):
            num, title, rows = await coro
            results[num] = (title, rows)
            done += 1
            logging.info(
                "[%d/%d] #%d '%s' -> %d options",
                done, len(issue_nums), num, title, len(rows),
            )

    elapsed = time.perf_counter() - t0
    logging.info(
        "Fetched %d issues in %.1fs  (%.1f issues/sec)",
        len(issue_nums), elapsed, len(issue_nums) / elapsed,
    )

    # Collect all rows and discover all stat columns dynamically.
    # Effect columns are fixed and always present (may be empty strings).
    all_rows  = []
    stat_cols = OrderedDict()
    for num in sorted(results):
        _, rows = results[num]
        for row in rows:
            all_rows.append(row)
            for k in row:
                if k not in ("issue_num", "issue_title", "option_num", "option_text") \
                        and k not in EFFECT_COLS:
                    stat_cols[k] = True

    base_cols  = ["issue_num", "issue_title", "option_num", "option_text"]
    fieldnames = base_cols + list(stat_cols) + EFFECT_COLS

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            for col in stat_cols:
                row.setdefault(col, "")
            for col in EFFECT_COLS:
                row.setdefault(col, "")
            writer.writerow(row)

    logging.info(
        "Wrote %d rows x %d columns -> %s", len(all_rows), len(fieldnames), output_path
    )


if __name__ == "__main__":
    asyncio.run(main())
