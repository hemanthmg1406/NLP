"""Build dataset_29.json in paper_list_29.txt order.

For each paper, use cached HTML when present; otherwise fetch from arXiv.
PDF-only or missing-HTML papers are skipped. HTML papers with no extracted
equations are still written as empty objects. Stops after at least 350
equations, finishing the current paper completely.

Usage:
    python run.py
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import fetcher
from pipeline import process_paper

PAPER_LIST   = Path("paper_list_29.txt")
CACHE_DIR    = Path("cache")
OUTPUT       = Path("dataset_29.json")
SKIPPED_LOG  = Path("skipped_papers.txt")
MIN_EQ       = 350
BACKUP_DIR   = Path("run_backups")


def load_paper_ids():
    """Read arxiv IDs from paper_list_29.txt in listed order.

    Returns
    -------
    list of str
        Bare arxiv IDs with the 'arXiv:' prefix stripped.
    """
    ids = []
    for line in PAPER_LIST.read_text().splitlines():
        line = line.strip()
        if line:
            ids.append(line.replace("arXiv:", "").strip())
    return ids


def count_equations(results):
    """Return total extracted equations across all processed papers."""
    return sum(len(eqs) for eqs in results.values())


def backup_existing_outputs():
    """Keep old outputs before starting a fresh ordered build."""
    existing = [path for path in (OUTPUT, SKIPPED_LOG) if path.exists()]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / stamp
    backup_path.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, backup_path / path.name)
    print(f"Backed up existing outputs to {backup_path}")


def ensure_html(arxiv_id, robots_parser):
    """Ensure HTML exists for arxiv_id, fetching if needed.

    Returns
    -------
    tuple
        (has_html, skip_reason)
    """
    html_path = CACHE_DIR / f"{arxiv_id}.html"
    if fetcher._valid_cached_file(html_path, "html"):
        return True, ""

    path, kind = fetcher.fetch_one(arxiv_id, robots_parser)
    if kind == "html" and path is not None:
        return True, ""
    if kind == "pdf":
        return False, "pdf_only"
    return False, "missing_html"


def run():
    """Process papers in list order until at least MIN_EQ equations are collected."""
    paper_ids = load_paper_ids()
    backup_existing_outputs()
    results = {}
    skipped = []
    robots_parser = None

    print(f"Starting fresh ordered build from {PAPER_LIST}")

    for index, arxiv_id in enumerate(paper_ids, start=1):
        html_path = CACHE_DIR / f"{arxiv_id}.html"
        if not fetcher._valid_cached_file(html_path, "html") and robots_parser is None:
            robots_parser = fetcher._robots()

        has_html, skip_reason = ensure_html(arxiv_id, robots_parser)
        if not has_html:
            skipped.append({"arxiv_id": arxiv_id, "reason": skip_reason})
            print(f"{index}: {arxiv_id}: skipped ({skip_reason})")
            continue

        total_eq = count_equations(results)
        print(f"\n{index}: Processing {arxiv_id}  (equations so far: {total_eq})")

        paper_result = process_paper(arxiv_id)
        results[arxiv_id] = paper_result
        total_eq = count_equations(results)

        # Save after every paper so progress is not lost on crash.
        OUTPUT.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if total_eq >= MIN_EQ:
            print(f"\nReached {total_eq} equations after {arxiv_id}. Stopping.")
            break

    SKIPPED_LOG.write_text(
        "\n".join(f"{item['arxiv_id']}\t{item['reason']}" for item in skipped),
        encoding="utf-8",
    )
    print(f"\nWrote {OUTPUT}  ({len(results)} papers, {total_eq} equations)")
    print(f"Skipped {len(skipped)} papers (PDF-only or missing HTML) -> {SKIPPED_LOG}")


if __name__ == "__main__":
    run()
