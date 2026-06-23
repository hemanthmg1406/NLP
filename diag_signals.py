"""Diagnostic signal dump for manual pattern analysis.

Picks N_SAMPLE random cached HTML papers, extracts raw meaning signals for the
first N_EQ enumerated equations per paper, and writes a structured Markdown
report.  No meaning assembly runs — we want the raw evidence only.

Usage
-----
    python diag_signals.py              # uses fixed seed, outputs diag_signals.md
    python diag_signals.py --seed 42    # override seed
"""

import argparse
import random
import sys
from pathlib import Path

from lxml import html as lxml_html

from review_equations import extract_equations
from meaning_extractor import (
    get_pre_text,
    get_post_text,
    extract_meaning_signals,
)


def _build_table_index(tree):
    """Map each eq_id to its ltx_equation table node for O(1) lookup."""
    index = {}
    for tab in tree.xpath('//table[contains(@class,"ltx_equation")]'):
        for span in tab.xpath('.//span[contains(@class,"ltx_tag_equation")]'):
            got = span.xpath('ancestor::*[@id][1]/@id')
            if got:
                index[got[0]] = tab
    return index

# Config
N_SAMPLE   = 20   # papers to sample
N_EQ       = 7    # equations per paper
CACHE_DIR  = Path("cache")
OUT_FILE   = Path(f"diag_signals_seed{SEED}.md")
SEED       = 1234


def pick_papers(n, seed):
    """Return n randomly sampled arXiv IDs from cached HTML files.

    Parameters
    ----------
    n : int
    seed : int

    Returns
    -------
    list of str
    """
    html_files = sorted(CACHE_DIR.glob("*.html"))
    if len(html_files) < n:
        print(f"Warning: only {len(html_files)} cached HTML files, sampling all.")
        n = len(html_files)
    random.seed(seed)
    chosen = random.sample(html_files, n)
    # Sort by arXiv ID for reproducible report order.
    chosen.sort()
    return [p.stem for p in chosen]


def dump_paper(arxiv_id, n_eq, lines):
    """Extract signals for one paper and append formatted rows to lines.

    Parameters
    ----------
    arxiv_id : str
    n_eq : int
    lines : list of str
        Accumulator for Markdown output lines.
    """
    path = CACHE_DIR / f"{arxiv_id}.html"
    try:
        tree = lxml_html.parse(str(path))
    except Exception as e:
        lines.append(f"  _parse error: {e}_\n")
        return

    equations = extract_equations(arxiv_id)
    if not equations:
        lines.append(f"  _no enumerated equations found_\n")
        return

    table_index = _build_table_index(tree)

    for eq in equations[:n_eq]:
        number  = eq["number"]
        latex   = eq["latex"]
        eq_id   = eq["eq_id"]

        table_node = table_index.get(eq_id)
        if table_node is None:
            lines.append(f"### Eq ({number})\n")
            lines.append(f"- **status**: table node not found\n\n")
            continue

        pre_text  = get_pre_text(table_node)
        post_text = get_post_text(table_node)
        signals   = extract_meaning_signals(table_node, latex, eq_id, pre_text, tree)

        lines.append(f"### Eq ({number})\n")
        lines.append(f"- **section**: {signals['contained_section'] or '—'}\n")
        lines.append(f"- **named_eq**: {signals['named_eq'] or '—'}\n")
        lines.append(f"- **inline_label**: {signals['inline_label'] or '—'}\n")
        lines.append(f"- **theorem**: {signals['theorem_env'] or '—'} / {signals['theorem_title'] or '—'}\n")
        intro = signals['intro_sentence']
        lines.append(f"- **intro_sentence**: {intro if intro else '—'}\n")
        post_first = post_text[:120] if post_text else "—"
        lines.append(f"- **post_text**: {post_first}\n")
        lines.append(f"- **latex**: `{latex}`\n")
        lines.append("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n", type=int, default=N_SAMPLE)
    args = parser.parse_args()

    papers = pick_papers(args.n, args.seed)
    print(f"Sampled {len(papers)} papers (seed={args.seed}): {papers}")

    lines = []
    lines.append("# Equation Signal Diagnostic\n\n")
    lines.append(f"Seed: {args.seed} | Papers: {len(papers)} | Equations per paper: {N_EQ}\n\n")
    lines.append("---\n\n")

    for i, arxiv_id in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {arxiv_id}", end="  ", flush=True)
        lines.append(f"## {i}. {arxiv_id}\n\n")
        dump_paper(arxiv_id, N_EQ, lines)
        lines.append("---\n\n")
        print("done")

    out = Path(f"diag_signals_seed{args.seed}.md")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
