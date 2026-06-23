"""Extract enumerated equations from cached LaTeXML HTML papers."""

import re
from pathlib import Path

from lxml import html as lxml_html

CACHE_DIR    = Path("cache")
DATASET_SIZE = 7


def extract_equations(arxiv_id, max_eq=DATASET_SIZE, cap=DATASET_SIZE):
    """Extract enumerated equations from one cached LaTeXML HTML paper.

    Anchors on the printed-number span (ltx_tag_equation), reads LaTeX from
    the x-tex annotation in the same equation row, and captures the nearest
    ancestor id (e.g. "S1.E1") for deep linking. Anchoring on the number
    avoids inline-math fragments that also carry x-tex annotations but are
    not enumerated. Stops after cap equations.

    Returns a list of {number, latex, eq_id, in_dataset}, at most cap items.
    """
    path = CACHE_DIR / f"{arxiv_id}.html"
    if not path.exists():
        return []

    tree = lxml_html.parse(str(path))
    rows = []
    seen = set()

    for table in tree.xpath('//table[contains(@class, "ltx_equation")]'):
        if len(rows) >= cap:
            break
        eqnos    = table.xpath('.//span[contains(@class, "ltx_tag_equation")]')
        eqn_rows = table.xpath('.//tr[contains(@class, "ltx_eqn_row")]')
        if not eqnos:
            continue

        if len(eqnos) == 1:
            # Single equation or multi-line block with one number. Use the full
            # table rather than individual rows so continuation lines are not lost.
            number = eqnos[0].text_content().strip().strip("()")
            latex  = _row_latex(table)
            eq_id  = (eqnos[0].xpath('ancestor::*[@id][1]/@id') or [""])[0]
            _append(rows, seen, number, latex, eq_id)
        else:
            # Align block: each numbered row is its own equation.
            for r in eqn_rows:
                if len(rows) >= cap:
                    break
                rno = r.xpath('.//span[contains(@class, "ltx_tag_equation")]')
                if not rno:
                    continue
                number = rno[0].text_content().strip().strip("()")
                eq_id  = (rno[0].xpath('ancestor::*[@id][1]/@id') or [""])[0]
                _append(rows, seen, number, _row_latex(r), eq_id)

    rows = rows[:cap]
    for i, r in enumerate(rows):
        r["in_dataset"] = i < max_eq
    return rows


def _row_latex(node):
    """Join all x-tex annotations under node into one LaTeX string.

    LaTeXML splits aligned lines into cells each with their own annotation,
    prefixed with \\displaystyle. Cells are concatenated and that directive
    is stripped so the stored equation reads as one clean expression.
    """
    out = []
    for a in node.xpath('.//annotation[@encoding="application/x-tex"]'):
        out.append(re.sub(r'^\\displaystyle\s*', '', a.text_content().strip()))
    return " ".join(p for p in out if p).strip()


def _clean_latex(latex):
    """Strip trailing label/legend rows from multi-line equation LaTeX.

    LaTeXML occasionally emits a trailing row containing only identifier
    tokens with no operators or '=' sign. A segment is treated as a legend
    dump when it has no '=' and contains at least three tokens after
    stripping LaTeX syntax. Only the last segment is ever stripped.
    """
    parts = [p.strip() for p in latex.split(' \\\\ ')]
    if len(parts) <= 1:
        return latex
    last = parts[-1]
    if '=' not in last:
        flat = re.sub(r'\\[a-zA-Z]+|\{|\}|\[|\]|\(|\)', ' ', last)
        if len(flat.split()) >= 3:
            parts = parts[:-1]
    return ' \\\\ '.join(parts)


def _append(rows, seen, number, latex, eq_id):
    """Add an equation to rows, skipping blanks, duplicates, and continuation rows.

    Continuation rows start with '=' (LHS dropped by LaTeXML row splitting).
    Legend dumps are stripped by _clean_latex before appending.
    """
    if not number or not latex or number in seen:
        return
    if latex.lstrip().startswith('='):
        return
    latex = _clean_latex(latex)
    if not latex:
        return
    rows.append({"number": number, "latex": latex, "eq_id": eq_id})
    seen.add(number)
