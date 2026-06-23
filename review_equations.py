"""Build a visual review of enumerated equations extracted from cached arXiv HTML.

Reads cached LaTeXML HTML, reproducibly samples papers from paper_list_29.txt, and
writes two review artifacts for their first seven enumerated equations:

  review.html : each paper linked, each equation rendered with MathJax next to its
                raw LaTeX, plus a deep link that jumps to the exact equation on arXiv.
  review.csv  : a tracking sheet (one row per equation) for marking ok / issue.

Run ``python review_equations.py`` to review the configured random sample, or pass
``--paper ARXIV_ID`` to choose another paper without modifying the supplied list.
Add ``--equation 1`` to isolate equation 1. It performs no network calls.
"""

import argparse
import csv
import html
import random
import re
from pathlib import Path

from lxml import html as lxml_html

CACHE_DIR = Path("cache")
ARXIV_HTML = "https://arxiv.org/html"
PAPER_LIST = "paper_list_29.txt"
DATASET_SIZE = 7  # equations per paper that go into the dataset (spec)

# Change only these two values to control the reproducible random review sample.
NUMBER_OF_PAPERS = 1
RANDOM_SEED = 32


def extract_equations(arxiv_id, max_eq=DATASET_SIZE, cap=DATASET_SIZE):
    """Extract enumerated equations from one cached LaTeXML HTML paper.

    Anchors on the printed-number span (ltx_tag_equation), reads the LaTeX from the
    x-tex annotation in the same equation row, and captures the nearest ancestor id
    (e.g. "S1.E1") to build a deep link to that exact equation. Anchoring on the
    number avoids the inline-math fragments that also carry x-tex annotations but are
    not enumerated.

    Extraction stops after `cap` equations. The review workflow sets both limits to
    seven so the generated artifacts contain exactly the dataset candidates being
    checked.

    Parameters
    ----------
    arxiv_id : str
        Bare id such as "2403.05230".
    max_eq : int
        How many leading equations belong in the final dataset (flagged in_dataset=True).
    cap : int
        Hard limit on equations parsed per paper (dataset size plus review buffer).

    Returns
    -------
    list of dict
        Each: {number, latex, eq_id, in_dataset}. At most `cap` items.
    """
    path = CACHE_DIR / f"{arxiv_id}.html"
    if not path.exists():
        return []

    tree = lxml_html.parse(str(path))
    rows = []
    seen = set()

    for table in tree.xpath('//table[contains(@class, "ltx_equation")]'):
        if len(rows) >= cap:
            break  # enough equations parsed, stop early
        eqnos = table.xpath('.//span[contains(@class, "ltx_tag_equation")]')
        eqn_rows = table.xpath('.//tr[contains(@class, "ltx_eqn_row")]')
        if not eqnos:
            continue  # unnumbered display equation, not enumerated

        if len(eqnos) == 1:
            # One number for the whole block: a single (possibly multi-line) equation.
            # Use _row_latex on the full table rather than iterating ltx_eqn_row tr
            # elements.  LaTeXML sometimes uses ltx_eqn_middle or plain <tr> for
            # continuation rows in long Lindblad / master equations, so per-row
            # iteration silently drops them.  _row_latex(table) collects every
            # annotation[@encoding="application/x-tex"] descendant in one pass,
            # which is complete by construction.
            number = eqnos[0].text_content().strip().strip("()")
            latex  = _row_latex(table)
            eq_id  = (eqnos[0].xpath('ancestor::*[@id][1]/@id') or [""])[0]
            _append(rows, seen, number, latex, eq_id)
        else:
            # Many numbers: an align block where each numbered row is its own equation.
            for r in eqn_rows:
                if len(rows) >= cap:
                    break
                rno = r.xpath('.//span[contains(@class, "ltx_tag_equation")]')
                if not rno:
                    continue  # an unnumbered continuation line within the align
                number = rno[0].text_content().strip().strip("()")
                eq_id = (rno[0].xpath('ancestor::*[@id][1]/@id') or [""])[0]
                _append(rows, seen, number, _row_latex(r), eq_id)

    rows = rows[:cap]
    for i, r in enumerate(rows):
        r["in_dataset"] = i < max_eq
    return rows

def inspect_equation_node(arxiv_id, eq_id):

    path = CACHE_DIR / f"{arxiv_id}.html"

    tree = lxml_html.parse(str(path))

    nodes = tree.xpath(f'//*[@id="{eq_id}"]')

    if not nodes:
        print(f"{eq_id} not found")
        return

    node = nodes[0]

    print("TAG:", node.tag)
    print("ID:", eq_id)

    parent = node.getparent()

    if parent is not None:
        print("PARENT TAG:", parent.tag)
    else:
        print("NO PARENT")

def _row_latex(node):
    """Join all x-tex annotations under a node into one LaTeX string.

    LaTeXML splits an aligned line into cells (e.g. LHS | =RHS), each with its own
    annotation, and prefixes each with \\displaystyle for rendering. We concatenate the
    cells and strip that directive so the stored equation reads as one clean expression.
    """
    out = []
    for a in node.xpath('.//annotation[@encoding="application/x-tex"]'):
        out.append(re.sub(r'^\\displaystyle\s*', '', a.text_content().strip()))
    return " ".join(p for p in out if p).strip()


def _clean_latex(latex):
    """Strip trailing label/legend rows from multi-line equation LaTeX.

    LaTeXML occasionally emits a separate <tr> inside an equation table that
    contains only variable-label tokens (no operators, no '=' sign) — a visual
    legend for the equation above it.  _row_latex joins all rows with ' \\\\ ',
    so these legend rows appear as a trailing segment.

    A segment is treated as a label dump when it has no '=' sign and contains
    at least three space-separated math tokens after stripping LaTeX commands
    and brace characters.  Only the LAST segment is ever stripped — genuine
    multi-line derivations always have '=' in every continuation line.

    Parameters
    ----------
    latex : str
        Raw joined LaTeX from _row_latex, may contain ' \\\\ ' separators.

    Returns
    -------
    str
        LaTeX with the legend row removed; original string if no legend found.
    """
    parts = [p.strip() for p in latex.split(' \\\\ ')]
    if len(parts) <= 1:
        return latex
    last = parts[-1]
    # Legend dump: no '=' and ≥ 3 tokens after stripping LaTeX syntax.
    if '=' not in last:
        flat = re.sub(r'\\[a-zA-Z]+|\{|\}|\[|\]|\(|\)', ' ', last)
        if len(flat.split()) >= 3:
            parts = parts[:-1]
    return ' \\\\ '.join(parts)


def _append(rows, seen, number, latex, eq_id):
    """Add an equation in document order, skipping blanks and duplicate numbers.

    Two malformation filters are applied before appending:

    1. Continuation rows — equations whose LaTeX begins with '=' were split off
       from a preceding multi-row derivation and lack the LHS.  They are not
       standalone enumerated equations and must be dropped.

    2. Label/legend dumps — trailing rows containing only identifier tokens are
       stripped by _clean_latex before the equation is stored.
    """
    if not number or not latex or number in seen:
        return
    # Drop continuation rows (LHS dropped, row starts with '=').
    if latex.lstrip().startswith('='):
        return
    latex = _clean_latex(latex)
    if not latex:
        return
    rows.append({"number": number, "latex": latex, "eq_id": eq_id})
    seen.add(number)


def count_eq_references(arxiv_id):
    """Count how often the prose refers to equations by number (e.g. 'Eq. 1').

    This is the signal for a degraded HTML conversion: if a paper refers to its
    equations by number yet none are extracted, the LaTeXML conversion likely dropped
    the display equations (their content is simply absent from the HTML).

    Parameters
    ----------
    arxiv_id : str
        Bare id such as "2403.05230".

    Returns
    -------
    int
        Number of "Eq./Eqs./Equation(s) <number>" mentions in the document text.
    """
    path = CACHE_DIR / f"{arxiv_id}.html"
    if not path.exists():
        return 0
    text = lxml_html.parse(str(path)).getroot().text_content()
    # Match "Eq. 1", "Eqs. (3)", "Equation 5", and also the degraded form "Eq. II",
    # where a broken reference fell back to a roman section number. The roman case is
    # itself a strong sign that the equation reference lost its target.
    return len(re.findall(r"\b(?:Eqs?|Equations?)\.?\s*~?\(?(?:\d+|[IVXLCDM]+)", text))


def classify_paper(arxiv_id, kind):
    """Decide whether a paper is usable, or the reason it must be skipped.

    Parameters
    ----------
    arxiv_id : str
        Bare id such as "2403.05230".
    kind : str
        Source kind from the fetch step: "html", "pdf", or "missing".

    Returns
    -------
    (str, int, int)
        status, n_equations, n_eq_references where status is one of:
        "ok"                : enumerated equations were extracted, use the paper,
        "no_html"           : fetch returned PDF-only or nothing,
        "degraded_html"     : HTML present but equations dropped (refs exist, 0 found),
        "no_enumerated_eqs" : HTML present, genuinely no numbered equations.
    """
    if kind != "html":
        return "no_html", 0, 0
    n_eqs = len(extract_equations(arxiv_id))
    if n_eqs > 0:
        return "ok", n_eqs, 0
    n_refs = count_eq_references(arxiv_id)
    if n_refs > 0:
        return "degraded_html", 0, n_refs
    return "no_enumerated_eqs", 0, 0


def _render_math(latex):
    """Wrap an extracted equation for MathJax preview only (does not touch the JSON).

    A bare split environment will not render on its own, so swap it for aligned, which
    does. A top-level \\ (our multi-line join) is only legal inside an alignment env, so
    wrap such bodies in gathered unless they already open with their own environment.
    """
    body = latex.replace("\\begin{split}", "\\begin{aligned}").replace("\\end{split}", "\\end{aligned}")
    if "\\\\" in body and not body.lstrip().startswith("\\begin{"):
        body = "\\begin{gathered}" + body + "\\end{gathered}"
    return "\\[" + body + "\\]"


def load_existing_status():
    """Read any issue notes already typed into review.csv so re-runs do not wipe them.

    Keyed by (arxiv_id, number) because that pair uniquely identifies an equation
    across runs even though row order or counts may change.

    Returns
    -------
    dict
        {(arxiv_id, number): status_text}
    """
    path = Path("review.csv")
    if not path.exists():
        return {}
    notes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("arxiv_id") and row.get("status"):
                aid = row["arxiv_id"].replace("arXiv:", "")  # normalise back to bare id
                notes[(aid, row["number"])] = row["status"]
    return notes


def build_review(ids, equation_number=None):
    """Write review.html and review.csv for the given list of arXiv ids.

    The HTML shows the rendered equation and an issue column carrying
    whatever was typed into review.csv's status field, so visual review and tracked
    notes live in one view. Existing notes are preserved across re-runs. When
    ``equation_number`` is supplied, only that printed equation is shown.
    """
    notes = load_existing_status()
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        # MathJax does not load the LaTeX physics package, so the authors' physics-package
        # macros (\ket, \expectationvalue, \norm, ...) render red. We define them here for
        # the PREVIEW ONLY. This never touches the extracted LaTeX, which is kept verbatim
        # as the author wrote it. Raw string + double backslashes so the JS receives single
        # backslashes in the macro bodies.
        r"<script>window.MathJax={tex:{macros:{"
        r"bm:['{\\boldsymbol{#1}}',1],"
        r"ket:['{\\left|#1\\right\\rangle}',1],"
        r"bra:['{\\left\\langle#1\\right|}',1],"
        r"braket:['{\\left\\langle#1\\middle|#2\\right\\rangle}',2],"
        r"ketbra:['{\\left|#1\\right\\rangle\\!\\left\\langle#2\\right|}',2],"
        r"outerproduct:['{\\left|#1\\right\\rangle\\!\\left\\langle#2\\right|}',2],"
        r"innerproduct:['{\\left\\langle#1\\middle|#2\\right\\rangle}',2],"
        r"expectationvalue:['{\\left\\langle#2\\right|#1\\left|#2\\right\\rangle}',2],"
        r"norm:['{\\left\\lVert#1\\right\\rVert}',1],"
        r"abs:['{\\left|#1\\right|}',1],"
        r"absolutevalue:['{\\left|#1\\right|}',1],"
        r"differential:['{\\mathrm{d}#1}',1],"
        r"crossproduct:'{\\times}',"
        r"tr:'{\\operatorname{tr}}',"
        r"Tr:'{\\operatorname{Tr}}'"
        r"}}};</script>",
        "<script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js'></script>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto}"
        "table{border-collapse:collapse;width:100%;table-layout:fixed}"
        "th,td{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "th.num,td.num{width:64px;font-weight:bold;position:sticky;left:0;background:#fff;z-index:1}"
        "th.issue,td.issue{width:180px}"
        ".eqwrap{overflow-x:auto;max-width:100%}"
        "code{font-size:85%;color:#333}.skip{opacity:.45}h2{margin-top:1.6em}</style>",
        "</head><body>",
        "<p>Reviewing the first seven enumerated equations from the first paper.</p>",
    ]
    csv_rows = []

    for arxiv_id in ids:
        eqs = extract_equations(arxiv_id)
        inspect_equation_node(arxiv_id, "S0.E1")
        print(f"\nPaper: {arxiv_id}")

        for eq in eqs:
            print("=" * 80)
            print("Equation Number:", eq["number"])
            print("Equation ID:", eq["eq_id"])
            print("LaTeX:")
            print(eq["latex"])
            print("=" * 80)
        if equation_number is not None:
            eqs = [e for e in eqs if e["number"] == equation_number]
            if not eqs:
                raise ValueError(
                    f"Paper {arxiv_id} has no extracted equation ({equation_number})"
                )
        paper_url = f"{ARXIV_HTML}/{arxiv_id}"
        in_count = sum(e["in_dataset"] for e in eqs)
        scope = f"equation ({equation_number})" if equation_number else "first seven equations"
        parts.append(f"<h2><a href='{paper_url}' target='_blank'>{arxiv_id}</a> "
                     f"&mdash; {html.escape(scope)}</h2>")
        parts.append("<table><tr><th class='num'>#</th><th>rendered</th>"
                     "<th class='issue'>issue (what went wrong)</th></tr>")

        for e in eqs:
            cls = "" if e["in_dataset"] else " class='skip'"
            rendered = _render_math(e["latex"])
            status = notes.get((arxiv_id, e["number"]), "")
            parts.append(
                f"<tr{cls}><td class='num'>({html.escape(e['number'])})</td>"
                f"<td><div class='eqwrap'>{html.escape(rendered)}</div></td>"
                f"<td class='issue'>{html.escape(status)}</td></tr>"
            )
            # CSV is the dataset tracking sheet: only the first-7 (in_dataset) rows.
            # Write the id with the "arXiv:" prefix so spreadsheets treat it as text and do
            # not drop the trailing zero (e.g. 2403.05230 shown as 2403.0523).
            if e["in_dataset"]:
                csv_rows.append([f"arXiv:{arxiv_id}", e["number"], e["eq_id"], e["latex"], status])

        parts.append("</table>")
        csv_rows.append([])  # blank row separates one paper's block from the next

    parts.append("</body></html>")
    Path("review.html").write_text("\n".join(parts), encoding="utf-8")

    with open("review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "number", "eq_id", "latex", "status"])
        w.writerows(csv_rows)

    print(f"wrote review.html and review.csv for {len(ids)} paper(s)")


def first_paper_id(paper_list=PAPER_LIST):
    """Return the first non-empty arXiv ID from the ordered paper list."""
    with open(paper_list, encoding="utf-8") as f:
        for line in f:
            arxiv_id = line.strip().replace("arXiv:", "")
            if arxiv_id:
                return arxiv_id
    raise ValueError(f"No arXiv IDs found in {paper_list}")


def select_random_papers(
    number_of_papers=NUMBER_OF_PAPERS,
    seed=RANDOM_SEED,
    paper_list=PAPER_LIST,
):
    """Select a reproducible sample of cached papers from the supplied list."""
    with open(paper_list, encoding="utf-8") as f:
        listed_ids = [
            line.strip().removeprefix("arXiv:")
            for line in f
            if line.strip()
        ]
    available_ids = [
        arxiv_id
        for arxiv_id in listed_ids
        if (CACHE_DIR / f"{arxiv_id}.html").exists()
    ]
    if number_of_papers < 1:
        raise ValueError("NUMBER_OF_PAPERS must be at least 1")
    if number_of_papers > len(available_ids):
        raise ValueError(
            f"Requested {number_of_papers} papers, but only "
            f"{len(available_ids)} listed papers are cached"
        )
    return random.Random(seed).sample(available_ids, number_of_papers)


def main():
    """Generate the one-paper review, optionally focused on one printed equation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        metavar="ARXIV_ID",
        help="paper to review, for example: 2409.02921 (overrides random sampling)",
    )
    parser.add_argument(
        "--equation",
        metavar="NUMBER",
        help="show only this printed equation number, for example: 1 or 4a",
    )
    args = parser.parse_args()
    if args.paper:
        arxiv_id = args.paper.removeprefix("arXiv:")
        cache_path = CACHE_DIR / f"{arxiv_id}.html"
        if not cache_path.exists():
            parser.error(f"cached paper not found: {cache_path}")
        paper_ids = [arxiv_id]
    else:
        try:
            paper_ids = select_random_papers()
        except ValueError as error:
            parser.error(str(error))
    build_review(paper_ids, equation_number=args.equation)


if __name__ == "__main__":
    main()
