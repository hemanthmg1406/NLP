"""Capture the textual context around each enumerated equation.

Meaning, symbol definitions, and relation cues live in the prose surrounding an equation,
not in the equation itself. For each enumerated equation this module returns a tight,
equation-centred context window:

  - the equation's own position is marked [TARGET]; any other display equations in range
    are marked [EQ], so later steps know which equation the text belongs to,
  - the window is trimmed to a few sentences either side of [TARGET], so a crowded
    multi-equation paragraph no longer dumps its whole derivation into every equation,
  - inline math is rewritten as clean LaTeX (line breaks and stray % comments removed)
    so declarations like "where $\\phi$ is the wave function" stay readable and matchable.

No network calls. Operates only on files already in cache/.
"""

import re
from copy import deepcopy
from pathlib import Path

from lxml import html as lxml_html

CACHE_DIR = Path("cache")

# Paragraphs of prose to pull on each side before sentence-trimming. One each side gives
# enough material when the equation sits at the very start or end of its own paragraph.
NEIGHBOUR_PARAS = 1

# Sentences to keep on each side of the [TARGET] equation. This is the main precision
# knob: smaller is tighter (less noise), larger captures definitions that sit further out.
WINDOW_SENTENCES = 2

# Abbreviations whose trailing dot must not be treated as a sentence end.
_ABBREV = ["e.g", "i.e", "cf", "fig", "figs", "eq", "eqs", "ref", "refs", "etc", "vs",
           "al", "resp", "no", "sec", "app", "approx", "viz"]


def get_contexts(arxiv_id):
    """Return the context window for every enumerated equation in one cached paper.

    Parameters
    ----------
    arxiv_id : str
        Bare id such as "2403.05230".

    Returns
    -------
    dict
        Maps eq_id (e.g. "S2.E1") to its trimmed, target-marked context string.
        Equations whose paragraph cannot be located are omitted.
    """
    path = CACHE_DIR / f"{arxiv_id}.html"
    if not path.exists():
        return {}

    tree = lxml_html.parse(str(path))
    contexts = {}

    for table in tree.xpath('//table[contains(@class, "ltx_equation")]'):
        eq_ids = _table_eq_ids(table)
        if not eq_ids:
            continue  # no numbered rows, not an enumerated equation
        para = _enclosing_para(table)
        if para is None:
            continue
        # Build one window for this block, with the block marked [TARGET], and assign it
        # to every numbered row in the block. An align block holds several numbered
        # equations that share the same surrounding prose, so each must get a context;
        # keying only the first row was what left later rows with no context.
        target = set(eq_ids)
        blocks = _prev_paras(para) + [para] + _next_paras(para)
        text = " ".join(_clean_para(b, target) for b in blocks)
        text = re.sub(r"\s+", " ", text).strip()
        window = _trim_to_window(text, WINDOW_SENTENCES)
        for eq_id in eq_ids:
            contexts[eq_id] = window

    return contexts


def _table_eq_ids(table):
    """Return the eq_id of every numbered equation in a table, matching the extractor.

    A single-number block yields one id; an align block with several numbered rows
    yields one id per numbered row (read from each row's number span). This mirrors
    review_equations.extract_equations so every extracted equation has a context.
    """
    eqnos = table.xpath('.//span[contains(@class, "ltx_tag_equation")]')
    if not eqnos:
        return []
    if len(eqnos) == 1:
        got = eqnos[0].xpath('ancestor::*[@id][1]/@id')
        return [got[0]] if got else []
    ids = []
    for row in table.xpath('.//tr[contains(@class, "ltx_eqn_row")]'):
        rno = row.xpath('.//span[contains(@class, "ltx_tag_equation")]')
        if rno:
            got = rno[0].xpath('ancestor::*[@id][1]/@id')
            if got:
                ids.append(got[0])
    return ids


def _enclosing_para(node):
    """Walk up to the nearest LaTeXML paragraph (div.ltx_para) containing the node."""
    cur = node.getparent()
    while cur is not None:
        if "ltx_para" in (cur.get("class") or ""):
            return cur
        cur = cur.getparent()
    return None


def _prev_paras(para):
    """Collect up to NEIGHBOUR_PARAS preceding sibling paragraphs, in document order."""
    out = []
    sib = para.getprevious()
    while sib is not None and len(out) < NEIGHBOUR_PARAS:
        if "ltx_para" in (sib.get("class") or ""):
            out.append(sib)
        sib = sib.getprevious()
    return list(reversed(out))


def _next_paras(para):
    """Collect up to NEIGHBOUR_PARAS following sibling paragraphs, in document order."""
    out = []
    sib = para.getnext()
    while sib is not None and len(out) < NEIGHBOUR_PARAS:
        if "ltx_para" in (sib.get("class") or ""):
            out.append(sib)
        sib = sib.getnext()
    return out


def _clean_para(para, target_ids):
    """Return a paragraph's prose with equations marked and inline math cleaned.

    The equation block being described (any of whose row ids is in target_ids) becomes
    [TARGET]; every other display equation becomes [EQ]. Inline math becomes its cleaned
    LaTeX wrapped in $...$. Works on a deep copy, so the parsed tree is never mutated.

    Parameters
    ----------
    para : lxml element
        A ltx_para block.
    target_ids : set of str
        The eq_ids of the equation block this context is being built for.
    """
    p = deepcopy(para)

    # Drop unresolved-reference junk so raw keys do not leak into the prose and mislead
    # the later symbol/meaning rules: undefined macros (\added, \deleted from the changes
    # package, shown as ltx_ERROR undefined) and unresolved \cite / \ref that LaTeXML left
    # as raw keys (ltx_missing_citation / ltx_missing_label). Resolved citations like "[23]"
    # are kept. This touches only the context text, never the equation LaTeX.
    for bad in p.xpath('.//span[contains(@class, "ltx_missing_citation") '
                       'or contains(@class, "ltx_missing_label") '
                       'or (contains(@class, "ltx_ERROR") and contains(@class, "undefined"))]'):
        _replace_with_text(bad, " ")

    for tab in p.xpath('.//table[contains(@class, "ltx_equation")]'):
        is_target = bool(set(_table_eq_ids(tab)) & target_ids)
        marker = " [TARGET] " if is_target else " [EQ] "
        _replace_with_text(tab, marker)

    for math in p.xpath('.//math'):
        tex = math.xpath('.//annotation[@encoding="application/x-tex"]/text()')
        token = f" ${_clean_tex(tex[0])}$ " if tex else " "
        _replace_with_text(math, token)

    return p.text_content()


def _clean_tex(tex):
    """Normalise an inline LaTeX string: drop % line-comments and collapse whitespace.

    LaTeXML preserves the source line wrapping, so a token can contain a '%' that comments
    out the rest of its line followed by a newline. We delete from '%' to the end of that
    line (joining the pieces as LaTeX would) and flatten remaining whitespace.
    """
    tex = re.sub(r"%[^\n]*\n\s*", "", tex)
    return re.sub(r"\s+", " ", tex).strip()


def _replace_with_text(node, text):
    """Remove an element from its parent, leaving a plain text token in its place."""
    parent = node.getparent()
    if parent is None:
        return
    prev = node.getprevious()
    if prev is not None:
        prev.tail = (prev.tail or "") + text + (node.tail or "")
    else:
        parent.text = (parent.text or "") + text + (node.tail or "")
    parent.remove(node)


def _trim_to_window(text, k):
    """Keep only k sentences either side of the sentence containing [TARGET].

    If the marker is not found (rare), the full text is returned rather than guessing.
    """
    sents = _split_sentences(text)
    target_idx = next((i for i, s in enumerate(sents) if "[TARGET]" in s), None)
    if target_idx is None:
        return text
    lo = max(0, target_idx - k)
    hi = min(len(sents), target_idx + k + 1)
    return " ".join(sents[lo:hi]).strip()


def _split_sentences(text):
    """Split text into sentences, protecting decimals and common abbreviations.

    A lightweight deterministic splitter (no model): protect "0.5" and "e.g." style dots,
    split on sentence-ending punctuation followed by whitespace, then restore the dots.
    """
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    for ab in _ABBREV:
        t = re.sub(rf"\b({ab})\.", lambda m: m.group(1) + "<DOT>", t, flags=re.I)
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


if __name__ == "__main__":
    # Quick visual check on the first cached paper.
    ctx = get_contexts("2403.05230")
    for eq_id in list(ctx)[:3]:
        print(f"\n=== {eq_id} ===")
        print(ctx[eq_id])
