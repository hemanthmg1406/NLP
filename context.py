"""Capture the textual context window around each enumerated equation.

For each enumerated equation, returns a tight prose window with the equation
marked [TARGET] and other display equations marked [EQ]. Inline math is
rewritten as clean LaTeX. The window is trimmed to WINDOW_SENTENCES on each
side to avoid dumping full derivations into every equation's context.
"""

import re
from copy import deepcopy
from pathlib import Path

from lxml import html as lxml_html

CACHE_DIR = Path("cache")

# Paragraphs to pull on each side before sentence-trimming.
NEIGHBOUR_PARAS = 1

# Sentences to keep on each side of [TARGET].
WINDOW_SENTENCES = 2

_ABBREV = ["e.g", "i.e", "cf", "fig", "figs", "eq", "eqs", "ref", "refs", "etc", "vs",
           "al", "resp", "no", "sec", "app", "approx", "viz"]


def get_contexts(arxiv_id):
    """Return the context window for every enumerated equation in one cached paper.

    Maps eq_id (e.g. "S2.E1") to its trimmed target-marked context string.
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
            continue
        para = _enclosing_para(table)
        if para is None:
            continue
        target = set(eq_ids)
        blocks = _prev_paras(para) + [para] + _next_paras(para)
        text = " ".join(_clean_para(b, target) for b in blocks)
        text = re.sub(r"\s+", " ", text).strip()
        window = _trim_to_window(text, WINDOW_SENTENCES)
        for eq_id in eq_ids:
            contexts[eq_id] = window

    return contexts


def _table_eq_ids(table):
    """Return the eq_id of every numbered equation row in a table.

    Single-number blocks yield one id; align blocks yield one id per numbered
    row. Mirrors review_equations.extract_equations so every extracted equation
    has a matching context entry.
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
    """Walk up to the nearest ltx_para div containing node."""
    cur = node.getparent()
    while cur is not None:
        if "ltx_para" in (cur.get("class") or ""):
            return cur
        cur = cur.getparent()
    return None


def _prev_paras(para):
    """Collect up to NEIGHBOUR_PARAS preceding sibling paragraphs in document order."""
    out = []
    sib = para.getprevious()
    while sib is not None and len(out) < NEIGHBOUR_PARAS:
        if "ltx_para" in (sib.get("class") or ""):
            out.append(sib)
        sib = sib.getprevious()
    return list(reversed(out))


def _next_paras(para):
    """Collect up to NEIGHBOUR_PARAS following sibling paragraphs in document order."""
    out = []
    sib = para.getnext()
    while sib is not None and len(out) < NEIGHBOUR_PARAS:
        if "ltx_para" in (sib.get("class") or ""):
            out.append(sib)
        sib = sib.getnext()
    return out


def _clean_para(para, target_ids):
    """Return paragraph prose with display equations marked and inline math cleaned.

    Equations in target_ids become [TARGET]; all other display equations become
    [EQ]. Inline math is replaced with its cleaned LaTeX in $...$. Unresolved
    citations and undefined macro errors are replaced with whitespace so they
    do not leak raw keys into later symbol/meaning rules.
    """
    p = deepcopy(para)

    for bad in p.xpath('.//span[contains(@class, "ltx_missing_citation") '
                       'or contains(@class, "ltx_missing_label") '
                       'or (contains(@class, "ltx_ERROR") and contains(@class, "undefined"))]'):
        _replace_with_text(bad, " ")

    for tab in p.xpath('.//table[contains(@class, "ltx_equation")]'):
        is_target = bool(set(_table_eq_ids(tab)) & target_ids)
        _replace_with_text(tab, " [TARGET] " if is_target else " [EQ] ")

    for math in p.xpath('.//math'):
        tex = math.xpath('.//annotation[@encoding="application/x-tex"]/text()')
        token = f" ${_clean_tex(tex[0])}$ " if tex else " "
        _replace_with_text(math, token)

    return p.text_content()


def _clean_tex(tex):
    """Strip LaTeX % line-comments and collapse whitespace."""
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
    """Keep k sentences on each side of the sentence containing [TARGET].

    Returns the full text unchanged when the marker is absent.
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

    Lightweight deterministic splitter with no model dependency. Protects dots
    in "0.5" and "e.g." style patterns before splitting on sentence-ending
    punctuation, then restores them.
    """
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    for ab in _ABBREV:
        t = re.sub(rf"\b({ab})\.", lambda m: m.group(1) + "<DOT>", t, flags=re.I)
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
