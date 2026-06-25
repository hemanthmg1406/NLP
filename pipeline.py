"""Build the equations knowledge-graph JSON for a selection of arXiv papers.

Set LIMIT to control how many papers are processed per run and N_EQUATIONS for
how many enumerated equations to extract per paper. Output is written to
output.json and merged with any prior run so partial results are preserved.
"""

import json
import re
from pathlib import Path

from lxml import html as lxml_html

import fetcher as robot_fetch
from equations import extract_equations
from context import get_contexts, _split_sentences
from symbols import (
    extract_identifiers,
    find_symbol_definitions,
    _sentence_definitions,
    _latex_compact_identifiers,
    _latex_structured_identifiers,
)
from meaning import (
    get_pre_text,
    get_post_text,
    extract_meaning_signals,
    build_meaning,
)
from relations import build_relations

N_EQUATIONS = 7

CACHE_DIR  = Path("cache")
PAPER_LIST = "paper_list_29.txt"


def load_paper_ids(paper_list=PAPER_LIST):
    """Return bare arXiv IDs from the paper list file in document order.

    Parameters
    ----------
    paper_list : str
        Path to paper_list_29.txt.

    Returns
    -------
    list of str
        IDs with any 'arXiv:' prefix stripped.
    """
    with open(paper_list, encoding="utf-8") as f:
        return [
            line.strip().removeprefix("arXiv:")
            for line in f if line.strip()
        ]


# Decorator prefixes from symbols that produce keys like 'hat_H'.
# When both 'hat_H' and 'H' have the same definition the decorated form is
# redundant and is dropped.
_DECORATOR_PREFIXES = {
    "hat", "bar", "tilde", "vec", "widehat", "widetilde", "overline",
    "bm", "boldsymbol", "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf",
    "mathsf",
}

_EXPLICIT_DEF_RE = re.compile(
    r"\b(?:let|denotes?|represents?|stands?\s+for|refers?\s+to|defined|define|"
    r"called|is|are)\b",
    re.I,
)


def _short(text, limit=180):
    """Collapse whitespace and truncate audit evidence."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def _source_category(source, evidence):
    """Map internal definition provenance to the required audit categories."""
    if source == "post_nearby":
        return "post_where"
    if source == "respectively":
        return "pre_explicit"
    if source in {"pre_nearby", "window"} and _EXPLICIT_DEF_RE.search(evidence or ""):
        return "pre_explicit"
    if source in {"pre_nearby", "window"}:
        return "pre_nearby"
    return "not_explained"


def _definition_evidence(sym, definition, source, pre_text, post_text, window_text):
    """Find a compact sentence that supports one extracted symbol definition."""
    if not definition:
        return ""
    regions = {
        "post_nearby": post_text,
        "pre_nearby": pre_text,
        "window": window_text,
        "respectively": f"{pre_text} {post_text}",
    }
    ordered = [regions.get(source, ""), post_text, pre_text, window_text]
    defn_head = re.escape(definition[:30])
    sym_head = re.escape(sym.split("_", 1)[-1])
    for text in ordered:
        for sent in _split_sentences(text or ""):
            clean = _short(sent, 240)
            if not clean:
                continue
            if re.search(defn_head, clean, re.I):
                return clean
            if re.search(sym_head, clean, re.I) and _EXPLICIT_DEF_RE.search(clean):
                return clean
    return definition


def _audit_symbol_definitions(identifiers, symbol_defs, source_map,
                              pre_text, post_text, window_text):
    """Build a compact symbol-definition audit string."""
    if not identifiers:
        return "no identifiers found"
    defined = [sym for sym in identifiers if sym in symbol_defs]
    unexplained = [sym for sym in identifiers if sym not in symbol_defs]
    examples = []
    for sym in identifiers:
        if sym in symbol_defs:
            evidence = _definition_evidence(
                sym, symbol_defs[sym], source_map.get(sym, ""),
                pre_text, post_text, window_text,
            )
            category = _source_category(source_map.get(sym, ""), evidence)
            examples.append(f"{sym}={symbol_defs[sym]} [{category}]")
            if len(examples) >= 3:
                break
    out = [
        f"defined={len(defined)}/{len(identifiers)}",
        f"unexplained={len(unexplained)}",
    ]
    if examples:
        out.append("examples: " + "; ".join(examples))
    return "; ".join(out)


def _audit_relations(number, rel_dict):
    """Build a compact relation audit string for one equation."""
    if not rel_dict:
        return "remaining pairs: none"

    def infer_rule(entry):
        desc = entry.get("description", "")
        if desc == "shared notation":
            return "symbol_overlap"
        if desc in {"similar form", "same formula", "possible relation", "same section"}:
            return "structural_sim"
        if desc == "uses previous equation":
            return "explicit_ref"
        if desc in {"derivation", "substitution", "continuation"}:
            return "cue_phrase"
        return "structural_sim"

    def infer_evidence(entry):
        return entry.get("evidence") or entry.get("description") or entry.get("grade", "")

    counts = {"strong": 0, "potential": 0, "none": 0}
    highlights = []
    for other, entry in rel_dict.items():
        grade = entry.get("grade", "none")
        counts[grade] = counts.get(grade, 0) + 1
        if grade == "none" or len(highlights) >= 4:
            continue
        highlights.append(
            f"({number},{other})={grade}:{entry.get('description', '')}"
        )
    summary = (
        f"strong={counts.get('strong', 0)}, "
        f"potential={counts.get('potential', 0)}, "
        f"none={counts.get('none', 0)}"
    )
    if highlights:
        summary += "; examples: " + "; ".join(highlights)
    return summary


def _dedup_symbols(symbol_defs):
    """Drop decorated variants (hat_X) when base symbol X has the same definition.

    Parameters
    ----------
    symbol_defs : dict
        Mapping of normalized symbol to definition text.

    Returns
    -------
    dict
        Cleaned symbol dict with redundant decorated keys removed.
    """
    if not symbol_defs:
        return symbol_defs
    result = dict(symbol_defs)
    for key in list(result.keys()):
        if "_" in key:
            prefix, base = key.split("_", 1)
            if prefix in _DECORATOR_PREFIXES and base in result:
                if result[base] == result[key]:
                    del result[key]
    return result


def _extract_respectively(text, identifiers):
    """Recover definitions from 'A, B, C denote X, Y, Z, respectively'.

    The 'respectively' coordination structure is the most common definition
    pattern missed by standard Hearst matchers (identified as the top hard
    case in SymDef, Martin-Boyle et al. 2023). Runs over combined pre+post
    context and fills in only symbols already in identifiers that were not
    assigned a definition by find_symbol_definitions.

    Parameters
    ----------
    text : str
        Combined pre+post context around the equation.
    identifiers : list of str
        Normalized symbol keys as produced by symbols_extract.

    Returns
    -------
    dict
        Mapping of normalized symbol to definition text for matched pairs only.
    """
    result = {}
    for sent in _split_sentences(text):
        if not re.search(r"\brespectively\b", sent, re.I):
            continue
        for sym, defn in _sentence_definitions(sent, identifiers).items():
            result.setdefault(sym, defn)
    return result


def _extract_equation_lhs_definition(latex, identifiers):
    """Return a formula definition from an equation with one clear LHS symbol."""
    if not latex or not identifiers:
        return {}
    if re.search(r"\\iff|\\Leftrightarrow|\\leq|\\geq|≤|≥|<|>", latex):
        return {}
    eq_pat = r":=|\\coloneqq|\\equiv|(?<![<>])=(?!=)"
    parts = re.split(eq_pat, latex, maxsplit=1)
    if len(parts) != 2:
        return {}
    lhs, rhs = parts[0].strip(), parts[1].strip()
    if not lhs or not rhs or len(rhs) > 180:
        return {}
    if re.search(eq_pat, rhs):
        return {}
    if re.search(r"\\begin\{(?:array|matrix|pmatrix|bmatrix|cases)\}", rhs):
        return {}
    lhs_keys = _latex_structured_identifiers(lhs) | _latex_compact_identifiers(lhs)
    matches = [sym for sym in identifiers if sym in lhs_keys]
    matches = [sym for sym in matches if len(sym) > 1 or sym.isupper()]
    if len(matches) != 1:
        return {}
    rhs = re.sub(r"\s+", " ", rhs).strip()
    return {matches[0]: f"${rhs}$"}


def _build_table_index(tree):
    """Map each eq_id to its ltx_equation table node for O(1) lookup.

    Parameters
    ----------
    tree : lxml tree
        Parsed document tree.

    Returns
    -------
    dict
        Mapping of eq_id to table element.
    """
    index = {}
    for tab in tree.xpath('//table[contains(@class,"ltx_equation")]'):
        for span in tab.xpath('.//span[contains(@class,"ltx_tag_equation")]'):
            got = span.xpath('ancestor::*[@id][1]/@id')
            if got:
                index[got[0]] = tab
    return index


def process_paper(arxiv_id, n_equations=N_EQUATIONS):
    """Run the full extraction pipeline for one paper.

    Parameters
    ----------
    arxiv_id : str
        Bare arXiv ID such as '2403.05230'.
    n_equations : int
        How many enumerated equations to produce JSON for.

    Returns
    -------
    dict
        Maps printed equation number (str) to
        {equation, meaning, symbols, relations, audit-trail}.
        Empty dict when HTML is missing or no equations are found.
    """
    path = CACHE_DIR / f"{arxiv_id}.html"
    if not path.exists():
        print(f"{arxiv_id}: no cached HTML, skipping")
        return {}

    tree      = lxml_html.parse(str(path))
    equations = extract_equations(arxiv_id)
    if not equations:
        print(f"{arxiv_id}: no enumerated equations found")
        return {}

    table_index     = _build_table_index(tree)
    contexts        = get_contexts(arxiv_id)

    result          = {}
    pre_texts_map   = {}
    post_texts_map  = {}
    identifiers_map = {}
    section_map     = {}

    for eq in equations[:n_equations]:
        number = eq["number"]
        latex  = eq["latex"]
        eq_id  = eq["eq_id"]

        table_node = table_index.get(eq_id)
        if table_node is None:
            continue

        pre_text  = get_pre_text(table_node)
        post_text = get_post_text(table_node)
        signals   = extract_meaning_signals(table_node, latex, eq_id, pre_text, tree)

        identifiers  = extract_identifiers(arxiv_id, eq_id, latex)
        base_ctx     = contexts.get(eq_id, "")
        combined_ctx = f"{pre_text} [TARGET] {post_text} [WINDOW] {base_ctx}".strip()
        source_map   = {}
        symbol_defs  = find_symbol_definitions(identifiers, combined_ctx, _sources=source_map)
        symbol_defs  = _dedup_symbols(symbol_defs)

        resp_defs = _extract_respectively(pre_text + " " + post_text, identifiers)
        for sym, defn in resp_defs.items():
            if sym not in symbol_defs:
                symbol_defs[sym] = defn
                source_map[sym] = "respectively"

        meaning = build_meaning(signals, symbol_defs, latex=latex)
        if not meaning:
            meaning = "[no meaning]"
            signals["_meaning_rule"] = "no_meaning"
            signals["_meaning_evidence"] = f"no usable local prose; latex={_short(latex)}"

        _meaning_ev = signals.get("_meaning_evidence", "") or ""
        _meaning_val = (
            f"rule={signals.get('_meaning_rule', 'none')}, "
            f"evidence='{_short(_meaning_ev)}', "
            f"output='{_short(meaning)}'"
        )

        _sym_def_val = _audit_symbol_definitions(
            identifiers, symbol_defs, source_map,
            pre_text, post_text, base_ctx,
        )

        audit = {
            "extract_equations": (
                f"source=html; method=ltx_tag_equation; eq=({number})"
            ),
            "extract_identifiers": (
                f"method=mathml_leaves+latex_ast; found={len(identifiers)}"
                + (f"; examples={', '.join(identifiers[:6])}" if identifiers else "")
                + "; stop_list_dropped=0"
            ),
            "find_symbol_definitions": _sym_def_val,
            "build_meaning":           _meaning_val,
            "build_relations":         "pending: computed after all equations processed",
        }

        output_symbols = {sym: symbol_defs.get(sym, "") for sym in identifiers}

        result[number] = {
            "equation":    latex,
            "meaning":     meaning,
            "symbols":     output_symbols,
            "relations":   {},
            "audit-trail": audit,
        }

        pre_texts_map[number]   = pre_text
        post_texts_map[number]  = (
            post_text + " " + (signals.get("post_explanation") or "")
        ).strip()
        identifiers_map[number] = identifiers
        section_map[number]     = signals.get("contained_section", "")

        print(f"  ({number}) {meaning[:80] if meaning else '[no meaning]'}")

    if len(result) >= 2:
        eq_slice = [eq for eq in equations[:n_equations] if eq["number"] in result]
        relations = build_relations(
            equations       = eq_slice,
            table_index     = table_index,
            tree            = tree,
            pre_texts       = pre_texts_map,
            post_texts      = post_texts_map,
            identifiers_map = identifiers_map,
            section_map     = section_map,
        )
        for number, rel_dict in relations.items():
            if number in result:
                result[number]["relations"] = rel_dict
                # Summarise relation grades for the audit trail.
                pairs = [
                    f"({number},{other}): {entry['grade']}"
                    for other, entry in rel_dict.items()
                    if entry["grade"] != "none"
                ]
                result[number]["audit-trail"]["build_relations"] = _audit_relations(
                    number, rel_dict
                )

    return result
