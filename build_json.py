"""Build the equations knowledge-graph JSON for a selection of arXiv papers.

Set LIMIT to control how many papers are processed per run and N_EQUATIONS for
how many enumerated equations to extract per paper. Output is written to
output.json and merged with any prior run so partial results are preserved.
"""

import json
import re
from pathlib import Path

from lxml import html as lxml_html

import robot_fetch
from review_equations import extract_equations
from context_extract import get_contexts, _split_sentences
from symbols_extract import (
    extract_identifiers,
    find_symbol_definitions,
)
from meaning_extractor import (
    get_pre_text,
    get_post_text,
    extract_meaning_signals,
    build_meaning,
)
from relations_extractor import build_relations

LIMIT       = 5
N_EQUATIONS = 7

CACHE_DIR      = Path("cache")
PAPER_LIST     = "paper_list_29.txt"
OUTPUT_FILE    = Path("output.json")
LAST_RUN_FILE  = Path(".last_run")
PROCESSED_FILE = Path(".processed")


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


def load_processed():
    """Return the set of arXiv IDs already dealt with.

    Returns
    -------
    set of str
    """
    if PROCESSED_FILE.exists():
        return set(PROCESSED_FILE.read_text(encoding="utf-8").split())
    return set()


def mark_processed(arxiv_id):
    """Append arxiv_id to the processed file so it is skipped on future runs.

    Parameters
    ----------
    arxiv_id : str
    """
    with PROCESSED_FILE.open("a", encoding="utf-8") as f:
        f.write(arxiv_id + "\n")


def select_next(all_ids, limit=LIMIT):
    """Return up to limit unprocessed IDs in document order.

    Parameters
    ----------
    all_ids : list of str
    limit : int

    Returns
    -------
    list of str
    """
    processed = load_processed()
    return [aid for aid in all_ids if aid not in processed][:limit]


# Decorator prefixes from symbols_extract that produce keys like 'hat_H'.
# When both 'hat_H' and 'H' have the same definition the decorated form is
# redundant and is dropped.
_DECORATOR_PREFIXES = {
    "hat", "bar", "tilde", "vec", "widehat", "widetilde", "overline",
    "bm", "boldsymbol", "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf",
    "mathsf",
}


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


_INLINE_MATH_RE  = re.compile(r'\$([^$]+)\$')
_RESP_TRIGGER_RE = re.compile(
    r'(?:where\s+)?(.+?)\s+'
    r'(?:denote|denotes|are|is|represent|represents|stand\s+for|refer(?:s)?\s+to)\s+'
    r'(.+?),?\s*respectively',
    re.I,
)


def _normalize_sym(latex):
    r"""Map a raw LaTeX identifier to the normalized key used by symbols_extract.

    Examples: '\\hat{H}' -> 'hat_H', '\\rho' -> 'rho', 'H_0' -> 'H_0'.

    Parameters
    ----------
    latex : str

    Returns
    -------
    str
    """
    latex = latex.strip()
    for prefix in _DECORATOR_PREFIXES:
        m = re.match(rf"^\\{prefix}\{{([^}}]+)\}}$", latex)
        if m:
            inner = re.sub(r"[\\{}]", "", m.group(1)).strip()
            return f"{prefix}_{inner}"
    return re.sub(r"[\\{}]", "", latex).strip()


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
    id_set = set(identifiers)
    for sent in _split_sentences(text):
        m = _RESP_TRIGGER_RE.search(sent)
        if not m:
            continue
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        syms_raw = _INLINE_MATH_RE.findall(lhs)
        if len(syms_raw) < 2:
            continue
        defs_raw = re.split(r',\s*(?:and\s+)?|\s+and\s+', rhs)
        defs_raw = [d.strip() for d in defs_raw if d.strip()]
        if len(syms_raw) != len(defs_raw):
            continue
        for sym_latex, defn in zip(syms_raw, defs_raw):
            key = _normalize_sym(sym_latex)
            if key in id_set:
                defn = re.sub(r'^the\s+', '', defn, flags=re.I).strip()
                if defn:
                    result[key] = defn
    return result


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

    for eq in equations[:n_equations]:
        number = eq["number"]
        latex  = eq["latex"]
        eq_id  = eq["eq_id"]

        table_node = table_index.get(eq_id)
        if table_node is None:
            print(f"  eq ({number}): table node not found, skipping")
            continue

        pre_text  = get_pre_text(table_node)
        post_text = get_post_text(table_node)
        signals   = extract_meaning_signals(table_node, latex, eq_id, pre_text, tree)

        identifiers  = extract_identifiers(arxiv_id, eq_id, latex)
        base_ctx     = contexts.get(eq_id, "")
        combined_ctx = (base_ctx + " " + post_text).strip()
        source_map   = {}
        symbol_defs  = find_symbol_definitions(identifiers, combined_ctx, _sources=source_map)
        symbol_defs  = _dedup_symbols(symbol_defs)

        resp_defs = _extract_respectively(pre_text + " " + post_text, identifiers)
        for sym, defn in resp_defs.items():
            if sym not in symbol_defs:
                symbol_defs[sym] = defn
                source_map[sym] = "respectively"

        meaning = build_meaning(signals, symbol_defs, latex=latex)

        audit = {
            "source":               "html",
            "model":                "encoder/classifier only — no generative model",
            "inline_label":         signals["inline_label"] or "none",
            "theorem_env":          signals["theorem_env"] or "none",
            "theorem_title":        signals["theorem_title"][:80] if signals["theorem_title"] else "none",
            "named_eq":             signals["named_eq"] or "none",
            "section_title":        signals["contained_section"] or "not found",
            "section_is_generic":   signals["section_is_generic"],
            "section_used_as_fallback": signals.get("_section_fallback", False),
            "intro_sentence":       signals["intro_sentence"][:120] if signals["intro_sentence"] else "none",
            "lead_in_phrase":       signals["lead_in_phrase"][:120] if signals.get("lead_in_phrase") else "none",
            "post_context_80":      post_text[:80] if post_text else "none",
            "post_explanation":     signals.get("post_explanation", "")[:120] or "none",
            "pre_context_120":      pre_text[:120] if pre_text else "none",
            "cross_ref":            signals["cross_ref"][:120] if signals["cross_ref"] else "none",
            "abbreviation_sh":      signals["abbrev"] or "none",
            "meaning_lhs":          signals.get("_meaning_lhs", "none") or "none",
            "meaning_shape":        signals.get("_meaning_shape", "unknown"),
            "meaning_rule":         signals.get("_meaning_rule", "none"),
            "meaning_source":       signals.get("_meaning_source", "none"),
            "meaning_evidence":     signals.get("_meaning_evidence", "none") or "none",
            "meaning_method":       "synth_first:lead_in+intro+post_expl+post_where+lhs_shape+named_eq+proof_step+section_fallback",
            "respectively_syms":    list(resp_defs.keys()) if resp_defs else "none",
            "identifiers":          identifiers,
            "symbol_defs_found":    list(symbol_defs.keys()),
            "symbol_def_sources":   {s: source_map.get(s, "unknown") for s in symbol_defs},
        }

        result[number] = {
            "equation":    latex,
            "meaning":     meaning,
            "symbols":     symbol_defs,
            "relations":   {},
            "audit-trail": audit,
        }

        pre_texts_map[number]   = pre_text
        post_texts_map[number]  = post_text
        identifiers_map[number] = identifiers

        print(f"  eq ({number}): {meaning[:120] if meaning else '[no meaning]'}")

    if len(result) >= 2:
        eq_slice = [eq for eq in equations[:n_equations] if eq["number"] in result]
        print(f"  computing relations for {len(eq_slice)} equations...")
        relations = build_relations(
            equations       = eq_slice,
            table_index     = table_index,
            tree            = tree,
            pre_texts       = pre_texts_map,
            post_texts      = post_texts_map,
            identifiers_map = identifiers_map,
        )
        for number, rel_dict in relations.items():
            if number in result:
                result[number]["relations"] = rel_dict

    return result


def main():
    """Fetch HTML for the next batch of papers, run the pipeline, write output.json.

    On each run the user chooses between re-running the last HTML batch (r) or
    advancing to the next LIMIT unprocessed papers (n). Results are merged into
    output.json so prior work is never lost.
    """
    all_ids = load_paper_ids()

    last   = LAST_RUN_FILE.read_text().split() if LAST_RUN_FILE.exists() else []
    choice = "n"
    if last:
        nxt = select_next(all_ids, limit=LIMIT)
        print(f"Last run : {last}")
        print(f"Next     : {nxt}")
        choice = input("(r) rerun last   (n) next papers: ").strip().lower()

    if choice == "r":
        candidates = last
    else:
        candidates = select_next(all_ids, limit=LIMIT)

    if not candidates:
        print("All papers processed.")
        return

    needs_fetch = any(
        not (CACHE_DIR / f"{aid}.html").exists() for aid in candidates
    )
    rp = robot_fetch._robots() if needs_fetch else None

    if OUTPUT_FILE.exists():
        try:
            full_output = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            full_output = {}
    else:
        full_output = {}

    html_this_run = []

    for arxiv_id in candidates:
        print(f"=== {arxiv_id} ===")

        html_path = CACHE_DIR / f"{arxiv_id}.html"
        if html_path.exists():
            kind = "html"
        elif rp is not None:
            _, kind = robot_fetch.fetch_one(arxiv_id, rp)
        else:
            kind = "missing"

        if kind != "html":
            print(f"  {arxiv_id}: no HTML version available, skipping")
            if choice != "r":
                mark_processed(arxiv_id)
            print()
            continue

        paper_result = process_paper(arxiv_id, n_equations=N_EQUATIONS)
        full_output[arxiv_id] = paper_result

        html_this_run.append(arxiv_id)
        if choice != "r":
            mark_processed(arxiv_id)

        if not paper_result:
            print(f"  {arxiv_id}: HTML found but no enumerated equations extracted")
        print()

    OUTPUT_FILE.write_text(
        json.dumps(full_output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if html_this_run:
        LAST_RUN_FILE.write_text("\n".join(html_this_run), encoding="utf-8")

    print(f"Saved {len(full_output)} paper(s) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
