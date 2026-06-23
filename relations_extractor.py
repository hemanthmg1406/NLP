"""Compute pairwise relations between enumerated equations in one paper.

Five non-generative signals are fused via a monotone decision rule:
1. Explicit cross-references: regex finds equation number patterns in context,
   spaCy parses the governing verb and maps it to a relation type via a cue lexicon.
2. Definitional dependency: if LHS(B) is in identifiers(A), equation B defines
   a quantity used in A — a directional strong relation.
3. Structural similarity (TED): MathML trees compared with Zhang-Shasha. High
   similarity with shared identifiers grades strong; without shared identifiers
   it grades potential (parallel form).
4. Symbol overlap (weighted Jaccard): 70% weight on exact normalized key match,
   30% on base-form match to preserve the operator/scalar distinction.
5. Textual similarity (TF-IDF cosine): prose context vectorized with unigrams
   and bigrams. Used only when Jaccard > 0; cosine alone is excluded because
   boilerplate prose produces non-zero overlap for unrelated same-section equations.

Decision rule priority: explicit_ref > lhs_defines > TED >= TREE_SIM_STRONG
> Jaccard >= JACCARD_POTENTIAL > TF-IDF cosine >= TFIDF_POTENTIAL (with j > 0)
> none.
"""

import re
from collections import Counter

import numpy as np
import spacy

from context_extract import _split_sentences
from mathml_tree import mathml_to_tree, tree_edit_distance

# TF-IDF cosines are naturally smaller than neural cosines (sparse vectors, no
# soft synonym similarity). Threshold calibrated on reviewed pairs: related
# pairs reach 0.25-0.60, unrelated pairs stay below 0.15.
TREE_SIM_STRONG   = 0.85
TFIDF_POTENTIAL   = 0.20
JACCARD_POTENTIAL = 0.40

_CUE_LEXICON = {
    "substitute": "substitution",
    "plug":       "substitution",
    "insert":     "substitution",
    "combine":    "combination",
    "derive":     "derivation",
    "obtain":     "derivation",
    "follow":     "derivation",
    "yield":      "derivation",
    "produce":    "derivation",
    "give":       "derivation",
    "get":        "derivation",
    "result":     "derivation",
    "simplify":   "equivalent",
    "equal":      "equivalent",
    "rewrite":    "equivalent",
    "express":    "equivalent",
    "reduce":     "special case",
    "recover":    "special case",
    "specialize": "special case",
    "generalize": "generalisation",
    "extend":     "generalisation",
    "negate":     "negation",
    "violate":    "negation",
    "contradict": "negation",
    "fit":        "operational dependency",
    "extract":    "operational dependency",
    "estimate":   "operational dependency",
    "measure":    "operational dependency",
    "determine":  "operational dependency",
    "calibrate":  "operational dependency",
    "limit":      "limit",
    "approach":   "limit",
    "take":       "limit",
    "replace":    "limit transformation",
    "discretize": "limit transformation",
    "coarsen":    "limit transformation",
}

_DECORATOR_PREFIXES = frozenset({
    "hat", "bar", "tilde", "vec", "widehat", "widetilde", "overline",
    "bm", "boldsymbol", "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf",
    "mathsf",
})

# Notation-style decorators where the decorated and plain forms name the same
# physical quantity (mathcal_F and F are the same object). Hat/bar/vec are
# excluded because they distinguish operators from scalars (hat_H != H).
_NOTATION_PREFIXES = frozenset({
    "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf", "mathsf"
})

_EQNUM_RE = re.compile(
    r"(?:Eqs?\.?|eqs?\.?|[Ee]quations?)\s*\(?(\d+(?:\.\d+)?[a-z]?)\)?"
    r"|"
    r"\((\d+(?:\.\d+)?[a-z]?)\)"
)

_LHS_RE = re.compile(
    r"^\s*(?:\\(?:hat|bar|tilde|vec|widehat|widetilde|overline|bm|boldsymbol"
    r"|mathcal|mathbb|mathscr|mathfrak|mathbf|mathsf)\{([^}]+)\}|([A-Za-z][A-Za-z0-9]*"
    r"(?:_\{[^}]+\}|_[A-Za-z0-9])?))"
)

_ZPF_CUES = re.compile(
    r'\b(?:zero[- ]?point\s+(?:fluctuation|motion|amplitude|flux)|'
    r'vacuum\s+(?:fluctuation|coupling|noise)|'
    r'zpf\b|x_\{?zpf\}?|'
    r'single\s+(?:phonon|photon|magnon)\s+coupling|'
    r'quantize|second\s+quantiz)',
    re.I,
)

_CLASSICAL_DELTA_RE = re.compile(
    r'\\delta\s*(?:[A-Za-z]|\{[^}]+\}|\\[A-Za-z]+)'
)

_ZPF_LATEX_RE = re.compile(
    r'\\zeta|zpf|zero.?point|x_\{?(?:\\mathrm\{)?zpf|\\phi_\{?(?:\\mathrm\{)?zpf',
    re.I,
)

_LIMIT_CUES = re.compile(
    r'\b(?:continuum\s+limit|thermodynamic\s+limit|mean.?field|replace\s+the\s+sum'
    r'|taking\s+[A-Z]\s*[→\-]\s*[∞\d]|density\s+of\s+states|infinite\s+volume'
    r'|macroscopic|infinite[-\s]N|N\s*[→\-]\s*∞|ensemble\s+average'
    r'|coarse.?grain|continuum\s+approximation)\b',
    re.I,
)

_SUM_RE = re.compile(r'\\sum\b')
_INT_RE = re.compile(r'\\int\b')

_LAMBDA       = 0.5
_REACH_THRESH = 0.2

_nlp = None


def _get_nlp():
    """Load spaCy en_core_web_sm on first call; return None if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp if _nlp is not False else None
    try:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except (OSError, IOError):
        _nlp = False
    return _nlp if _nlp is not False else None


def extract_lhs_symbol(latex):
    """Extract and normalize the primary symbol defined on the LHS of an equation.

    Takes everything before the first '=' and matches the outermost LaTeX
    identifier or decorated command. Returns the same no-backslash key form
    used by symbols_extract (e.g. 'hat_H', 'h_q'), or None when no clear LHS
    is found (e.g. commutators, brace expressions) or the LHS is a function
    application like H(t).
    """
    lhs_raw = latex.split("=")[0] if "=" in latex else latex
    lhs_raw = re.sub(r"^[\s\\{}\[\](|]+", "", lhs_raw).strip()

    m = _LHS_RE.match(lhs_raw)
    if not m:
        return None

    inner = (m.group(1) or m.group(2) or "").strip()
    if not inner:
        return None

    # Function application H(t;θ) is not a scalar definition.
    if lhs_raw[m.end():].lstrip().startswith("("):
        return None

    for prefix in ("hat", "bar", "tilde", "vec", "widehat", "widetilde",
                   "overline", "bm", "boldsymbol",
                   "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf"):
        if re.match(rf"^\\{prefix}\{{", lhs_raw.strip()):
            key = re.sub(r"[\\{}_]", "", inner).strip()
            return f"{prefix}_{key}" if key else None

    key = re.sub(r"\{([^}]+)\}", r"\1", inner)
    key = re.sub(r"[\\{}]", "", key).strip()
    return key if key else None


def _substitute_eqrefs(text, valid_numbers):
    """Replace equation number patterns with EQREF_N placeholder tokens.

    Only replaces numbers corresponding to actual equations in the paper to
    avoid false matches on citation numbers like (2024) or (Fig. 3).
    Returns (substituted_text, {eq_number: token_str}).
    """
    eqref_map = {}

    def _replace(m):
        num = m.group(1) or m.group(2)
        if num in valid_numbers:
            token = f"EQREF_{num.replace('.', '_')}"
            eqref_map[num] = token
            return token
        return m.group(0)

    return _EQNUM_RE.sub(_replace, text), eqref_map


def _extract_cue_verb(text, eqref_token):
    """Parse the governing verb of an EQREF token via spaCy dependency trees.

    Traverses up from the token's position (max 5 hops) to find the nearest
    governing verb, then maps its lemma via _CUE_LEXICON. Falls back to
    'reference' when spaCy is unavailable or no known verb is found.
    """
    nlp = _get_nlp()
    if nlp is None:
        return "reference"

    sentences = _split_sentences(text)
    target_sent = next((s for s in sentences if eqref_token in s), text)

    doc = nlp(target_sent)
    for token in doc:
        if token.text != eqref_token:
            continue
        head = token
        for _ in range(5):
            if head.pos_ == "VERB":
                return _CUE_LEXICON.get(head.lemma_.lower(), "reference")
            if head.head is head:
                break
            head = head.head
        for tok in doc:
            if tok.pos_ == "VERB":
                return _CUE_LEXICON.get(tok.lemma_.lower(), "reference")
        return "reference"

    return "reference"


def find_explicit_refs(source_num, context_text, valid_numbers,
                       source_latex=None, latex_map=None):
    """Detect all equation cross-references in one equation's context text.

    Two passes: (1) regex + spaCy to find '(N)' patterns and classify the
    governing cue verb; (2) limit-transformation pass that checks for
    discrete-to-continuum cue phrases paired with complementary aggregate
    operator types (sum vs integral) across referenced equations.

    Returns {target_num: relation_type_str} for all referenced equations found.
    """
    others = valid_numbers - {source_num}
    if not context_text or not others:
        return {}

    substituted, eqref_map = _substitute_eqrefs(context_text, others)
    refs = {target_num: _extract_cue_verb(substituted, token)
            for target_num, token in eqref_map.items()}

    if source_latex is not None and latex_map and _LIMIT_CUES.search(context_text):
        src_is_sum = bool(_SUM_RE.search(source_latex))
        src_is_int = bool(_INT_RE.search(source_latex))
        for num in others:
            tgt_latex = latex_map.get(num, "")
            tgt_is_sum = bool(_SUM_RE.search(tgt_latex))
            tgt_is_int = bool(_INT_RE.search(tgt_latex))
            if (src_is_sum and tgt_is_int) or (src_is_int and tgt_is_sum):
                if num not in refs:
                    refs[num] = "limit transformation"

    return refs


def _split_key(key):
    """Split 'hat_H' into ('hat', 'H'); plain 'H' into (None, 'H')."""
    if "_" in key:
        prefix, base = key.split("_", 1)
        if prefix in _DECORATOR_PREFIXES:
            return prefix, base
    return None, key


def weighted_jaccard(ids_a, ids_b):
    """Weighted two-tiered Jaccard similarity between identifier sets.

    Two tiers preserve the operator/scalar distinction: hat_H (operator) and H
    (scalar) are different physical quantities so exact match outweighs
    base-form match. Score = 0.7 * J_exact + 0.3 * J_base.
    """
    set_a, set_b = set(ids_a), set(ids_b)
    if not set_a and not set_b:
        return 0.0

    exact_union = len(set_a | set_b)
    j_exact = len(set_a & set_b) / exact_union if exact_union else 0.0

    bases_a = {_split_key(k)[1] for k in set_a}
    bases_b = {_split_key(k)[1] for k in set_b}
    base_union = len(bases_a | bases_b)
    j_base = len(bases_a & bases_b) / base_union if base_union else 0.0

    return 0.7 * j_exact + 0.3 * j_base


def build_tfidf_matrix(texts):
    """Fit TF-IDF on equation prose contexts and return an L2-normalised dense matrix.

    Uses sublinear_tf and unigrams+bigrams. No stop-word removal: physics prose
    reuses "is", "the", "where" in definitional sentences and removing them
    destroys the signal. min_df=1 because the corpus is at most 7 documents.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        token_pattern=r"(?u)\b\w[\w.]*\b",
    )
    sparse = vec.fit_transform(texts)
    return normalize(sparse, norm="l2").toarray()


def tfidf_cosine(matrix, i, j):
    """Cosine similarity between rows i and j of an L2-normalised TF-IDF matrix."""
    return float(np.dot(matrix[i], matrix[j]))


def classify_relation(tree_sim, jaccard_sim, cosine_sim,
                      explicit_ref_type=None, shared_ids=None,
                      lhs_defines=False):
    """Assign a relation grade and description via the monotone decision rule.

    Priority order: explicit_ref > lhs_defines > TED strong > TED potential
    (parallel form) > Jaccard >= JACCARD_POTENTIAL > TF-IDF (with j > 0) > none.
    Returns (grade, description); description is empty string for 'none'.
    """
    if shared_ids is None:
        shared_ids = set()

    if explicit_ref_type is not None:
        return "strong", explicit_ref_type

    if lhs_defines:
        return "strong", "defines component"

    if tree_sim >= TREE_SIM_STRONG:
        if shared_ids:
            sym_str = ", ".join(sorted(shared_ids)[:4])
            return "strong", f"equivalent — shared form and symbols ({sym_str})"
        return "potential", f"parallel form [tree_sim={tree_sim:.2f}]"

    if jaccard_sim >= JACCARD_POTENTIAL:
        sym_str = ", ".join(sorted(shared_ids)[:4]) if shared_ids else ""
        label = f"shared symbols ({sym_str})" if sym_str else "overlapping notation"
        return "potential", f"{label} [j={jaccard_sim:.2f}]"

    if cosine_sim >= TFIDF_POTENTIAL and jaccard_sim > 0:
        sym_str = ", ".join(sorted(shared_ids)[:4]) if shared_ids else ""
        label = (f"shared symbols ({sym_str}), contextually related"
                 if sym_str else "contextually related")
        return "potential", f"{label} [tfidf={cosine_sim:.2f}, j={jaccard_sim:.2f}]"

    return "none", ""


def build_relations(equations, table_index, tree, pre_texts,
                    post_texts, identifiers_map):
    """Compute all pairwise relations for the equations in one paper.

    Called once per paper after per-equation signals are extracted. Reuses
    pre_texts, post_texts, and identifiers from build_json.process_paper to
    avoid redundant DOM traversal. Returns {eq_number: {other_number: {grade,
    description}}} with an entry for every pair including grade 'none'.
    """
    numbers = [eq["number"] for eq in equations]
    n = len(numbers)
    result = {num: {} for num in numbers}

    if n < 2:
        return result

    valid_numbers = set(numbers)
    latex_map = {eq["number"]: eq["latex"] for eq in equations}

    explicit_refs = {}
    for eq in equations:
        num = eq["number"]
        context = (
            (pre_texts.get(num) or "") + " " + (post_texts.get(num) or "")
        ).strip()
        found = find_explicit_refs(
            num, context, valid_numbers,
            source_latex=eq["latex"],
            latex_map=latex_map,
        )
        for target_num, rel_type in found.items():
            key = (num, target_num)
            if key not in explicit_refs:
                explicit_refs[key] = rel_type

    lhs_map = {}
    for eq in equations:
        sym = extract_lhs_symbol(eq["latex"])
        if sym:
            lhs_map[eq["number"]] = sym

    math_trees = {eq["number"]: mathml_to_tree(table_index.get(eq["eq_id"]))
                  for eq in equations}

    prose_texts = [
        ((pre_texts.get(num) or "") + " " + (post_texts.get(num) or "")).strip()
        for num in numbers
    ]
    tfidf_mat = build_tfidf_matrix(prose_texts)
    num_idx = {num: i for i, num in enumerate(numbers)}

    # Symbols appearing in more than 70% of equations (minimum 3) are
    # high-frequency noise that inflate Jaccard across unrelated pairs.
    sym_freq = Counter(sym for num in numbers
                       for sym in (identifiers_map.get(num) or []))
    freq_stop = {sym for sym, cnt in sym_freq.items()
                 if cnt > 0.70 * n and cnt >= 3}
    if freq_stop:
        print(f"  jaccard stop-list ({len(freq_stop)} symbols): {sorted(freq_stop)}")

    lhs_strong_pairs = set()

    for i, num_a in enumerate(numbers):
        ids_a = identifiers_map.get(num_a) or []
        for j, num_b in enumerate(numbers):
            if i == j:
                continue

            ids_b   = identifiers_map.get(num_b) or []
            ref_type = explicit_refs.get((num_a, num_b))

            lhs_b = lhs_map.get(num_b)
            lhs_defines = False
            if lhs_b:
                ids_a_set = set(ids_a)
                if lhs_b in ids_a_set:
                    lhs_defines = True
                else:
                    parts = lhs_b.split("_", 1)
                    if len(parts) == 2 and parts[0] in _NOTATION_PREFIXES:
                        lhs_defines = parts[1] in ids_a_set

            t_sim = tree_edit_distance(math_trees[num_a], math_trees[num_b])

            ids_a_filt = [s for s in ids_a if s not in freq_stop]
            ids_b_filt = [s for s in ids_b if s not in freq_stop]
            j_sim   = weighted_jaccard(ids_a_filt, ids_b_filt)
            c_sim   = tfidf_cosine(tfidf_mat, num_idx[num_a], num_idx[num_b])
            shared  = set(ids_a_filt) & set(ids_b_filt)

            grade, desc = classify_relation(
                t_sim, j_sim, c_sim, ref_type, shared, lhs_defines
            )

            if grade == "strong" and lhs_defines and ref_type is None:
                lhs_strong_pairs.add((num_a, num_b))

            entry = {"grade": grade}
            if desc:
                entry["description"] = desc
            result[num_a][num_b] = entry

    # Classical-to-quantum (ZPF) derivation: classical equations contain
    # delta-prefixed symbols (δω, δΦ); the quantum counterpart evaluates that
    # at one zero-point fluctuation. TED and Jaccard fail here because symbols
    # change completely. Detection uses each equation's own LaTeX, not shared
    # context text.
    zpf_nums = {eq["number"] for eq in equations if _ZPF_LATEX_RE.search(eq["latex"])}
    classical_delta_nums = {eq["number"] for eq in equations
                            if _CLASSICAL_DELTA_RE.search(eq["latex"])}
    if zpf_nums and classical_delta_nums:
        print(f"  ZPF equations: {sorted(zpf_nums)}, "
              f"classical-delta equations: {sorted(classical_delta_nums)}")
    for q_num in zpf_nums:
        for c_num in classical_delta_nums:
            if c_num == q_num:
                continue
            for src, tgt, label in [
                (c_num, q_num, "classical-to-quantum derivation"),
                (q_num, c_num, "quantum coupling derived from classical expression"),
            ]:
                if result.get(src, {}).get(tgt, {}).get("grade") != "strong":
                    result[src][tgt] = {"grade": "strong", "description": label}

    # Bidirectional definitional dependency: if B defines a component of A,
    # then from B's perspective A is the equation that depends on what B defines.
    for num_a, num_b in lhs_strong_pairs:
        if result.get(num_b, {}).get(num_a, {}).get("grade") != "strong":
            result[num_b][num_a] = {
                "grade": "strong",
                "description": "component used in derivation",
            }

    # Bidirectionalize all strong pairs. A mathematical dependency between two
    # equations is symmetric: the referenced equation is as related to the
    # referencing one as vice versa.
    for num_a in numbers:
        for num_b in numbers:
            if num_a == num_b:
                continue
            if result.get(num_a, {}).get(num_b, {}).get("grade") == "strong":
                if result.get(num_b, {}).get(num_a, {}).get("grade") != "strong":
                    fwd_desc = result[num_a][num_b].get("description", "strong relation")
                    result[num_b][num_a] = {
                        "grade": "strong",
                        "description": f"bidirectional: {fwd_desc}",
                    }

    return _dag_reachability(result, numbers)


def _dag_reachability(relations, numbers):
    """BFS reachability over the strong-edge DAG with lambda-decay path scoring.

    Upgrades 'none' pairs to 'potential' when the max-product path score over
    strong edges exceeds _REACH_THRESH. Score formula: lambda^(hops-1) per
    path, max over all paths. Shorter paths score higher since lambda < 1,
    penalizing longer indirect chains. Existing 'potential' or 'strong' grades
    are never overwritten.
    """
    strong_adj = {
        num: {other for other, entry in relations.get(num, {}).items()
              if entry.get("grade") == "strong"}
        for num in numbers
    }

    for src in numbers:
        visited = {src: 1.0}
        queue   = [(src, 1.0, None)]

        while queue:
            node, score, via = queue.pop(0)
            hop_score = score * _LAMBDA

            for nbr in strong_adj.get(node, set()):
                if nbr == src or hop_score <= visited.get(nbr, 0.0):
                    continue
                visited[nbr] = hop_score
                queue.append((nbr, hop_score, via if via is not None else nbr))

        for tgt, path_score in visited.items():
            if tgt == src or path_score >= 1.0 or path_score < _REACH_THRESH:
                continue
            if relations.get(src, {}).get(tgt, {}).get("grade") == "none":
                via_node = next(
                    (v for v in strong_adj.get(src, set())
                     if v != tgt and tgt in visited and
                     relations.get(src, {}).get(v, {}).get("grade") == "strong"),
                    None,
                )
                via_str = f" via ({via_node})" if via_node else ""
                relations[src][tgt] = {
                    "grade": "potential",
                    "description": f"indirect dependency{via_str} [path_score={path_score:.2f}]",
                }

    return relations
