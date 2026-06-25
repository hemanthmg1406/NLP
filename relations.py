"""Compute pairwise relations between enumerated equations in one paper.

1. Explicit cross-references: regex finds equation number patterns in context,
   spaCy parses the governing verb and maps it to a relation type via a cue lexicon.
2. Definitional dependency: if the right-hand side of A uses the simple LHS
   symbol defined by B, B is marked as a formula dependency of A.
3. Structural similarity (TED): MathML trees compared with Zhang-Shasha.
   Similarity can mark a parallel form, but does not create a strong relation.
4. Symbol overlap (weighted Jaccard): 70% weight on exact normalized key match,
   30% on base-form match to preserve the operator/scalar distinction.
5. Textual similarity (SciBERT cosine): prose context encoded by
   allenai/scibert_scivocab_uncased; mean-pooled last hidden state. Falls back
   to TF-IDF when the model is unavailable. Used only as supporting evidence.

Decision rule priority: explicit_ref > rhs_lhs_dependency > conservative potential
signals > none. No topic-specific relation shortcuts are applied.
"""

import re
from collections import Counter

import numpy as np
import spacy

from context import _split_sentences
from mathml_tree import mathml_to_tree, tree_edit_distance
from symbols import _latex_compact_identifiers, _latex_structured_identifiers

# SciBERT cosine similarity threshold. Neural embeddings produce higher baseline
# cosines than TF-IDF (soft synonym overlap). Calibrated on reviewed pairs:
# related pairs cluster at 0.80-0.95, unrelated same-section pairs at 0.65-0.75.
TREE_SIM_STRONG    = 0.85
SCIBERT_POTENTIAL  = 0.78
JACCARD_POTENTIAL  = 0.55
_MIN_SHARED_SYMBOLS = 2
# Keep the old name as alias so any external import still works.
TFIDF_POTENTIAL    = SCIBERT_POTENTIAL

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
    "rewrite":    "derivation",
    "express":    "derivation",
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

_PREFIXED_EQNUM_RE = re.compile(
    r"\b(?:Eqs?\.?|eqs?\.?|[Ee]quations?|[Ff]ormulas?)\s*"
    r"\(?\s*(\d+(?:\.\d+)?[a-z]?)\s*\)?"
)

_BARE_EQNUM_RE = re.compile(r"(?<![A-Za-z0-9])\((\d+(?:\.\d+)?[a-z]?)\)")

_BARE_REF_CUE_RE = re.compile(
    r"\b(?:above|below|by|combine[sd]?|derived?|eqs?\.?|equations?|follows?|"
    r"formulae?|formulas?|from|insert(?:ing|ed)?|plug(?:ging|ged)?|prove|"
    r"reference|relations?|show|shown|substitut(?:e|ed|ing)|suffices?|use[sd]?"
    r"|using)\b",
    re.I,
)

_DERIVATION_CONTEXT_RE = re.compile(
    r"\b(?:as a result|becomes?|derivative of|derive[sd]?|differentiat(?:e|ed|ing)"
    r"|follows?|from the|hence|net\s+\w+\s+becomes?|obtain(?:ed)?|reduce[sd]?"
    r"|simplif(?:y|ies|ied)|therefore|thus|total\s+\w+\s+becomes?|we get|"
    r"which gives|which yields|yields?)\b",
    re.I,
)

_LHS_RE = re.compile(
    r"^\s*(?:\\(?:hat|bar|tilde|vec|widehat|widetilde|overline|bm|boldsymbol"
    r"|mathcal|mathbb|mathscr|mathfrak|mathbf|mathsf)\{([^}]+)\}|([A-Za-z][A-Za-z0-9]*"
    r"(?:_\{[^}]+\}|_[A-Za-z0-9])?))"
)

_EQUALITY_RE = re.compile(r":=|\\coloneqq|\\equiv|(?<![<>])=(?!=)")

_LAMBDA       = 0.5
_REACH_THRESH = 0.2

_WEAK_RELATION_SYMBOLS = frozenset({
    "cal", "det", "dim", "exp", "iff", "log", "max", "min", "mod", "Pr",
    "Re", "Im", "rm", "sgn", "sin", "cos", "tan", "tr", "Tr",
})

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


def _reference_type_from_window(window, default="explicit equation reference"):
    """Classify a local equation-reference phrase without topic shortcuts."""
    w = window.lower()
    if re.search(r"\b(substitut(?:e|ed|ing)|insert(?:ing|ed)?|plug(?:ging|ged)?)\b", w):
        return "substitution"
    if re.search(r"\b(prove|show|suffices?)\b", w):
        return "proof dependency"
    if re.search(r"\b(combine[sd]?|from|using|used|by)\b", w):
        return "uses referenced equation"
    if re.search(r"\b(derive[sd]?|follows?|obtain(?:ed)?|result|yield(?:s|ed)?)\b", w):
        return "derivation"
    return default


def _substitute_eqrefs(text, valid_numbers):
    """Replace equation number patterns with EQREF_N placeholder tokens.

    Only replaces numbers corresponding to actual equations in the paper to
    avoid false matches on citation numbers like (2024) or (Fig. 3).
    Returns (substituted_text, {eq_number: token_str}).
    """
    eqref_map = {}
    replacements = []

    for pattern, require_cue in (
        (_PREFIXED_EQNUM_RE, False),
        (_BARE_EQNUM_RE, True),
    ):
        for m in pattern.finditer(text):
            num = m.group(1)
            if num not in valid_numbers:
                continue
            start, end = m.span()
            window = text[max(0, start - 70): min(len(text), end + 70)]
            if require_cue and not _BARE_REF_CUE_RE.search(window):
                continue
            if any(not (end <= s or start >= e) for s, e, _ in replacements):
                continue
            token = f"EQREF_{num.replace('.', '_')}"
            eqref_map[num] = (token, _reference_type_from_window(window))
            replacements.append((start, end, token))

    if not replacements:
        return text, {}

    pieces = []
    cursor = 0
    for start, end, token in sorted(replacements):
        pieces.append(text[cursor:start])
        pieces.append(token)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), eqref_map


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

    Returns {target_num: relation_type_str} for all referenced equations found.
    """
    others = valid_numbers - {source_num}
    if not context_text or not others:
        return {}

    substituted, eqref_map = _substitute_eqrefs(context_text, others)
    refs = {}
    for target_num, (token, local_type) in eqref_map.items():
        parsed_type = _extract_cue_verb(substituted, token)
        refs[target_num] = parsed_type if parsed_type != "reference" else local_type

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


def _relation_symbol_is_informative(symbol):
    """Filter notation tokens that are too weak for relation evidence."""
    if not symbol:
        return False
    if symbol in _WEAK_RELATION_SYMBOLS:
        return False
    if len(symbol) == 1:
        return False
    if re.fullmatch(r"d[a-z]", symbol):
        return False
    letters = re.sub(r"[^A-Za-z]", "", symbol)
    if len(letters) <= 1:
        return False
    return True


def _has_overlap_evidence(shared_ids):
    """Return True when shared symbols are specific enough to link equations."""
    shared_ids = {s for s in shared_ids if _relation_symbol_is_informative(s)}
    if len(shared_ids) >= _MIN_SHARED_SYMBOLS:
        return True
    return any("_" in s and len(re.sub(r"[^A-Za-z]", "", s)) >= 2
               for s in shared_ids)


def _rhs_identifier_set(latex):
    """Return structured identifiers from the right-hand side of an equation."""
    if not latex or not _EQUALITY_RE.search(latex):
        return set()
    parts = _EQUALITY_RE.split(latex, maxsplit=1)
    rhs = parts[1] if len(parts) == 2 else ""
    return _latex_structured_identifiers(rhs) | _latex_compact_identifiers(rhs)


def _lhs_identifier_set(latex):
    """Return normalized identifiers from the left-hand side of an equation."""
    if not latex or not _EQUALITY_RE.search(latex):
        return set()
    lhs = _EQUALITY_RE.split(latex, maxsplit=1)[0]
    return _latex_structured_identifiers(lhs) | _latex_compact_identifiers(lhs)


def _clear_lhs_identifier_set(latex):
    """Return LHS identifiers only when the LHS is a clear defined quantity.

    This intentionally rejects function applications, products, limits, bras/kets,
    and bracketed cocycles. Their LHS contains many bound variables, which creates
    false strong edges when those variables recur later.
    """
    if not latex or not _EQUALITY_RE.search(latex):
        return set()
    lhs = _EQUALITY_RE.split(latex, maxsplit=1)[0].strip()
    if not lhs:
        return set()
    if re.match(r"^\\(?:begin|lim|prod|sum|int)(?:\b|_)", lhs):
        return set()
    if lhs.startswith("[") or re.search(r"\\(?:ket|bra|braket)\b", lhs):
        return set()
    if re.search(r"(?<![_A-Za-z])\(", lhs) or r"\left(" in lhs:
        return set()

    ids = _lhs_identifier_set(latex)
    candidates = {
        s for s in ids
        if _relation_symbol_is_informative(s) or (len(s) == 1 and s.isupper())
    }
    if len(candidates) == 1:
        return candidates
    return set()


def _symbol_matches_with_notation_prefix(symbol, ids):
    """Match a normalized symbol against RHS identifiers with notation aliases."""
    if symbol in ids:
        return True
    parts = symbol.split("_", 1)
    if len(parts) == 2 and parts[0] in _NOTATION_PREFIXES:
        return parts[1] in ids
    for ident in ids:
        iparts = ident.split("_", 1)
        if len(iparts) == 2 and iparts[0] in _NOTATION_PREFIXES:
            if iparts[1] == symbol:
                return True
    return False


def _lhs_sets_match(lhs_ids, rhs_ids):
    """True when one clear LHS identifier is reused on another equation RHS."""
    for sym in lhs_ids:
        if not _relation_symbol_is_informative(sym):
            continue
        if _symbol_matches_with_notation_prefix(sym, rhs_ids):
            return True
    return False


def _lhs_rollup_match(lhs_a, lhs_b):
    """Detect aggregate/specific LHS pairs such as I and I_n_m."""
    for a in lhs_a:
        for b in lhs_b:
            if a == b:
                continue
            if len(a) == 1 and a.isupper() and b.startswith(a + "_"):
                return True
            if len(b) == 1 and b.isupper() and a.startswith(b + "_"):
                return True
    return False


def _has_adjacent_derivation(context, shared_ids, lhs_a, lhs_b):
    """High-precision derivation cue for immediately consecutive equations."""
    if not context or not _DERIVATION_CONTEXT_RE.search(context):
        return False
    informative_shared = {
        s for s in shared_ids
        if _relation_symbol_is_informative(s)
    }
    return _lhs_rollup_match(lhs_a, lhs_b) and bool(informative_shared)


_scibert_model = None
_scibert_tokenizer = None
_scibert_available = None  # None = untested, False = unavailable, True = loaded


def _load_scibert():
    """Load allenai/scibert_scivocab_uncased on first call.

    Uses HuggingFace transformers with CUDA when available, CPU otherwise.
    Sets _scibert_available to False on failure so subsequent calls skip loading.
    Returns (tokenizer, model) or (None, None).
    """
    global _scibert_model, _scibert_tokenizer, _scibert_available
    if _scibert_available is False:
        return None, None
    if _scibert_available is True:
        return _scibert_tokenizer, _scibert_model
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        model_name = "allenai/scibert_scivocab_uncased"
        _scibert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _scibert_model = AutoModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _scibert_model = _scibert_model.to(device)
        _scibert_model.eval()
        _scibert_available = True
    except Exception:
        _scibert_available = False
        return None, None
    return _scibert_tokenizer, _scibert_model


def _scibert_embed(texts):
    """Encode a list of strings with SciBERT and return an L2-normalised matrix.

    Mean-pools the last hidden state over non-padding tokens. Falls back to
    TF-IDF when SciBERT is unavailable so the pipeline degrades gracefully.
    Returns (matrix, source_label) where source_label is 'scibert' or 'tfidf'.
    """
    tok, model = _load_scibert()
    if tok is None:
        return _tfidf_fallback(texts), "tfidf"

    import torch
    device = next(model.parameters()).device
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tok(
                text[:512],
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            embeddings.append(emb.squeeze(0).cpu().numpy())

    mat = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms, "scibert"


def _tfidf_fallback(texts):
    """TF-IDF L2-normalised matrix as a fallback when SciBERT is unavailable."""
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


def build_tfidf_matrix(texts):
    """Compatibility shim: returns (matrix, 'tfidf'). Prefer _scibert_embed."""
    return _tfidf_fallback(texts), "tfidf"


def tfidf_cosine(matrix, i, j):
    """Cosine similarity between rows i and j of an L2-normalised matrix."""
    return float(np.dot(matrix[i], matrix[j]))


def classify_relation(tree_sim, jaccard_sim, cosine_sim,
                      explicit_ref_type=None, shared_ids=None,
                      lhs_dependency=False, same_section=True,
                      adjacent_derivation=False,
                      cosine_source="scibert"):
    """Assign a relation grade and description via a conservative decision rule.

    Only explicit references and RHS-to-LHS formula dependency produce strong
    edges. Tree/Jaccard/cosine evidence can produce potential edges, and only
    when the shared symbols are specific enough to avoid common-notation
    matches.

    Returns (grade, description); description is empty string for 'none'.
    """
    if shared_ids is None:
        shared_ids = set()
    shared_ids = {s for s in shared_ids if _relation_symbol_is_informative(s)}
    has_overlap_evidence = _has_overlap_evidence(shared_ids)

    if explicit_ref_type is not None:
        return "strong", explicit_ref_type

    if lhs_dependency:
        return "strong", "uses defined quantity"

    if adjacent_derivation:
        return "strong", "adjacent derivation"

    if tree_sim >= TREE_SIM_STRONG:
        if shared_ids and has_overlap_evidence and same_section:
            sym_str = ", ".join(sorted(shared_ids)[:4])
            return "potential", f"parallel form with shared symbols ({sym_str}) [tree_sim={tree_sim:.2f}]"
        return "potential", f"parallel form [tree_sim={tree_sim:.2f}]"

    if same_section and has_overlap_evidence and jaccard_sim >= JACCARD_POTENTIAL:
        sym_str = ", ".join(sorted(shared_ids)[:4])
        return "potential", f"shared symbols ({sym_str}) [j={jaccard_sim:.2f}]"

    if (same_section and has_overlap_evidence and
            cosine_sim >= SCIBERT_POTENTIAL and
            jaccard_sim >= JACCARD_POTENTIAL):
        sym_str = ", ".join(sorted(shared_ids)[:4])
        return "potential", (
            f"shared symbols ({sym_str}), contextually related "
            f"[{cosine_source}={cosine_sim:.2f}, j={jaccard_sim:.2f}]"
        )

    return "none", ""


def _top_section(section_name):
    """Return the top-level section label (e.g. 'Model' from 'Model.Subsection').

    arXiv LaTeXML section titles are often compound strings; strip everything
    after the first separator to get the root section for cross-section comparison.
    """
    if not section_name:
        return ""
    # Split on period, colon, dash or em-dash that separates a section hierarchy.
    return re.split(r"[.:]\s*|\s+[-–—]\s+", section_name.strip())[0].strip().lower()


def build_relations(equations, table_index, tree, pre_texts,
                    post_texts, identifiers_map, section_map=None):
    """Compute all pairwise relations for the equations in one paper.

    Called once per paper after per-equation signals are extracted. Reuses
    pre_texts, post_texts, and identifiers from build_json.process_paper to
    avoid redundant DOM traversal.

    section_map is an optional {eq_number: section_title} dict used for the
    section-distance guard. When absent, the guard is skipped.

    Returns {eq_number: {other_number: {grade, description}}} with an entry
    for every pair including grade 'none'.
    """
    numbers = [eq["number"] for eq in equations]
    n = len(numbers)
    result = {num: {} for num in numbers}

    if n < 2:
        return result

    valid_numbers = set(numbers)
    latex_map = {eq["number"]: eq["latex"] for eq in equations}
    rhs_ids_map = {num: _rhs_identifier_set(latex)
                   for num, latex in latex_map.items()}
    lhs_ids_map = {num: _lhs_identifier_set(latex)
                   for num, latex in latex_map.items()}
    clear_lhs_ids_map = {num: _clear_lhs_identifier_set(latex)
                         for num, latex in latex_map.items()}

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
    embed_mat, embed_source = _scibert_embed(prose_texts)
    num_idx = {num: i for i, num in enumerate(numbers)}

    # Symbols appearing in more than 70% of equations (minimum 3) are
    # high-frequency noise that inflate Jaccard across unrelated pairs.
    sym_freq = Counter(sym for num in numbers
                       for sym in (identifiers_map.get(num) or []))
    freq_stop = {sym for sym, cnt in sym_freq.items()
                 if cnt > 0.70 * n and cnt >= 3}

    # Build top-section lookup for section-distance guard.
    top_sections = {}
    if section_map:
        for num in numbers:
            top_sections[num] = _top_section(section_map.get(num, ""))

    for i, num_a in enumerate(numbers):
        ids_a   = identifiers_map.get(num_a) or []
        latex_a = latex_map.get(num_a, "")
        for j, num_b in enumerate(numbers):
            if i == j:
                continue

            ids_b    = identifiers_map.get(num_b) or []
            latex_b  = latex_map.get(num_b, "")
            ref_type = explicit_refs.get((num_a, num_b))

            lhs_b = lhs_map.get(num_b)
            lhs_dependency = False
            adjacent_derivation = False

            # Section-distance guard: equations in different top-level sections
            # share common algebra symbols legitimately but are usually not
            # directly related. A Jaccard or cosine-only strong grade is
            # downgraded to potential when sections differ.
            same_section = True
            if top_sections:
                sec_a = top_sections.get(num_a, "")
                sec_b = top_sections.get(num_b, "")
                if sec_a and sec_b and sec_a != sec_b:
                    same_section = False

            t_sim = tree_edit_distance(math_trees[num_a], math_trees[num_b])

            ids_a_filt = [
                s for s in ids_a
                if s not in freq_stop and _relation_symbol_is_informative(s)
            ]
            ids_b_filt = [
                s for s in ids_b
                if s not in freq_stop and _relation_symbol_is_informative(s)
            ]
            j_sim  = weighted_jaccard(ids_a_filt, ids_b_filt)
            c_sim  = tfidf_cosine(embed_mat, num_idx[num_a], num_idx[num_b])
            shared = set(ids_a_filt) & set(ids_b_filt)

            rhs_ids_a = {
                s for s in rhs_ids_map.get(num_a, set())
                if s not in freq_stop and _relation_symbol_is_informative(s)
            }

            if i > j:
                clear_lhs_b = {
                    s for s in clear_lhs_ids_map.get(num_b, set())
                    if s not in freq_stop
                }
                if (lhs_b and lhs_b not in freq_stop and
                        _relation_symbol_is_informative(lhs_b)):
                    lhs_dependency = _symbol_matches_with_notation_prefix(
                        lhs_b, rhs_ids_a
                    )
                if not lhs_dependency:
                    lhs_dependency = _lhs_sets_match(clear_lhs_b, rhs_ids_a)
                if lhs_dependency and clear_lhs_b:
                    for k in range(j + 1, i):
                        intervening = {
                            s for s in clear_lhs_ids_map.get(numbers[k], set())
                            if s not in freq_stop
                        }
                        if clear_lhs_b & intervening:
                            lhs_dependency = False
                            break

            if not lhs_dependency and i == j + 1 and same_section:
                lhs_a_set = clear_lhs_ids_map.get(num_a, set())
                lhs_b_set = clear_lhs_ids_map.get(num_b, set())
                formula_shared = (
                    (set(ids_a_filt) & set(ids_b_filt)) |
                    (rhs_ids_a & {
                        s for s in rhs_ids_map.get(num_b, set())
                        if s not in freq_stop and _relation_symbol_is_informative(s)
                    })
                )
                context_a = (
                    (pre_texts.get(num_a) or "") + " " +
                    (post_texts.get(num_a) or "")
                ).strip()
                adjacent_derivation = _has_adjacent_derivation(
                    context_a, formula_shared, lhs_a_set, lhs_b_set
                )

            grade, desc = classify_relation(
                t_sim, j_sim, c_sim, ref_type, shared, lhs_dependency,
                same_section=same_section,
                adjacent_derivation=adjacent_derivation,
                cosine_source=embed_source,
            )

            entry = {"grade": grade}
            if desc:
                entry["description"] = desc
            result[num_a][num_b] = entry

    return result


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
