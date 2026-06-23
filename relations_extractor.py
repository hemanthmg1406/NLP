"""Compute pairwise relations between enumerated equations in one paper.

Five non-generative signals are combined via a monotone decision rule:

  1. Explicit cross-references
     Regex finds '(N)' equation number patterns in the pre/post context of
     each equation. Matched references are substituted with EQREF_N tokens
     and passed to spaCy dependency parsing. The HEAD verb of EQREF_N is
     looked up in a cue-phrase lexicon to assign a relation type label.

  2. Definitional dependency
     If the LHS symbol of equation B appears in the identifier set of equation
     A, then B provides the definition of a quantity used in A. This is a
     strict parent-child relation (e.g. eq 1 contains h_q, eq 2a defines h_q).

  3. Structural similarity (TED)
     MathML expression trees are compared with the Zhang-Shasha algorithm
     (via mathml_tree.tree_edit_distance). High TED similarity => equations
     share the same mathematical form. Requires shared identifiers to grade
     'strong'; otherwise the match is a parallel form (potential).

  4. Symbol overlap (weighted Jaccard)
     Two-tiered Jaccard: 70 % weight on exact normalized key match (hat_H vs
     hat_H), 30 % weight on base-form match (hat_H vs H). This preserves the
     operator/scalar distinction while still capturing shared symbols.

  5. Textual context similarity (TF-IDF cosine)
     Pre/post-equation prose is vectorised with TF-IDF (unigrams + bigrams,
     sublinear tf). High cosine indicates equations share specific rare
     vocabulary in their context. Used only when jaccard > 0; TF-IDF cosine
     alone is excluded because boilerplate prose still produces non-zero overlap
     for unrelated same-section equations.

Monotone decision rule (priority order)
----------------------------------------
  1. explicit_ref_type set                      → strong  (lexicon description)
  2. LHS(B) ∈ identifiers(A)                   → strong  ('defines component')
  3. tree_sim >= TREE_SIM_STRONG
       AND shared identifiers                   → strong  ('equivalent')
  4. tree_sim >= TREE_SIM_STRONG
       AND no shared identifiers                → potential ('parallel form')
  5. jaccard >= JACCARD_POTENTIAL               → potential ('shared symbols …')
  6. tfidf_cosine >= TFIDF_POTENTIAL AND jaccard > 0 → potential ('shared symbols … contextually related')
  7. otherwise                                  → none

All pairs are emitted including 'none' to satisfy the project schema which
requires a relations entry for every other equation in the paper.
"""

import re

import numpy as np
import spacy

from context_extract import _split_sentences
from mathml_tree import mathml_to_tree, tree_edit_distance

# ---------------------------------------------------------------------------
# Signal 5: TF-IDF cosine similarity (replaces neural bi-encoder).
#
# Why TF-IDF instead of a neural embedding model:
#   Neural bi-encoders (specter, math_pretrained_bert) encode each context
#   independently. Equations 1-7 in the same paper share the same pre_text
#   paragraph, so their embeddings cluster near cosine ~ 1.0 regardless of
#   actual relation — the signal is non-discriminating within a paper.
#
#   TF-IDF weights rare paper-specific terms (e.g. "cavity damping kappa",
#   "Bell state amplitude") above shared boilerplate ("where", "Hamiltonian").
#   Two equations whose prose contexts share specific rare terms score high;
#   two equations that merely appear in the same section score low.
#   No model loading, no GPU dependency, deterministic across runs.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Decision thresholds.
# TFIDF_POTENTIAL is lower than the old COSINE_POTENTIAL = 0.75 because
# TF-IDF cosines are naturally smaller than neural cosines (sparse vectors,
# no soft similarity across synonyms). Empirically calibrated on reviewed
# papers: related equation pairs reach 0.25-0.60; unrelated pairs < 0.15.
# ---------------------------------------------------------------------------
TREE_SIM_STRONG   = 0.85
TFIDF_POTENTIAL   = 0.20   # was COSINE_POTENTIAL = 0.75 (neural scale)
JACCARD_POTENTIAL = 0.40

# ---------------------------------------------------------------------------
# Cue-phrase lexicon: spaCy verb lemma → canonical relation type description.
# Covers the six types named in the project spec plus common physics phrasing.
# ---------------------------------------------------------------------------
_CUE_LEXICON = {
    # Substitution
    "substitute": "substitution",
    "plug":       "substitution",
    "insert":     "substitution",
    # Combination / derivation
    "combine":    "combination",
    "derive":     "derivation",
    "obtain":     "derivation",
    "follow":     "derivation",
    "yield":      "derivation",
    "produce":    "derivation",
    "give":       "derivation",
    "get":        "derivation",
    "result":     "derivation",
    # Equivalent forms
    "simplify":   "equivalent",
    "equal":      "equivalent",
    "rewrite":    "equivalent",
    "express":    "equivalent",
    # Special case / generalisation
    "reduce":     "special case",
    "recover":    "special case",
    "specialize": "special case",
    "generalize": "generalisation",
    "extend":     "generalisation",
    # Negation
    "negate":     "negation",
    "violate":    "negation",
    "contradict": "negation",
    # Operational dependency: fit/measurement output feeds into formula
    # "fitted with (1) to extract C" → Eq 1 operationally produces C for Eq 2
    "fit":        "operational dependency",
    "extract":    "operational dependency",
    "estimate":   "operational dependency",
    "measure":    "operational dependency",
    "determine":  "operational dependency",
    "calibrate":  "operational dependency",
    # Limit
    "limit":      "limit",
    "approach":   "limit",
    "take":       "limit",
    # Aggregate / continuum transformations (not verb-based — injected directly
    # by the limit-transformation detector in find_explicit_refs).
    "replace":    "limit transformation",
    "discretize": "limit transformation",
    "coarsen":    "limit transformation",
}

# Decorator prefixes mirroring build_json._DECORATOR_PREFIXES.
# mathsf added: physics papers use \mathsf{G} for Green's tensors, \mathsf{H} for
# Hamiltonians in operator notation. Without it, LHS extraction misses these symbols
# and the definitional dependency signal fails for tensor-valued equations.
_DECORATOR_PREFIXES = frozenset({
    "hat", "bar", "tilde", "vec", "widehat", "widetilde", "overline",
    "bm", "boldsymbol", "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf",
    "mathsf",
})

# Regex matching equation number patterns like (1), (3a), Eq. 5, equation 2.
# Two alternative forms (joined with |) so both are detected in one pass:
#
#   Form A — "Eq(s). N" or "equation(s) N": prefix is mandatory; parens around
#   the number are optional. Catches "Eq. 5", "Eq. (5)", "Eqs. 2 and 3",
#   "equation 4" — patterns the old regex missed entirely.
#
#   Form B — bare "(N)": parens are mandatory; no prefix required. Catches the
#   most common arXiv style: "substituting (3) into (5)".
#
# When both groups are present only one will be non-None; _substitute_eqrefs
# reads whichever group matched via `m.group(1) or m.group(2)`.
_EQNUM_RE = re.compile(
    r"(?:Eqs?\.?|eqs?\.?|[Ee]quations?)\s*\(?(\d+(?:\.\d+)?[a-z]?)\)?"
    r"|"
    r"\((\d+(?:\.\d+)?[a-z]?)\)"
)

# Matches a leading LaTeX command or plain identifier on the LHS of an equation.
# mathsf included alongside other decorators so Green's tensors (\mathsf{G}) are
# recognised as defined symbols.
_LHS_RE = re.compile(
    r"^\s*(?:\\(?:hat|bar|tilde|vec|widehat|widetilde|overline|bm|boldsymbol"
    r"|mathcal|mathbb|mathscr|mathfrak|mathbf|mathsf)\{([^}]+)\}|([A-Za-z][A-Za-z0-9]*"
    r"(?:_\{[^}]+\}|_[A-Za-z0-9])?))"
)

# ---------------------------------------------------------------------------
# Lazy-loaded resources (loaded once per process).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Signal 2 helper: LHS symbol extraction
# ---------------------------------------------------------------------------

def extract_lhs_symbol(latex):
    """Extract the primary symbol defined on the left-hand side of an equation.

    Looks at everything before the first '=' and matches the outermost LaTeX
    identifier or decorated command. The result is normalized to the same
    no-backslash key form used by symbols_extract (e.g. 'hat_H', 'h_q').

    This is used for the definitional dependency signal: if LHS(B) appears in
    identifiers(A), equation B defines a quantity used in A, which is a strong
    directional relation (A depends on B).

    Parameters
    ----------
    latex : str
        Raw LaTeX string of the equation.

    Returns
    -------
    str or None
        Normalized identifier string, or None when no clear LHS is found
        (e.g. commutator equations whose LHS is a brace expression).
    """
    # Take only the part before the first '=' sign.
    lhs_raw = latex.split("=")[0] if "=" in latex else latex
    # Strip common wrappers that surround the whole LHS (anticommutators, sets).
    lhs_raw = re.sub(r"^[\s\\{}\[\](|]+", "", lhs_raw).strip()

    m = _LHS_RE.match(lhs_raw)
    if not m:
        return None

    # Group 1 catches decorated forms like \hat{H} → 'H'; group 2 catches plain.
    inner = (m.group(1) or m.group(2) or "").strip()
    if not inner:
        return None

    # Guard: if the raw LHS has '(' immediately after the matched symbol, this
    # is a function APPLICATION (e.g. H(t;θ) = ...) not a scalar definition.
    # Returning None prevents conflating H and H(t;θ) as the same object —
    # the source of the (1,3) false positive where generic H matched parameterized H.
    remaining = lhs_raw[m.end():].lstrip()
    if remaining.startswith("("):
        return None

    # Determine decorator prefix from the original lhs_raw.
    for prefix in ("hat", "bar", "tilde", "vec", "widehat", "widetilde",
                   "overline", "bm", "boldsymbol",
                   "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf"):
        if re.match(rf"^\\{prefix}\{{", lhs_raw.strip()):
            # Normalize inner: strip braces, subscript braces.
            key = re.sub(r"[\\{}_]", "", inner).strip()
            return f"{prefix}_{key}" if key else None

    # Plain symbol — strip subscript braces for normalization.
    key = re.sub(r"\{([^}]+)\}", r"\1", inner)
    key = re.sub(r"[\\{}]", "", key).strip()
    return key if key else None


# ---------------------------------------------------------------------------
# Signal 1: Explicit cross-reference detection
# ---------------------------------------------------------------------------

def _substitute_eqrefs(text, valid_numbers):
    """Replace '(N)' equation number patterns with EQREF_N placeholder tokens.

    Only replaces numbers that correspond to actual equations in the current
    paper to avoid false matches on citation numbers like (2024) or (Fig. 3).

    Parameters
    ----------
    text : str
        Pre/post context text, already cleaned.
    valid_numbers : set of str
        Printed equation numbers present in this paper (e.g. {'1', '2', '1a'}).

    Returns
    -------
    tuple[str, dict]
        (substituted_text, {eq_number: EQREF_token_str})
    """
    eqref_map = {}

    def _replace(m):
        # Group 1 fires for Form A ("Eq. N"); group 2 fires for Form B ("(N)").
        num = m.group(1) or m.group(2)
        if num in valid_numbers:
            # Use underscores so spaCy treats EQREF_3 as a single token.
            token = f"EQREF_{num.replace('.', '_')}"
            eqref_map[num] = token
            return token
        return m.group(0)  # leave citation numbers untouched

    substituted = _EQNUM_RE.sub(_replace, text)
    return substituted, eqref_map


def _extract_cue_verb(text, eqref_token):
    """Run spaCy dependency parsing to find the HEAD verb of an EQREF token.

    Finds the sentence containing the EQREF token, parses its dependency tree,
    and traverses up from the token's position to find the nearest governing
    verb. The verb lemma is then looked up in _CUE_LEXICON.

    Parameters
    ----------
    text : str
        Full context text with EQREF tokens substituted in.
    eqref_token : str
        The specific EQREF_N string to locate.

    Returns
    -------
    str
        Relation type from _CUE_LEXICON, or 'reference' when no known cue
        verb is found (still marks the pair as explicitly referenced).
    """
    nlp = _get_nlp()
    if nlp is None:
        return "reference"  # spaCy unavailable: fall back to plain reference label

    # Narrow to the sentence containing this EQREF token.
    sentences = _split_sentences(text)
    target_sent = next((s for s in sentences if eqref_token in s), text)

    doc = nlp(target_sent)
    for token in doc:
        if token.text != eqref_token:
            continue
        # Traverse up dependency tree (max 5 hops) to find governing verb.
        head = token
        for _ in range(5):
            if head.pos_ == "VERB":
                return _CUE_LEXICON.get(head.lemma_.lower(), "reference")
            if head.head is head:
                break
            head = head.head
        # If no verb found in path, check whole sentence for the nearest verb.
        for tok in doc:
            if tok.pos_ == "VERB":
                return _CUE_LEXICON.get(tok.lemma_.lower(), "reference")
        return "reference"

    return "reference"


# Cue phrases signalling a discrete-to-continuum or similar limit transformation.
# When these appear near an equation, the equation may be a macroscopic rewrite
# of a microscopic sum — a relation that TED and jaccard cannot detect because
# symbols and tree structure change completely across the transformation.
# Quantum zero-point / vacuum-coupling cues: when these appear near an equation,
# that equation likely represents the quantum version of a classical expression.
# Pairs a ZPF-context equation against equations that contain classical δ-terms
# (δω, δΦ, δx) — the canonical classical-to-quantum substitution pattern in
# optomechanics and electromechanics papers.
_ZPF_CUES = re.compile(
    r'\b(?:zero[- ]?point\s+(?:fluctuation|motion|amplitude|flux)|'
    r'vacuum\s+(?:fluctuation|coupling|noise)|'
    r'zpf\b|x_\{?zpf\}?|'
    r'single\s+(?:phonon|photon|magnon)\s+coupling|'
    r'quantize|second\s+quantiz)',
    re.I,
)

# Classical differential/perturbation term in LaTeX.
# Matches \delta followed by: a plain letter, a brace group, OR another LaTeX
# command (\omega, \Phi, \Phi_s etc.). The original pattern missed \delta\omega
# and \delta\Phi because they start with backslash, not a letter.
_CLASSICAL_DELTA_RE = re.compile(
    r'\\delta\s*(?:[A-Za-z]|\{[^}]+\}|\\[A-Za-z]+)'
)

# ZPF indicator in the equation's own LaTeX (not context, which is shared across
# all equations in the same paragraph and therefore non-discriminating).
# Matches zero-point fluctuation symbols: \zeta, x_zpf, \Phi_zpf, or the
# string "zpf" appearing in a subscript or mathrm group.
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

# Detects aggregate operators in raw LaTeX: \sum or \int as evidence that an
# equation is a discrete sum or a continuum integral respectively.
_SUM_RE  = re.compile(r'\\sum\b')
_INT_RE  = re.compile(r'\\int\b')


def find_explicit_refs(source_num, context_text, valid_numbers,
                       source_latex=None, latex_map=None):
    """Detect all equation cross-references in one equation's context text.

    Two detection passes:
    1. Regex + spaCy: finds '(N)' patterns and classifies the governing cue verb.
    2. Limit-transformation pass: if context contains discrete-to-continuum cue
       phrases AND one of the referenced equations is an aggregate (sum/integral)
       of the opposite type, marks the relation as 'limit transformation'. This
       catches microscopic→macroscopic rewrites where symbols change completely
       and TED / jaccard produce no signal.

    Parameters
    ----------
    source_num : str
        The equation number whose context we are scanning.
    context_text : str
        Combined pre_text + post_text for the source equation.
    valid_numbers : set of str
        All printed equation numbers in this paper.
    source_latex : str or None
        LaTeX of the source equation — used to detect its aggregate type.
    latex_map : dict or None
        {eq_number: latex_str} for all equations in the paper — used to check
        whether a referenced equation uses the complementary aggregate operator.

    Returns
    -------
    dict
        {target_num: relation_type_str} for all referenced equations found.
    """
    others = valid_numbers - {source_num}
    if not context_text or not others:
        return {}

    substituted, eqref_map = _substitute_eqrefs(context_text, others)
    refs = {}
    for target_num, token in eqref_map.items():
        refs[target_num] = _extract_cue_verb(substituted, token)

    # Second pass: limit transformation detection.
    # Only runs when cue phrases are present and latex is available for both sides.
    if source_latex is not None and latex_map and _LIMIT_CUES.search(context_text):
        src_is_sum = bool(_SUM_RE.search(source_latex))
        src_is_int = bool(_INT_RE.search(source_latex))
        for num in others:
            tgt_latex = latex_map.get(num, "")
            tgt_is_sum = bool(_SUM_RE.search(tgt_latex))
            tgt_is_int = bool(_INT_RE.search(tgt_latex))
            # Flag when source and target are complementary aggregate types:
            # sum↔integral is the canonical discrete-to-continuum transformation.
            if (src_is_sum and tgt_is_int) or (src_is_int and tgt_is_sum):
                # Only upgrade: do not overwrite an already-detected explicit ref.
                if num not in refs:
                    refs[num] = "limit transformation"

    return refs


# ---------------------------------------------------------------------------
# Signal 2: Weighted two-tiered Jaccard
# ---------------------------------------------------------------------------

def _split_key(key):
    """Split 'hat_H' into ('hat', 'H'); plain 'H' into (None, 'H').

    Parameters
    ----------
    key : str
        Normalized identifier key as produced by symbols_extract.

    Returns
    -------
    tuple[str or None, str]
        (prefix_or_None, base_form)
    """
    if "_" in key:
        prefix, base = key.split("_", 1)
        if prefix in _DECORATOR_PREFIXES:
            return prefix, base
    return None, key


def weighted_jaccard(ids_a, ids_b):
    """Compute weighted two-tiered Jaccard similarity between identifier sets.

    Two-tiered to preserve the operator/scalar distinction in quantum physics:
    hat_H (operator) and H (scalar) are different physical quantities, so an
    exact match is worth more than a base-form match.

        score = 0.7 * J_exact + 0.3 * J_base

    where J_exact is computed over full normalized keys and J_base over base
    forms only (decorator prefix stripped).

    Parameters
    ----------
    ids_a : list of str
        Normalized identifier keys for equation A.
    ids_b : list of str

    Returns
    -------
    float
        Weighted Jaccard in [0, 1]. Returns 0.0 when both sets are empty.
    """
    set_a = set(ids_a)
    set_b = set(ids_b)

    if not set_a and not set_b:
        return 0.0

    # Exact Jaccard over full normalized keys (e.g. 'hat_H' vs 'hat_H').
    exact_union = len(set_a | set_b)
    j_exact = len(set_a & set_b) / exact_union if exact_union else 0.0

    # Base-form Jaccard: strip decorator prefix before comparing.
    bases_a = {_split_key(k)[1] for k in set_a}
    bases_b = {_split_key(k)[1] for k in set_b}
    base_union = len(bases_a | bases_b)
    j_base = len(bases_a & bases_b) / base_union if base_union else 0.0

    return 0.7 * j_exact + 0.3 * j_base


# ---------------------------------------------------------------------------
# Signal 5: TF-IDF cosine similarity on prose context
# ---------------------------------------------------------------------------

def build_tfidf_matrix(texts):
    """Fit TF-IDF on a small corpus of equation prose contexts and return
    an L2-normalised dense matrix (one row per text).

    Parameters
    ----------
    texts : list of str
        One combined prose string (pre_text + post_text) per equation.
        LaTeX is excluded — structure is handled by Signal 3 (TED).

    Returns
    -------
    np.ndarray
        Shape (len(texts), vocab_size), L2-normalised so that row-dot-product
        equals cosine similarity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    # sublinear_tf: log(1 + tf) dampens high-frequency boilerplate terms.
    # ngram_range (1,2): bigrams capture "density matrix", "decay rate" etc.
    # No sklearn stop-word list: physics prose reuses "is", "the", "where"
    # in definitional sentences — removing them would destroy the signal.
    # min_df=1: corpus is tiny (≤7 docs), can't afford minimum-frequency filtering.
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        token_pattern=r"(?u)\b\w[\w.]*\b",  # allow dots in "H.c.", "i.e."
    )
    sparse = vec.fit_transform(texts)
    return normalize(sparse, norm="l2").toarray()


def tfidf_cosine(matrix, i, j):
    """Cosine similarity between rows i and j of an L2-normalised TF-IDF matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Output of build_tfidf_matrix.
    i, j : int
        Row indices.

    Returns
    -------
    float
    """
    return float(np.dot(matrix[i], matrix[j]))


# ---------------------------------------------------------------------------
# Decision function
# ---------------------------------------------------------------------------

def classify_relation(tree_sim, jaccard_sim, cosine_sim,
                      explicit_ref_type=None, shared_ids=None,
                      lhs_defines=False):
    """Monotone decision rule: assign a relation grade and description.

    Priority (highest to lowest):
      1. Explicit cross-reference → strong, lexicon description.
      2. Definitional dependency (LHS(B) ∈ identifiers(A)) → strong,
         'defines component'. This catches the pattern where equation B gives
         the explicit definition of a scalar/operator used in equation A.
      3. TED ≥ TREE_SIM_STRONG AND shared identifiers → strong, 'equivalent'.
         Structural match with overlapping symbols means the same math object.
      4. TED ≥ TREE_SIM_STRONG AND no shared identifiers → potential,
         'parallel form'. Same tree structure but different LHS symbols means
         the two equations are parallel constructions (e.g. one-electron vs
         two-electron integral), not the same object.
      5. Jaccard ≥ JACCARD_POTENTIAL → potential, 'shared symbols (X, Y)'.
      6. TF-IDF cosine ≥ TFIDF_POTENTIAL AND jaccard > 0 → potential,
         'shared symbols (X, Y), contextually related'. TF-IDF cosine alone
         (jaccard=0) is excluded: two unrelated equations in the same section
         can share enough prose boilerplate to exceed the threshold.
      7. Otherwise → none.

    Parameters
    ----------
    tree_sim : float
    jaccard_sim : float
    cosine_sim : float
    explicit_ref_type : str or None
        Relation type from cue-phrase lexicon when an explicit ref was found.
    shared_ids : set of str or None
        Exact identifier intersection (ids_a ∩ ids_b).
    lhs_defines : bool
        True when the LHS symbol of the target equation is found in the source
        equation's identifier set — i.e. target defines a component of source.

    Returns
    -------
    tuple[str, str]
        (grade, description). description is empty string for 'none'.
    """
    if shared_ids is None:
        shared_ids = set()

    # Priority 1: explicit textual cross-reference.
    if explicit_ref_type is not None:
        return "strong", explicit_ref_type

    # Priority 2: definitional dependency.
    # B defines a quantity that appears in A — direct parent-child relation.
    if lhs_defines:
        return "strong", "defines component"

    # Priority 3 / 4: structural similarity via TED.
    if tree_sim >= TREE_SIM_STRONG:
        if shared_ids:
            # Shared symbols + identical structure → same mathematical object.
            sym_str = ", ".join(sorted(shared_ids)[:4])
            return "strong", f"equivalent — shared form and symbols ({sym_str})"
        else:
            # Identical structure but different symbols → parallel construction.
            # E.g. one-electron integral and two-electron integral: same AST,
            # different LHS, different physical meaning.
            return "potential", f"parallel form [tree_sim={tree_sim:.2f}]"

    # Priority 5: symbol overlap alone.
    if jaccard_sim >= JACCARD_POTENTIAL:
        sym_str = ", ".join(sorted(shared_ids)[:4]) if shared_ids else ""
        label = f"shared symbols ({sym_str})" if sym_str else "overlapping notation"
        return "potential", f"{label} [j={jaccard_sim:.2f}]"

    # Priority 6: TF-IDF cosine + non-zero jaccard.
    # TF-IDF cosine alone is not used (jaccard > 0 required): two equations in
    # the same section can share enough prose to reach the threshold even when
    # mathematically unrelated. Non-zero jaccard confirms shared notation.
    if cosine_sim >= TFIDF_POTENTIAL and jaccard_sim > 0:
        sym_str = ", ".join(sorted(shared_ids)[:4]) if shared_ids else ""
        label = (f"shared symbols ({sym_str}), contextually related"
                 if sym_str else "contextually related")
        return "potential", f"{label} [tfidf={cosine_sim:.2f}, j={jaccard_sim:.2f}]"

    return "none", ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_relations(equations, table_index, tree, pre_texts,
                    post_texts, identifiers_map):
    """Compute all pairwise relations for the equations in one paper.

    Called once per paper after all per-equation signals have been extracted.
    Reuses pre_texts, post_texts, and identifiers already computed in
    build_json.process_paper() to avoid redundant DOM traversal.

    Parameters
    ----------
    equations : list of dict
        [{number, latex, eq_id}, ...] — same slice passed to process_paper.
    table_index : dict
        {eq_id: table_element} from build_json._build_table_index.
    tree : lxml tree
        Full document tree (used only for mathml_to_tree via table nodes).
    pre_texts : dict
        {eq_number: str} — pre-equation prose for each equation.
    post_texts : dict
        {eq_number: str} — post-equation 'where...' clause for each equation.
    identifiers_map : dict
        {eq_number: list of str} — normalized identifier keys per equation.

    Returns
    -------
    dict
        {eq_number: {other_number: {grade, description}}}
        Every equation has an entry for every other equation (including grade
        'none') to satisfy the project schema.
    """
    numbers = [eq["number"] for eq in equations]
    n = len(numbers)

    # Initialize result with empty dicts for all equations.
    result = {num: {} for num in numbers}

    if n < 2:
        return result

    valid_numbers = set(numbers)

    # --- Signal 1: explicit cross-references + limit transformation detection ---
    # Build latex_map once so find_explicit_refs can check aggregate operator types.
    latex_map = {eq["number"]: eq["latex"] for eq in equations}

    explicit_refs = {}   # (source_num, target_num) → relation_type
    for eq in equations:
        num = eq["number"]
        context = (
            (pre_texts.get(num) or "") + " " +
            (post_texts.get(num) or "")
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

    # --- Signal 2: definitional dependency ---
    # Build {eq_number: lhs_symbol} by extracting the LHS of each equation.
    # lhs_map[num] = the symbol that equation 'num' defines on its left side.
    lhs_map = {}
    for eq in equations:
        sym = extract_lhs_symbol(eq["latex"])
        if sym:
            lhs_map[eq["number"]] = sym

    # --- Signal 3: MathML expression trees ---
    math_trees = {}
    for eq in equations:
        table = table_index.get(eq["eq_id"])
        math_trees[eq["number"]] = mathml_to_tree(table)

    # --- Signal 5: TF-IDF cosine on prose context ---
    # Use pre_text + post_text only (no LaTeX): LaTeX structure is already
    # covered by Signal 3 (TED). TF-IDF weights paper-specific rare terms
    # above boilerplate, so related equations that share specific vocabulary
    # ("cavity damping", "Bell state amplitude") score distinctly higher than
    # unrelated equations that merely appear in the same section.
    prose_texts = [
        ((pre_texts.get(num) or "") + " " + (post_texts.get(num) or "")).strip()
        for num in numbers
    ]
    tfidf_mat = build_tfidf_matrix(prose_texts)
    # Map number → row index for O(1) access inside the pairwise loop.
    num_idx = {num: i for i, num in enumerate(numbers)}

    # --- Frequency-based jaccard stop-list ---
    # Symbols appearing in more than half the paper's equations are high-frequency
    # "noise" identifiers (e.g. θ as a global parameter, ℏ in every Hamiltonian).
    # They inflate jaccard across unrelated pairs, producing false potential grades.
    # Excluding them makes jaccard discriminate within-paper relations more precisely.
    from collections import Counter
    sym_freq = Counter()
    for num in numbers:
        for sym in (identifiers_map.get(num) or []):
            sym_freq[sym] += 1
    n_eqs = len(numbers)
    # Threshold: >70% AND minimum 3 occurrences.
    # The minimum-3 guard prevents the filter from activating on tiny papers
    # (n ≤ 3): with n=2, every shared symbol hits 100% and would be filtered,
    # eliminating the Jaccard signal entirely. Requiring cnt >= 3 means
    # freq_stop is inactive when fewer than 3 equations are present.
    freq_stop = {
        sym for sym, cnt in sym_freq.items()
        if cnt > 0.70 * n_eqs and cnt >= 3
    }
    if freq_stop:
        print(f"  jaccard stop-list ({len(freq_stop)} symbols): "
              f"{sorted(freq_stop)}")

    # --- Build pairwise relations ---
    # Also track which (A, B) pairs fired strong via lhs_defines so we can
    # immediately set the reverse direction to strong as well.
    lhs_strong_pairs = set()

    for i, num_a in enumerate(numbers):
        ids_a = identifiers_map.get(num_a) or []

        for j, num_b in enumerate(numbers):
            if i == j:
                continue

            ids_b   = identifiers_map.get(num_b) or []
            ref_key = (num_a, num_b)

            # Signal 1: explicit reference from A's context pointing to B.
            ref_type = explicit_refs.get(ref_key)

            # Signal 2: definitional dependency.
            # Does equation B define a symbol that appears in equation A?
            # lhs_map[num_b] is the symbol B assigns; check it against A's ids.
            #
            # Two-level match to handle normalization inconsistencies between
            # extract_lhs_symbol (which preserves decorator prefix as "mathcal_F")
            # and extract_identifiers (which may strip notation-style prefixes
            # and return just "F"). We allow base-form matching ONLY for
            # notation-style decorators (mathcal, mathbb, mathscr, mathfrak,
            # mathbf, mathsf) where the decorated and plain forms refer to the
            # same physical quantity. We do NOT extend this to hat/bar/vec/tilde
            # because those distinguish operators from scalars (hat_H ≠ H).
            _NOTATION_PREFIXES = frozenset({
                "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf", "mathsf"
            })
            lhs_b = lhs_map.get(num_b)
            lhs_defines = False
            if lhs_b:
                ids_a_set = set(ids_a)
                if lhs_b in ids_a_set:
                    lhs_defines = True
                else:
                    # Try base-form match for notation-style decorators only.
                    parts = lhs_b.split("_", 1)
                    if len(parts) == 2 and parts[0] in _NOTATION_PREFIXES:
                        lhs_defines = parts[1] in ids_a_set

            # Signal 3: structural similarity.
            t_sim = tree_edit_distance(math_trees[num_a], math_trees[num_b])

            # Signal 4: weighted Jaccard over identifier sets.
            # Use freq-filtered identifier sets: high-frequency symbols (appearing
            # in >50% of the paper's equations) are excluded from both sets before
            # the Jaccard calculation so they don't inflate unrelated pair scores.
            ids_a_filt = [s for s in ids_a if s not in freq_stop]
            ids_b_filt = [s for s in ids_b if s not in freq_stop]
            j_sim = weighted_jaccard(ids_a_filt, ids_b_filt)

            # Signal 5: TF-IDF cosine similarity of prose contexts.
            c_sim = tfidf_cosine(tfidf_mat, num_idx[num_a], num_idx[num_b])

            # Exact symbol intersection over filtered sets — used to name shared
            # identifiers in the potential description so the grader sees a
            # concrete semantic link. Uses filtered sets so freq-noise symbols
            # don't appear in the description either.
            shared = set(ids_a_filt) & set(ids_b_filt)

            grade, desc = classify_relation(
                t_sim, j_sim, c_sim, ref_type, shared, lhs_defines
            )

            # Track lhs_defines strong pairs for bidirectional upgrade below.
            if grade == "strong" and lhs_defines and ref_type is None:
                lhs_strong_pairs.add((num_a, num_b))

            entry = {"grade": grade}
            if desc:
                entry["description"] = desc
            result[num_a][num_b] = entry

    # Post-processing: classical-to-quantum (ZPF) derivation detection.
    # Pattern: classical equations contain \delta-prefixed symbols (δω, δΦ, δx);
    # the quantum equation evaluates that classical expression at one zero-point
    # fluctuation, so its LaTeX contains ζ_zpf, x_zpf, or similar ZPF symbols.
    # Symbols change completely across this substitution — TED and Jaccard fail.
    #
    # Detection uses the equation's OWN LaTeX, NOT context text. Context is
    # shared across all equations in the same paragraph, making it non-
    # discriminating when multiple equations appear in the same derivation block.
    zpf_nums = set()
    classical_delta_nums = set()
    for eq in equations:
        num = eq["number"]
        lat = eq["latex"]
        if _ZPF_LATEX_RE.search(lat):
            zpf_nums.add(num)
        if _CLASSICAL_DELTA_RE.search(lat):
            classical_delta_nums.add(num)
    if zpf_nums and classical_delta_nums:
        print(f"  ZPF equations (by LaTeX): {sorted(zpf_nums)}, "
              f"classical-δ equations: {sorted(classical_delta_nums)}")
    for q_num in zpf_nums:
        for c_num in classical_delta_nums:
            if c_num == q_num:
                continue
            for src, tgt, label in [
                (c_num, q_num, "classical-to-quantum derivation"),
                (q_num, c_num, "quantum coupling derived from classical expression"),
            ]:
                current = result.get(src, {}).get(tgt, {}).get("grade", "none")
                if current not in ("strong",):
                    result[src][tgt] = {
                        "grade": "strong",
                        "description": label,
                    }

    # Post-processing: bidirectional definitional dependency.
    # When (A, B) = strong "defines component" (B defines a symbol used in A),
    # the reverse direction (B, A) = "component used in derivation" is equally
    # informative: from B's perspective, A is the equation that depends on what
    # B defines. Only upgrade if the current grade is lower than strong.
    for num_a, num_b in lhs_strong_pairs:
        current = result.get(num_b, {}).get(num_a, {}).get("grade", "none")
        if current != "strong":
            result[num_b][num_a] = {
                "grade": "strong",
                "description": "component used in derivation",
            }

    # Post-processing: general bidirectionalization of all strong pairs.
    # If A→B is strong for any reason (explicit cross-reference, TED, Jaccard),
    # the reverse B→A should also be strong. A mathematical dependency between
    # two equations is not one-directional: the equation that is referenced or
    # used is just as related to the referencing equation as vice versa.
    # Only upgrade — never downgrade an existing strong entry.
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

    # Post-processing: full DAG reachability with λ-decay path scoring.
    # Replaces the earlier depth-2 cutoff with complete BFS over the strong-edge
    # DAG. Each hop multiplies confidence by λ=0.5 (max-product over all paths).
    # score ≥ 0.5 is a direct strong edge (already set above); score in [0.2, 0.5)
    # → upgrade 'none' to 'potential'; below 0.2 → leave as 'none'.
    # Grounded in the typed reachability approach of the MDGD research survey.
    result = _dag_reachability(result, numbers)

    return result


# Decay factor per hop in the DAG reachability scorer.
# 0.5 means: direct edge = 1.0, 1-hop indirect = 0.5, 2-hop = 0.25, 3-hop = 0.125.
# Score threshold for upgrading 'none' → 'potential': 0.2 (covers up to ~2 hops).
_LAMBDA        = 0.5
_REACH_THRESH  = 0.2


def _dag_reachability(relations, numbers):
    """Full BFS reachability on the strong-edge DAG with λ-decay path scoring.

    After pairwise signals have been applied, this post-processing step finds
    all equation pairs connected by a chain of strong edges (any length) and
    upgrades pairs graded 'none' to 'potential' when the max-product path score
    exceeds _REACH_THRESH.

    Path score formula (max-product over all paths A→…→C):
        s(P) = λ^(|P|-1)   (each intermediate hop multiplies by λ)
        s(A,C) = max over all paths P from A to C of s(P)

    The shortest path has the highest score because λ < 1. This penalizes
    longer indirect chains, preventing runaway false positives in dense graphs.

    Only 'none' pairs are upgraded — existing 'potential' or 'strong' grades
    are never overwritten. The description records the best intermediate node
    and the path score so the grader can trace the inference.

    Parameters
    ----------
    relations : dict
        {num_a: {num_b: {grade, description}}} as built by build_relations.
    numbers : list of str
        Equation numbers in document order.

    Returns
    -------
    dict
        Same structure with reachable 'none' pairs upgraded.
    """
    # Build adjacency on strong edges only.
    # Potential edges are used for candidate generation in the pairwise pass
    # but must not propagate in the reachability DAG — they are too noisy.
    strong_adj = {
        num: {
            other for other, entry in relations.get(num, {}).items()
            if entry.get("grade") == "strong"
        }
        for num in numbers
    }

    for src in numbers:
        # BFS from src over strong edges; track (node, score, best_via) per visit.
        # visited maps node → best score seen so far.
        visited = {src: 1.0}
        queue   = [(src, 1.0, None)]   # (node, score_to_here, first_hop_after_src)

        while queue:
            node, score, via = queue.pop(0)
            hop_score = score * _LAMBDA

            for nbr in strong_adj.get(node, set()):
                if nbr == src:
                    continue
                if hop_score <= visited.get(nbr, 0.0):
                    continue   # already reached nbr via a better path

                visited[nbr] = hop_score
                # Track the first intermediate node after src for the description.
                queue.append((nbr, hop_score, via if via is not None else nbr))

        # Upgrade 'none' pairs whose best path score exceeds the threshold.
        for tgt, path_score in visited.items():
            if tgt == src:
                continue
            if path_score >= 1.0:
                continue   # direct strong edge — already graded correctly
            if path_score < _REACH_THRESH:
                continue

            current_grade = relations.get(src, {}).get(tgt, {}).get("grade")
            if current_grade == "none":
                # Recover the via-node from the BFS queue result.
                # Re-derive: first strong neighbor of src whose subtree reaches tgt.
                via_node = next(
                    (v for v in strong_adj.get(src, set())
                     if v != tgt and tgt in visited and
                     relations.get(src, {}).get(v, {}).get("grade") == "strong"),
                    None,
                )
                via_str = f" via ({via_node})" if via_node else ""
                relations[src][tgt] = {
                    "grade": "potential",
                    "description": (
                        f"indirect dependency{via_str} "
                        f"[path_score={path_score:.2f}]"
                    ),
                }

    return relations
