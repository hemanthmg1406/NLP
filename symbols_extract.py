"""Extract equation symbols and simple text definitions from cached arXiv HTML.

This is Pipeline A for Stage 4. It is deterministic and rule-based:
1. enumerate identifiers from MathML, with a small LaTeX fallback,
2. search the cleaned equation context for definition patterns.

No network calls. No generated text. No equation LaTeX is modified.
"""

import re
from pathlib import Path

from lxml import html as lxml_html
from pylatexenc.latexwalker import (LatexWalker, LatexMacroNode,
                                     LatexCharsNode, LatexGroupNode,
                                     LatexEnvironmentNode)

from context_extract import get_contexts, _clean_para, _split_sentences
from review_equations import extract_equations

# Penn Treebank POS tags that constitute a simple noun phrase.
# Adapted from ScholarPhi (Allen AI) search_symbol_nickname.
_NP_TAGS = frozenset({"DT", "JJ", "NN", "NNS", "NNP", "NNPS", "VBG", "VBN"})

# Lazy-loaded spaCy model.  Populated on first call to _get_nlp().
# False = unavailable sentinel so we do not retry after a failed load.
_NLP = None


def _get_nlp():
    """Return a loaded spaCy en_core_web_sm model, or None if unavailable.

    Loading is deferred to the first call so the module imports cleanly on
    machines where spaCy is not installed.  On subsequent calls the cached
    instance is returned without re-loading.

    Returns
    -------
    spacy.Language or None
    """
    global _NLP
    if _NLP is not None:
        return _NLP if _NLP is not False else None
    try:
        import spacy  # noqa: PLC0415 — deferred intentionally
        _NLP = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        _NLP = False
    return _NLP if _NLP is not False else None


CACHE_DIR = Path("cache")

GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi",
    "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi",
    "Sigma", "Upsilon", "Phi", "Psi", "Omega"
}

DECORATORS = {
    "hat", "widehat", "bar", "overline", "tilde", "vec", "bm", "boldsymbol",
    "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf"
}

COMMAND_STOP = {
    "begin", "end", "left", "right", "middle", "frac", "dfrac", "tfrac", "sqrt",
    "sum", "prod", "int", "oint", "lim", "arg", "operatorname", "mathrm",
    "mathsf", "text", "textrm", "displaystyle", "scriptstyle", "quad", "qquad",
    "cdot", "times", "otimes", "oplus", "leq", "geq", "neq", "approx", "sim",
    "simeq", "equiv", "in", "to", "rightarrow", "leftarrow", "mapsto", "pm",
    "mp", "dagger", "prime", "rangle", "langle", "ket", "bra", "braket",
    "innerproduct", "outerproduct", "expectationvalue", "norm", "abs",
    "absolutevalue", "coloneqq", "eqqcolon", "underbrace", "overbrace",
    # delimiter commands — these are fence/norm symbols, never physics identifiers
    "lVert", "rVert", "lvert", "rvert", "Vert", "vert",
    "lfloor", "rfloor", "lceil", "rceil", "lbrace", "rbrace",
    # partial derivative is an operator, not a physics variable
    "partial",
}

IDENTIFIER_STOP = {
    "d", "e", "i", "j", "k", "n", "m", "l", "x", "y", "z", "t",
    "o",  # Landau little-o notation; never a standalone physics symbol
    "sin", "cos", "tan", "log", "ln", "exp", "tr", "Tr", "trace", "Pr",
    "Re", "Im", "det", "rank", "dim", "mod", "sup", "inf", "limsup",
    "liminf", "argmin", "argmax", "Perm", "Haf", "min", "max", "pi"
}

# LaTeX operators whose subscript is a bound (index) variable, not a physics symbol.
# Detected structurally via AST walk — no hardcoded variable names needed.
_BOUND_OPS = {
    "sum", "prod", "int", "oint", "iint", "iiint", "iiiint", "idotsint",
    "lim", "limsup", "liminf",
    "sup", "inf", "max", "min", "argmin", "argmax",
    "bigcup", "bigcap", "bigvee", "bigwedge",
    "forall", "exists",
}


def _all_identifiers_in_group(group_node):
    """Return all alphabetic bound-variable identifiers from a subscript group.

    Handles multi-index subscripts like ``{a,b,c}`` or ``{i=0}^{N}``.
    Each comma-separated token is inspected for a leading alpha identifier.
    Macro nodes (e.g. ``\\rho``) are included directly.

    Parameters
    ----------
    group_node : LatexGroupNode
        The brace group immediately after an underscore in a bound operator
        context, e.g. ``{j=0}`` or ``{a,b}``.

    Returns
    -------
    set of str
        All leading identifiers found, e.g. ``{'a', 'b'}`` for ``{a,b}``.
    """
    found = set()
    # collect raw char text and macro names in one pass
    raw_chars = []
    for node in group_node.nodelist:
        if isinstance(node, LatexCharsNode):
            raw_chars.append(node.chars)
        elif isinstance(node, LatexMacroNode):
            # bare macro in subscript is a bound variable, e.g. \rho
            found.add(node.macroname)
    # split on comma or semicolon to handle {a,b} and {i;j}
    raw = "".join(raw_chars)
    for part in re.split(r"[,;]", raw):
        m = re.match(r"\s*([A-Za-z]+)", part)
        if m:
            found.add(m.group(1))
    return found


def _collect_bound(nodelist, bound):
    """Recursively walk a pylatexenc node list and collect bound variable names.

    Parameters
    ----------
    nodelist : list
        List of pylatexenc AST nodes.
    bound : set
        Accumulator — bound variable names are added here in place.
    """
    if not nodelist:
        return
    i = 0
    while i < len(nodelist):
        node = nodelist[i]
        if isinstance(node, LatexMacroNode):
            if node.macroname in _BOUND_OPS:
                # scan forward for the pattern  _  {subscript}
                j = i + 1
                while j < len(nodelist):
                    nxt = nodelist[j]
                    if isinstance(nxt, LatexCharsNode) and "_" in nxt.chars:
                        if j + 1 < len(nodelist) and isinstance(nodelist[j + 1], LatexGroupNode):
                            # _all_identifiers_in_group handles multi-index
                            # subscripts like {a,b} — captures every variable
                            bound.update(_all_identifiers_in_group(nodelist[j + 1]))
                        break
                    elif isinstance(nxt, LatexGroupNode):
                        break
                    j += 1
            if node.nodeargd and node.nodeargd.argnlist:
                _collect_bound([a for a in node.nodeargd.argnlist if a], bound)
        elif isinstance(node, LatexGroupNode):
            _collect_bound(node.nodelist, bound)
        elif isinstance(node, LatexEnvironmentNode):
            _collect_bound(node.nodelist, bound)
        i += 1


def get_bound_variables(latex):
    """Return variable names bound by sum/int/prod/lim/max etc. in a LaTeX equation.

    Parameters
    ----------
    latex : str
        Raw LaTeX equation string.

    Returns
    -------
    set of str
        Normalised variable names that are summation/integration/optimisation
        indices — these should be excluded from the physics symbol list.
        Returns empty set on parse failure so the caller can fall back safely.

    Notes
    -----
    Uses pylatexenc AST walking, not a hardcoded stop list. A variable like
    ``n`` is only excluded if it appears as ``\\sum_{n=...}`` in THIS equation;
    in another equation where ``n`` is a free physics parameter it is kept.
    """
    try:
        w = LatexWalker(latex)
        nodes, _, _ = w.get_latex_nodes()
        bound = set()
        _collect_bound(nodes, bound)
        return bound
    except Exception:
        # pylatexenc can fail on exotic physics macros; safe to return empty
        return set()


WEAK_DEFINITIONS = {
    "small", "defined as", "given by", "below", "above", "respectively",
    "the following", "as follows", "zero", "one"
}


def extract_identifiers(arxiv_id, eq_id, latex):
    """Return normalized identifiers for one equation.

    Parameters
    ----------
    arxiv_id : str
        Bare arXiv id such as ``2403.05230``.
    eq_id : str
        LaTeXML equation id, e.g. ``S2.E1``.
    latex : str
        Extracted equation LaTeX.

    Returns
    -------
    list of str
        Sorted normalized identifiers, using no-backslash names such as ``psi``.
    """
    # detect bound variables (summation indices etc.) from the LaTeX AST —
    # these are dropped without a hardcoded stop list
    bound = get_bound_variables(latex)

    symbols = set()
    symbols.update(_mathml_identifiers(arxiv_id, eq_id))
    symbols.update(_latex_command_identifiers(latex))
    symbols.update(_latex_decorated_identifiers(latex))
    symbols.update(_latex_subscript_identifiers(latex))
    symbols.update(_latex_simple_identifiers(latex))

    out = []
    for sym in symbols:
        norm = normalize_identifier(sym)
        if _keep_identifier(norm, bound):
            out.append(norm)
    return sorted(set(out), key=lambda s: (s.lower(), s))


def find_symbol_definitions(symbols, context, paper_dict=None):
    """Find definitions for symbols, with local context taking strict priority.

    Search order per symbol:

    1. Local context window (``context``): the pre-equation prose and post-equation
       'where...' clause combined by the caller.  This is the most specific source
       because it is scoped to the equation's immediate surroundings.  When a
       definition is found here, the paper-level dict is NOT consulted — this
       prevents cross-equation contamination where a symbol used with different
       meanings in different sections (e.g. ``E`` as a Pauli operator in §2 vs.
       ``E`` as an edge set in §3) picks up the wrong document-wide definition.

    2. Paper-level dict (``paper_dict``): fallback used ONLY when the local
       context search finds nothing.  Captures symbols whose definition appears
       far from the equation (first-use in document order).

    Parameters
    ----------
    symbols : list of str
        Normalized identifiers from ``extract_identifiers``.
    context : str
        Cleaned context string (pre_text + post_text) for the equation.
    paper_dict : dict, optional
        Paper-level symbol-to-definition mapping from ``build_paper_symbol_dict``.
        Defaults to empty dict.

    Returns
    -------
    dict
        Mapping ``symbol -> description``. Missing definitions are omitted.
    """
    if paper_dict is None:
        paper_dict = {}
    definitions = {}
    for symbol in symbols:
        # Local context first — equation-scoped, prevents cross-section leakage.
        desc = _definition_for_symbol(symbol, context)
        if desc:
            definitions[symbol] = desc
        elif symbol in paper_dict:
            # Paper dict as last resort — covers symbols defined far from the eq.
            definitions[symbol] = paper_dict[symbol]
    return definitions


def build_paper_symbol_dict(arxiv_id):
    """Build a paper-level symbol-to-definition dictionary by scanning the full HTML.

    Physics papers define each symbol once on first use. The per-equation context
    window often misses that definition if it appears several paragraphs away.
    This function scans every prose paragraph in the cached HTML in document order
    and records the first definition found for each symbol.

    Strategy mirrors TaDDEx (Martin et al., 2023): treat each inline math token
    as a potential target, apply Hearst patterns then POS nickname search, take
    the first hit in document order.

    Parameters
    ----------
    arxiv_id : str
        Bare arXiv id such as ``2403.05230``.

    Returns
    -------
    dict
        Maps normalized symbol key (e.g. ``'phi_0'``) to its definition string.
        Only the first definition found in document order is stored.
    """
    path = CACHE_DIR / f"{arxiv_id}.html"
    if not path.exists():
        return {}

    tree = lxml_html.parse(str(path))
    paper_dict = {}
    nlp = _get_nlp()

    # Walk every ltx_para div in document order.
    # _clean_para converts inline <math> → $tex$ and display equations → [EQ].
    for para in tree.xpath('//div[contains(@class,"ltx_para")]'):
        para_text = _clean_para(para, set())  # empty target set: no [TARGET] marker
        para_text = re.sub(r"\s+", " ", para_text).strip()
        if not para_text:
            continue

        for sent in _split_sentences(para_text):
            # Skip sentences without any inline math — nothing to define here.
            if "$" not in sent:
                continue

            # Attempt definition extraction for each inline math token.
            for m in re.finditer(r"\$([^$]{1,200})\$", sent):
                raw_tex = m.group(1).strip()
                norm = normalize_identifier(_clean_tex_token(raw_tex))
                if not norm or not _keep_identifier(norm):
                    continue
                # First occurrence in document order wins — do not overwrite.
                if norm in paper_dict:
                    continue

                # Pass 1: Hearst patterns on the sentence treated as a mini-context.
                desc = _definition_for_symbol(norm, sent)
                if desc:
                    paper_dict[norm] = desc
                    continue

                # Pass 2: POS nickname search (ScholarPhi approach).
                if nlp is not None:
                    target_forms = _symbol_latex_variants(norm)
                    target_forms.add(raw_tex)  # include the raw tex variant too
                    pos_desc = _pos_symbol_nickname(sent, target_forms, nlp)
                    if pos_desc:
                        pos_clean = _clean_definition(pos_desc)
                        if _use_definition(pos_clean):
                            paper_dict[norm] = pos_clean

    return paper_dict


def extract_symbols_for_paper(arxiv_id):
    """Extract identifiers and definitions for the dataset equations of one paper.

    Parameters
    ----------
    arxiv_id : str
        Bare arXiv id such as ``2403.05230``.

    Returns
    -------
    list of dict
        One record per dataset equation with keys:
        ``number``, ``eq_id``, ``latex``, ``identifiers``, ``definitions``.
    """
    contexts = get_contexts(arxiv_id)
    # Build paper-level symbol dict once — scans the full HTML in document order.
    # Definitions found here take priority over the narrow context window because
    # they correspond to the author's first-use definition of each symbol.
    paper_dict = build_paper_symbol_dict(arxiv_id)
    rows = []
    for eq in extract_equations(arxiv_id):
        if not eq["in_dataset"]:
            continue
        identifiers = extract_identifiers(arxiv_id, eq["eq_id"], eq["latex"])
        context = contexts.get(eq["eq_id"], "")
        definitions = find_symbol_definitions(identifiers, context, paper_dict)
        rows.append({
            "number": eq["number"],
            "eq_id": eq["eq_id"],
            "latex": eq["latex"],
            "context": context,
            "identifiers": identifiers,
            "definitions": definitions,
        })
    return rows


def normalize_identifier(token):
    """Normalize a raw MathML/LaTeX token to a stable symbol key.

    Parameters
    ----------
    token : str
        Raw identifier token.

    Returns
    -------
    str
        Cleaned symbol key, preserving useful decorators as prefixes.
    """
    token = token.strip()
    token = token.replace("\\", "")
    token = token.strip("{} ")
    token = re.sub(r"[^A-Za-z0-9_]", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _mathml_identifiers(arxiv_id, eq_id):
    """Collect identifier leaves from MathML, preserving simple subscripts."""
    path = CACHE_DIR / f"{arxiv_id}.html"
    if not path.exists() or not eq_id:
        return set()

    tree = lxml_html.parse(str(path))
    nodes = tree.xpath(f'//*[@id="{eq_id}"]')
    if not nodes:
        return set()

    root = nodes[0]
    symbols = set()

    for sub in root.xpath('.//*[local-name()="msub" or local-name()="msubsup"]'):
        children = [c for c in sub if isinstance(c.tag, str)]
        if len(children) >= 2:
            base_node, sub_node = children[0], children[1]
            # skip if subscript is a complex expression (has operators or multiple
            # identifier leaves) — text_content() would flatten it into junk like
            # a_02k1 from a_{0,2^k-1} or c1A1c2A2_s from \expectationvalue{...}_s
            if _mathml_node_is_complex(sub_node) or _mathml_node_is_complex(base_node):
                # Subscript (or base) is a complex expression: flattening it
                # produces junk keys (e.g. "c,D,o,p" from H_{1D,dipole}).
                # Collect identifier leaves from the BASE node only — the subscript
                # is discarded. This preserves "H" from H_{1D,dipole} without
                # pulling in the subscript characters.
                if base_node.tag.endswith("}mi"):
                    mi_list = [base_node]
                else:
                    mi_list = base_node.xpath('.//*[local-name()="mi"]')
                for mi in mi_list:
                    t = mi.text_content().strip()
                    if t:
                        symbols.add(t)
                continue
            base = _mathml_identifier_token(base_node)
            suffix = _mathml_identifier_token(sub_node)
            combined = _combine_base_subscript(base, suffix)
            if combined:
                symbols.add(combined)

    for mi in root.xpath('.//*[local-name()="mi"]'):
        if mi.xpath('ancestor::*[local-name()="msub" or local-name()="msubsup"]'):
            continue
        text = mi.text_content().strip()
        if not text:
            continue
        # Skip single-letter <mi> nodes that are flanked by <mo> containing "."
        # (MathML rendering of abbreviations like h.c. / H.c.).
        # A standalone physics symbol never has a bare period as its neighbour.
        if len(text) == 1:
            prev_sib = mi.getprevious()
            next_sib = mi.getnext()

            def _is_mo_dot(node):
                # MathML <mo> elements may carry a namespace or not; handle both.
                return (node is not None
                        and isinstance(node.tag, str)
                        and (node.tag == "mo" or node.tag.endswith("}mo"))
                        and node.text_content().strip() == ".")

            if _is_mo_dot(prev_sib) or _is_mo_dot(next_sib):
                continue
        symbols.add(text)

    return symbols


def _mathml_node_is_complex(node):
    """Return True if a MathML node represents a multi-token expression.

    Parameters
    ----------
    node : lxml element
        A MathML subtree (base or subscript of msub/msubsup).

    Returns
    -------
    bool
        True when the node contains operator elements ``<mo>`` or more than
        one identifier leaf ``<mi>``, meaning it is not a simple symbol.

    Notes
    -----
    A simple subscript has exactly one ``<mi>`` or ``<mn>`` child and no
    operators. Anything more complex (a_{0,2^k-1}, \expectationvalue{...}_s)
    will produce garbage keys when its text content is flattened.
    """
    has_operator = bool(node.xpath('.//*[local-name()="mo"]') or
                        node.tag.endswith("}mo"))
    mi_nodes = node.xpath('.//*[local-name()="mi"]')
    if node.tag.endswith("}mi"):
        mi_nodes = [node] + list(mi_nodes)
    return has_operator or len(mi_nodes) > 1


def _mathml_identifier_token(node):
    """Return a compact token for a MathML subtree."""
    text = re.sub(r"\s+", "", node.text_content().strip())
    return _clean_tex_token(text)


def _latex_command_identifiers(latex):
    """Collect Greek command identifiers that appear standalone in LaTeX.

    A Greek command is skipped when it appears *only* as the base of a
    subscripted expression (``\\sigma_x``) — that form is already captured by
    ``_latex_subscript_identifiers`` as ``sigma_x``.  Emitting both ``sigma``
    and ``sigma_x`` from the same equation inflates the symbol list with
    spurious entries.  A negative lookahead for ``_`` filters these out.
    """
    symbols = set()
    # Match \cmd NOT immediately followed by optional whitespace + underscore.
    # This keeps \sigma in "H = \sigma \cdot B" but drops it in "\sigma_x = ..."
    for cmd in re.findall(r"\\([A-Za-z]+)(?!\s*_)", latex):
        if cmd in GREEK:
            symbols.add(cmd)
    return symbols


def _latex_decorated_identifiers(latex):
    """Collect decorated symbols while preserving the decorator in the key.

    Examples
    --------
    ``\\mathcal{D}`` -> ``mathcal_D``
    ``\\hat{X}`` -> ``hat_X``
    ``\\bar{\\theta}`` -> ``bar_theta``
    """
    symbols = set()
    pattern = r"\\(" + "|".join(sorted(DECORATORS, key=len, reverse=True)) + r")\s*\{([^{}]{1,40})\}"
    for deco, body in re.findall(pattern, latex):
        base = _clean_tex_token(body)
        if base and _keep_identifier(base):
            symbols.add(f"{deco}_{base}")
    return symbols


def _latex_subscript_identifiers(latex):
    """Collect identifiers such as ``p_A``, ``r_{F}``, and ``g_{0}``."""
    symbols = set()
    base = r"(?:\\[A-Za-z]+|[A-Za-z])"
    sub = r"(?:\{[^{}]{1,40}\}|[A-Za-z0-9])"
    for raw_base, raw_sub in re.findall(rf"({base})\s*_\s*({sub})", latex):
        combined = _combine_base_subscript(raw_base, raw_sub)
        if combined:
            symbols.add(combined)
    return symbols


def _latex_simple_identifiers(latex):
    """Collect remaining simple one-letter identifiers without fragmenting words.

    Superscript labels (``^{J}``, ``^S``) are stripped early so attack-class
    markers and perturbation-order labels do not leak as physics symbols.
    Abbreviations like ``h.c.`` (Hermitian conjugate) are also removed.
    """
    symbols = set()

    cleaned = re.sub(r"\\(?:operatorname|mathrm|text|textrm)\{[^{}]*\}", " ", latex)
    # Strip h.c. / H.c. (Hermitian conjugate abbreviation) before single-letter scan.
    cleaned = re.sub(r"\b[Hh]\.c\.", " ", cleaned)
    # Strip superscript annotations: ^{J}, ^{rm sys}, ^2, etc.
    # These are typically order labels, attack labels, or perturbation indices —
    # never standalone physics identifiers. Legitimate superscript symbols like
    # e^{i\phi} are captured by _latex_command_identifiers, not here.
    cleaned = re.sub(r"\^\s*\{[^{}]{1,20}\}", " ", cleaned)
    cleaned = re.sub(r"\^\s*[A-Za-z0-9]", " ", cleaned)
    cleaned = re.sub(r"\\[A-Za-z]+\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9])", " ", cleaned)
    cleaned = re.sub(r"[A-Za-z]\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9])", " ", cleaned)
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = re.sub(r"[A-Za-z]{2,}", " ", cleaned)

    for letter in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", cleaned):
        symbols.add(letter)

    return symbols


def _keep_identifier(symbol, bound=None):
    """Return True when a normalized token is worth keeping as a physics symbol.

    Parameters
    ----------
    symbol : str
        Normalized identifier key.
    bound : set, optional
        Bound variable names detected from the equation AST (summation indices
        etc.). If the symbol matches a bound variable it is dropped. Defaults
        to empty set so callers that do not pass bound still work.
    """
    if bound is None:
        bound = set()
    if not symbol:
        return False
    # drop if it is a known bound index variable in this equation
    if symbol in bound:
        return False
    if symbol in COMMAND_STOP:
        return False
    # IDENTIFIER_STOP kept only for clear non-physics single letters that are
    # never meaningful on their own (standard operators / functions)
    if symbol in IDENTIFIER_STOP:
        return False
    if symbol.isdigit():
        return False
    if len(symbol) > 32:
        return False
    if _looks_like_fused_junk(symbol):
        return False
    if "_" in symbol:
        base, suffix = symbol.split("_", 1)
        if not base or not suffix:
            return False
        if len(suffix) > 20:
            return False
        # reject operator_subscript combos like tr_B, log_2, max_rho
        if base in IDENTIFIER_STOP or base in COMMAND_STOP:
            return False
        # Only apply the case-insensitive stop check for multi-char bases.
        # Single-char bases (e.g. "E" in E_v) are legitimate physics symbols;
        # IDENTIFIER_STOP's lowercase entries ("e") must not block them.
        if len(base) > 1 and base.lower() in {s.lower() for s in IDENTIFIER_STOP}:
            return False
        # drop if the base is a bound variable
        if base in bound:
            return False
    return True


def _looks_like_fused_junk(symbol):
    """Reject tokens caused by flattening long expressions into one string."""
    if len(symbol) >= 14 and "_" not in symbol:
        return True
    if len(re.findall(r"[A-Z]", symbol)) >= 4 and len(symbol) >= 8:
        return True
    if re.search(r"(time|simulation|Caching|DFTT)", symbol, flags=re.IGNORECASE):
        return False
    return False


def _symbol_present_case_sensitive(target_regex, chunk):
    """Return True only if the symbol appears in the chunk with exact case.

    Parameters
    ----------
    target_regex : str
        The regex built by ``_symbol_regex``, which embeds the symbol variants.
    chunk : str
        The definition chunk to check.

    Returns
    -------
    bool
        True when a case-sensitive search finds the symbol in the chunk.

    Notes
    -----
    ``re.IGNORECASE`` on the full pattern causes H to match $h$ and steal its
    definition. This guard re-runs the same regex without that flag so cue
    words can still be case-insensitive at the call site while the symbol
    itself is matched exactly.
    """
    return bool(re.search(target_regex, chunk))


def _pos_symbol_nickname(context_text, target_forms, nlp):
    """Search for a symbol's nickname using POS tagging (ScholarPhi approach).

    Replaces the target ``$symbol$`` occurrences with ``TARGETSYM`` and all
    other ``$...$`` tokens with ``MATHSYM``, then POS-tags the sentence.
    Searches LEFT of ``TARGETSYM`` for a contiguous DT/JJ/NN sequence
    (the noun phrase naming the symbol).  Falls back to searching RIGHT if
    nothing is found on the left.

    This reproduces the core logic of ScholarPhi's ``search_symbol_nickname``
    (Allen AI, 2021) using only ``en_core_web_sm`` — no downloaded model
    weights required.

    Parameters
    ----------
    context_text : str
        Raw context string, may contain ``$...$``, ``[TARGET]``, ``[EQ]``.
    target_forms : set of str
        LaTeX strings (without dollar signs) that spell the target symbol,
        e.g. ``{'phi_0', '\\\\phi_0', '\\\\phi_{0}'}``.
    nlp : spacy.Language
        Loaded spaCy model.

    Returns
    -------
    str
        Extracted noun phrase text, or empty string if not found.
    """
    # Replace every $target_form$ with the placeholder TARGETSYM.
    # Sort by length descending so longer forms match first (avoids partial hits).
    escaped = sorted(
        (re.escape(f) for f in target_forms), key=len, reverse=True
    )
    target_pat = re.compile(r"\$\s*(?:" + "|".join(escaped) + r")\s*\$")
    text = target_pat.sub("TARGETSYM", context_text)

    # Replace equation-level placeholder and any remaining $...$ math tokens.
    text = text.replace("[TARGET]", "MATHSYM").replace("[EQ]", "MATHSYM")
    text = re.sub(r"\$[^$]{1,200}\$", "MATHSYM", text)

    doc = nlp(text)

    for sent in doc.sents:
        # Find the first TARGETSYM token in this sentence.
        toks = list(sent)
        target_idx = next(
            (i for i, t in enumerate(toks) if t.text == "TARGETSYM"), None
        )
        if target_idx is None:
            continue

        # Walk LEFT collecting contiguous NP-tag tokens (Penn Treebank tags).
        np_left = []
        for i in range(target_idx - 1, -1, -1):
            t = toks[i]
            if t.tag_ in _NP_TAGS and t.text not in {"MATHSYM", "TARGETSYM"}:
                np_left.insert(0, t.text)
            else:
                break
        if np_left:
            phrase = " ".join(np_left).strip().strip(",:;-")
            # reject trivially short results (just "the", "a", one letter)
            if len(phrase) > 2 and phrase.lower() not in {"the", "a", "an"}:
                return phrase

        # Walk RIGHT when nothing useful is left of the symbol.
        np_right = []
        for i in range(target_idx + 1, len(toks)):
            t = toks[i]
            if t.tag_ in _NP_TAGS and t.text not in {"MATHSYM", "TARGETSYM"}:
                np_right.append(t.text)
            else:
                break
        if np_right:
            phrase = " ".join(np_right).strip().strip(",:;-")
            if len(phrase) > 2 and phrase.lower() not in {"the", "a", "an"}:
                return phrase

    return ""


def _definition_for_symbol(symbol, context):
    """Return the first high-precision definition found for one symbol."""
    if not context:
        return ""

    target = _symbol_regex(symbol)
    for chunk in _definition_chunks(context):
        # desc capture: stop at real sentence boundaries (. ; [) but:
        # - allow dots inside abbreviations (Eq., Fig.) by requiring dot+space+capital
        # - allow inline math $...$ to pass through so "defined as $r_F=...$" captures fully
        # - stop at ( to avoid pulling in parenthetical references
        _desc = r"(?:[^.;\[()\$]|\.\s*(?![A-Z\d])|\$[^\$]{1,80}\$)+"
        patterns = [
            rf"^(?:where|with)?\s*{target}\s+(?:is|denotes|represents|stands for|refers to)\s+(?P<desc>{_desc})",
            rf"^(?:where|with)?\s*{target}\s+is\s+defined\s+as\s+(?P<desc>{_desc})",
            rf"^(?:let|take)\s+{target}\s+(?:be|denote)\s+(?P<desc>{_desc})",
            rf"^(?:where|with)?\s*{target}\s+are\s+(?P<desc>{_desc})",
            # pre-symbol: "the Hamiltonian $H$" or "the density matrix $\rho$"
            rf"(?:the|a|an)\s+(?P<desc>[A-Za-z][A-Za-z\s-]{{1,50}}?)\s*{target}",
            # post-symbol fallback: "X the Hamiltonian" (no verb)
            rf"(?:the|a|an)\s+(?P<desc>[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){{0,6}})\s+{target}\s*$",
        ]

        for pattern in patterns:
            # IGNORECASE only on cue words (where/is/denotes etc), not on the
            # symbol token itself — otherwise H matches $h$ and steals its def
            match = re.search(pattern, chunk, flags=re.IGNORECASE)
            if match and _symbol_present_case_sensitive(target, chunk):
                desc = _clean_definition(match.group("desc"))
                if _use_definition(desc):
                    return desc

    # Hearst + pre-symbol patterns all failed.
    # Fallback: POS-tag-based noun-phrase search (ScholarPhi approach).
    nlp = _get_nlp()
    if nlp is not None:
        target_forms = _symbol_latex_variants(symbol)
        pos_desc = _pos_symbol_nickname(context, target_forms, nlp)
        if pos_desc:
            pos_clean = _clean_definition(pos_desc)
            if _use_definition(pos_clean):
                return pos_clean

    return ""


def _symbol_regex(symbol):
    """Build a regex that can match a normalized symbol in prose or inline LaTeX.

    Single-character symbols only match inside math-mode delimiters ``$...$``
    to prevent matching English articles or prepositions (e.g. "a", "s") as
    physics symbols. Multi-character symbols also allow bare word-boundary
    matches for cases like ``rho`` appearing in prose without dollar signs.
    """
    variants = _symbol_latex_variants(symbol)
    inline = r"\$\s*(?:" + "|".join(sorted(map(re.escape, variants), key=len, reverse=True)) + r")\s*\$"
    if len(symbol) == 1:
        # bare single-letter match causes too many false definition triggers
        return inline
    bare = r"\b" + re.escape(symbol) + r"\b"
    return r"(?:" + inline + r"|" + bare + r")"


def _symbol_latex_variants(symbol):
    """Return LaTeX spellings that correspond to one normalized symbol key."""
    variants = {symbol}

    if symbol in GREEK:
        variants.add("\\" + symbol)

    if "_" in symbol:
        base, suffix = symbol.split("_", 1)
        variants.update({
            f"{base}_{suffix}",
            f"{base}_{{{suffix}}}",
            f"{{{base}}}_{{{suffix}}}",
        })
        if base in GREEK:
            variants.update({
                f"\\{base}_{suffix}",
                f"\\{base}_{{{suffix}}}",
                f"{{\\{base}}}_{{{suffix}}}",
            })
        if base in DECORATORS and "_" in suffix:
            sub_base, sub_suffix = suffix.split("_", 1)
            variants.update({
                f"\\{base}{{{sub_base}}}_{sub_suffix}",
                f"\\{base}{{{sub_base}}}_{{{sub_suffix}}}",
            })
        if base in DECORATORS:
            variants.add(f"\\{base}{{{suffix}}}")

    if len(symbol) == 1 and symbol.isalpha():
        for deco in DECORATORS:
            variants.add(f"\\{deco}{{{symbol}}}")

    return variants


def _definition_chunks(context):
    """Split context into small definition-bearing chunks near cue words."""
    text = context.replace("[TARGET]", " [TARGET] ").replace("[EQ]", " [EQ] ")
    pieces = re.split(r"(?<=[.;])\s+|\s+\band\b\s+|,\s+(?=[\$\\A-Za-z])", text)

    chunks = []
    prefix = ""
    for piece in pieces:
        piece = re.sub(r"\s+", " ", piece).strip(" ,;:")
        if not piece:
            continue
        if re.match(r"^(where|with)\b", piece, flags=re.I):
            prefix = re.match(r"^(where|with)\b", piece, flags=re.I).group(1)
            chunks.append(piece)
            continue
        if prefix and re.search(r"\b(is|are|denotes|represents|stands for|refers to|defined as)\b", piece, re.I):
            chunks.append(prefix + " " + piece)
            continue
        if re.search(r"\b(where|with|let|take|is|are|denotes|represents|stands for|refers to|defined as)\b", piece, re.I):
            chunks.append(piece)

    return chunks


def _clean_definition(text):
    """Clean a matched definition span without making it verbose."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,:;-")
    text = re.sub(r"\s*\[TARGET\].*$", "", text)
    text = re.sub(r"\s*\[EQ\].*$", "", text)
    # strip trailing parenthetical references like "(as defined in Eq. 1)"
    text = re.sub(r"\s*\(?\s*(?:as defined in|see|cf\.?|in Eq|of Eq)\b.*$", "", text, flags=re.IGNORECASE)
    # strip trailing dangling prepositions/conjunctions left by partial pattern matches:
    # e.g. "quasi-exactly solvable sextic potential with" → strip "with"
    #      "power of two such that" → strip "such that"
    #      "permutation matrix with free phases for every" → strip "for every"
    text = re.sub(
        r"\s+(?:with|for every|for|such that|where|which|as|are|is|of|in|by|to|from|and|or|that|on)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip(" ,:;-()")
    return text[:180].strip()


def _use_definition(desc):
    """Reject weak spans that are not useful symbol meanings."""
    if not desc:
        return False
    low = desc.lower().strip(" .,:;-")
    if low in WEAK_DEFINITIONS:
        return False
    if len(low.split()) == 1 and low not in {"hamiltonian", "dissipator", "projector"}:
        return False
    if low.startswith(("defined as", "given by", "equal to")):
        return False
    return True


def _combine_base_subscript(base, suffix):
    """Combine a raw base and subscript into one normalized symbol key.

    Rejects subscripts that contain mathematical expressions (commas, powers,
    arithmetic operators) because cleaning them produces meaningless fused keys
    like a_02k1 from a_{0,2^k-1}.
    """
    # check raw suffix before cleaning — expression characters signal a complex
    # subscript that cannot be collapsed into a meaningful single key
    raw_suffix = suffix.strip("{} ")
    if re.search(r"[,\^+\-]", raw_suffix):
        return ""
    base = _clean_tex_token(base)
    suffix = _clean_tex_token(suffix)
    if not base or not suffix:
        return ""
    if base in COMMAND_STOP or suffix in COMMAND_STOP:
        return ""
    if len(suffix) > 20:
        return ""
    return f"{base}_{suffix}"


def _clean_tex_token(token):
    """Convert a tiny LaTeX token fragment into a stable symbol fragment.

    Text-mode font commands (``\\rm``, ``\\mathrm``, ``\\text``) are stripped
    before anything else so that subscripts like ``{\\rm sys}`` yield ``sys``
    rather than ``rmsys``.
    """
    token = token.strip()
    # Strip text-mode font switches BEFORE brace/backslash removal so that
    # e.g. \mathrm{sys} → {sys} → sys, and \rm sys → sys.
    token = re.sub(r"\\(?:rm|mathrm|text|textrm|mathit|mathsf)\b\s*", "", token)
    token = token.strip("{} ")
    token = re.sub(
        r"\\(?:hat|widehat|bar|overline|tilde|vec|bm|boldsymbol|mathcal|mathbb|mathscr|mathfrak|mathbf)\{([^{}]+)\}",
        r"\1",
        token,
    )
    token = token.replace("\\", "")
    token = re.sub(r"[^A-Za-z0-9]", "", token)
    return token