"""Derive meaning for enumerated equations using a layered non-generative strategy.

Signal extraction priority:
  1. Theorem/Lemma/Definition environment title with a descriptive name.
  2. Section title (when not generic) — only a last-resort context clause.
     Named-equation detection is kept for audit/hints, not as a meaning source.
  3. Schwartz-Hearst abbreviation — purpose clause.
  4. Introducing sentence — last substantive non-dangling sentence from pre-text,
     verbatim from the paper (no generation).
  5. Cross-reference context — sentence elsewhere in the document that cites
     this equation and names what it represents.
  6. Symbol definitions — always appended when available.

No generative model. No network calls.
"""

import re
from copy import deepcopy

from abbreviations import schwartz_hearst
from context import (
    _enclosing_para,
    _prev_paras,
    _next_paras,
    _clean_para,
    _split_sentences,
)

# Named equation lexicon: (prose_regex, latex_regex, canonical_name).
# prose_regex uses \b word boundaries to prevent substring false positives.
# Empty string means skip that field. Checked in order; first hit wins.
_NAMED_EQ = [
    (r"\bschr[oö]dinger\b",             r"i\s*\\hbar|i\s*\\partial_t",              "Schrödinger equation"),
    (r"\bhamiltonian\b",                 r"\\mathcal\{H\}|\\hat\{\\mathcal\{H\}\}|\\hat\{H\}", "Hamiltonian"),
    (r"\blagrangian\b",                  r"\\mathcal\{L\}\s*=",                      "Lagrangian"),
    (r"\bpartition function\b",          r"Z\s*=",                                   "partition function"),
    (r"\bmaster equation\b",             r"\\frac\{d\}\{dt\}.*\\rho|\\dot\{\\rho\}","master equation"),
    (r"\blindblad\b",                    r"",                                         "Lindblad equation"),
    (r"\bdensity (?:matrix|operator)\b", r"\\rho\s*=",                               "density matrix"),
    (r"\bfree energy\b",                 r"F\s*=",                                   "free energy"),
    (r"\bentropy\b",                     r"S\s*=\s*-.*\\log|S\s*=.*\\mathrm\{tr\}", "entropy"),
    (r"\baction\b",                      r"S\s*=\s*\\int",                           "action functional"),
    (r"\bpropagator\b|\bgreen.s function\b", r"G\s*[({]",                           "propagator"),
    # Commutator [A, B]: require a comma inside the brackets to avoid matching
    # square-bracket indexing such as Tr[...] or matrix[i,j].
    (r"\bcommut(?:ation|ator)\b",        r"\\left\[[^,\]]+,[^,\]]+\\right\]|\\comm\b", "commutation relation"),
    # Anticommutator {A, B}: require a comma inside — distinguishes {A,B} from
    # set notation {U ∈ SU | ...} which uses | not a comma after the first element.
    # Anticommutator {A, B}: exclude | (set-builder separator) but allow nested
    # curly braces from LaTeX macros like \hat{A}.  Set notation {U | U∈SU}
    # uses | as separator so [^|]+ correctly rejects it.
    (r"\banticommut\w*\b",               r"\\left\\{[^|]+,[^|]+\\right\\}|\\acomm\b",               "anticommutation relation"),
    # Expectation value <A>: require a closing \rangle to confirm a matched pair,
    # so that |n⟩⟨n| outer-product projectors (bra-ket without inner content) don't fire.
    (r"\bexpectation value\b",           r"\\langle[^|]+\\rangle",                "expectation value"),
    (r"\buncertainty (?:principle|relation)\b", r"\\Delta.*\\Delta",               "uncertainty relation"),
    (r"\bbell inequality\b|\bchsh\b",    r"",                                         "Bell/CHSH inequality"),
    (r"\bcost function\b|\bobjective function\b", r"",                              "cost/objective function"),
    (r"\bfourier transform\b",           r"\\hat\{f\}|\\mathcal\{F\}",             "Fourier transform"),
    # Eigenvalue equation: require \hat{} operator before ψ — bare 'E.*\psi' is
    # too generic and fires for edge-set E and quantum state ψ in graph theory.
    (r"\beigenval\w*\b|\beigen\w*equat\w*\b", r"\\hat\{[A-Za-z]+\}.*\\psi",       "eigenvalue equation"),
    (r"\btrace\b",                       r"\\mathrm\{tr\}|\\operatorname\{tr\}",   "trace expression"),
    # Variational expression: require functional-derivative form \frac{\delta or
    # \delta S / \delta\mathcal — Kronecker delta \delta_{ij} must NOT fire.
    (r"\bvariational\b",                 r"\\frac\{\\delta|\\delta\s*[A-Z]|\\delta\s*\\mathcal", "variational expression"),
    (r"\btransfer matrix\b",             r"T\s*=|\\mathcal\{T\}",                  "transfer matrix"),
    (r"\bcorrelation function\b",        r"\\langle.*\\rangle",                     "correlation function"),
    (r"\bwigner\b",                      r"W\s*\(",                                 "Wigner function"),
    (r"\bfidelity\b",                    r"F\s*=.*\\langle|\\mathrm\{F\}",         "fidelity"),
    (r"\bvon neumann\b",                 r"S\s*=.*\\mathrm\{tr\}.*\\log",          "von Neumann entropy"),
    # Bloch vector: \vec{\sigma} is reliable; \vec{r} is too generic (also
    # appears in position vectors, Lennard-Jones potentials, etc.).
    (r"\bbloch\b",                       r"\\vec\{\\sigma\}",                      "Bloch vector equation"),
    (r"\bunitary (?:evolution|operator)\b", r"e\s*\^\s*\{?-i|\\hat\{U\}",         "unitary evolution operator"),
    (r"\bnoise spectrum\b|\bnoise spectral density\b", r"S\s*\(\\omega\)|S\s*\\left\(\\omega", "noise spectral density"),
]

# Generic section titles — suppressed from the meaning template.

_GENERIC_SECTIONS = {
    "background", "introduction", "preliminaries", "preliminary",
    "conclusions", "conclusion", "discussion", "related work",
    "notation", "notations", "setup", "overview", "summary", "outline",
    "results", "main results", "applications", "motivation",
    "acknowledgements", "acknowledgments", "appendix",
    "methods", "approach", "techniques", "tools", "model",
}

# LaTeXML theorem-type environment class suffixes.
_THEOREM_ENV_TYPES = {
    "theorem", "lemma", "proposition", "definition", "corollary",
    "remark", "conjecture", "claim", "observation", "fact", "example",
}

# Words stripped from the END of a section title to derive the concept 'name'.
_SECTION_SUFFIX_WORDS = {
    "algorithm", "method", "equation", "equations", "theorem", "lemma", "proof",
    "model", "framework", "approach", "analysis", "results", "discussion",
    "introduction", "background", "overview", "conclusion", "conclusions",
    "definition", "corollary", "proposition", "example", "formulation",
    "derivation", "derivations", "section", "subsection", "chapter",
}

# Sentence-ending patterns that signal an incomplete setup clause.
# Sentences ending with these are NOT used as introducing sentences because
# they lead directly into the equation ("...can be written as [EQ]").
# Includes common equation-introduction phrases found in quantum physics papers.
_DANGLING_RE = re.compile(
    r'\b(?:as|as follows|namely|given by|written as|the following|'
    r'we have|we get|following|denoted|defined|expressed|reads|'
    r'below|is|are|becomes|yields|gives|satisfy|satisfies|satisfying|'
    r'reduces to|simplifies to|takes the form|take the form|'
    r'has the form|have the form|is given by|are given by|'
    r'determined by|defined by|governed by|generated by|produced by|'
    r'obtained by|expressed by|given in|follows|scales|where|'
    r'shown in|stated as|defined as|reads as|'
    r'expressed as|represented as|written in the form|'
    # "has [adjective?] representation" — equation is the matrix/block/operator rep.
    r'has\s+(?:\w+\s+)*representation|'
    # "proportional to", "equal to", "equivalent to" — equation is the RHS.
    r'proportional to|equal to|equivalent to|'
    # Bare "with" or "therefore"/"thus"/"hence" at sentence end.
    r'therefore|thus|hence|'
    r'(?:denot\w+|expressed)\s+(?:by|with|as)|'
    # "of the form", "in the form", plain "given by" / "obtained by" — not already covered.
    r'of the form|in the form|given by|obtained by)\s*[,:\.]?\s*$'
    # Citation at end: only flag as dangling when followed by comma or colon (setup clause),
    # NOT plain period — "[29]." ends a complete sentence, "[29]," or "[29]:" does not.
    r'|\[\s*[\d,;\s]*\]\s*[,:]\s*$',
    re.I
)

_MIN_PRE_WORDS   = 8
_MIN_INTRO_WORDS = 8   # primary threshold; see _MIN_INTRO_WORDS_SHORT for fallback
_MIN_INTRO_WORDS_SHORT = 5  # fallback: named conditions like "(C5) Additivity..." are valid

# Phrases that signal the Schwartz-Hearst algorithm returned a prose fragment
# rather than a genuine abbreviation long-form expansion.
_ABBREV_REJECT_RE = re.compile(
    r'\b(?:at best|at least|however|but\b|so that|in order|is not|are not|'
    r'do not|cannot|we note|note that|thus|hence|therefore|although|since|'
    r'valid|available|investigated|possible|presented|applied)\b',
    re.I
)

# Generic section-level openers that add no per-equation information.
# Sentences matching this pattern are skipped; post_text is preferred over them.
_SECTION_OPENER_RE = re.compile(
    r'^In\s+(?:this|the|our)\s+'
    r'(?:section|paper|work|chapter|subsection|appendix|following|above|below)',
    re.I
)

# Used in _truncate_dangling: the rightmost dangling word/phrase triggers
# truncation at that position so the setup clause up to that point is kept.
# We find the START of the dangling match and cut there.
_DANGLING_TRUNC_RE = re.compile(
    r'\s*\b(?:becomes|yields|gives|follows|scales|is|are|reads|below|where|'
    r'reduces to|simplifies to|takes the form|take the form|'
    r'has the form|have the form|is given by|are given by|given by|obtained by|'
    r'defined by|defined as|expressed as|represented as|'
    r'has\s+(?:\w+\s+)*representation|'
    r'proportional to|equal to|equivalent to|'
    # Bare "as" or "where": very short terminal — must start far enough into the sentence
    # (enforced by the m.start() > 20 guard in _truncate_dangling).
    r'(?:can\s+be\s+(?:written|expressed|obtained|defined|given)\s+)?as|'
    r'we\s+(?:define|write|denote|get|obtain|have)\s+(?:the\s+\w+\s+)?(?:degree\s+)?as)'
    r'\s*[,:\.]?\s*$',
    re.I
)


# Words inside \text{...} that are NOT equation labels: grammatical connectors,
# h.c. abbreviations, and subscript shortforms in operator names.
_TEXT_FILTER = frozenset({
    "where", "and", "or", "if", "then", "with", "for", "of", "the",
    "h.c.", "c.c.", "H.c.", "C.C.", "a.e.", "i.e.", "e.g.",
    "etc", "vs", "resp", "const", "otherwise",
    # Subscript shortforms: appear in operator names, not as equation labels.
    "eff", "ext", "int", "tot", "max", "min", "opt", "loc", "eq",
    "in", "out", "sys", "env", "bath", "vac", "free", "kin", "cl",
    "phys", "num", "den", "rel", "abs", "th", "el", "mag", "em",
    "crit", "sat", "ss", "exc", "gr", "ref", "src",
    # State/basis labels commonly written in \text{} inside kets/subscripts.
    "gs", "g.s.", "ground", "excited",
    # Math operators written in \text{} — real/imaginary part, modulo, trace, etc.
    # Stored lowercase because _extract_inline_label checks label.lower() in this set.
    # These appear INSIDE expression notation, not as standalone equation labels.
    "re", "im", "mod", "tr", "det", "rank", "span", "diag",
    "sgn", "sign", "exp", "log", "ln", "sin", "cos", "tan",
    "arg", "dim", "ker", "coker", "supp", "vol", "prob",
})


def _extract_inline_label(latex):
    """Extract \\text{...} annotation labels from an equation's LaTeX string.

    Many quantum physics papers embed a short label directly inside the equation
    to identify it, e.g.::

        H(k)\\mathcal{T}_+^{-1} = +H(-k) \\quad (\\text{TRS})
        \\{A, B\\} = 0 \\quad \\text{(anticommutation relation)}

    These labels are the most discriminating per-equation signal when multiple
    sibling equations share the same section title, paragraph, and pre-text
    (e.g. a symmetry classification table listing TRS, TRS†, PHS, PHS†, CS).

    Parameters
    ----------
    latex : str
        Raw LaTeX string of the equation.

    Returns
    -------
    str
        The extracted label (e.g. 'TRS', 'PHS†', 'anticommutation relation'),
        or empty string when no informative label is found.
    """
    # Collect (position, label) pairs so we can look ahead for ^† after each match.
    candidates = []
    for m in re.finditer(r'\\text\{([^}]{1,60})\}', latex):
        raw = m.group(1)
        label = re.sub(r'[${}\\^_]', '', raw).strip(" .,;:()")
        if not label or label.lower() in _TEXT_FILTER or len(label) > 40:
            continue
        # Commas almost always mean prose embedded inside \text{...}, e.g.
        # \text{$S_{i,j}$ bad}; those are not equation-level labels.
        if "," in label:
            continue
        # Reject single-character labels: subscript shortforms like _{\text{s}},
        # _{\text{b}}, _{\text{i}} appear frequently in operator names and are
        # not equation-level discriminators.
        if len(label) < 2:
            continue
        # Reject \text{} that is immediately preceded by a letter (no space or
        # brace boundary). This catches composite operator names like i\text{CNOT}
        # or C\text{NOT} where the \text{} is a typographic suffix of the operator
        # name, not a standalone equation label.
        if m.start() > 0 and latex[m.start() - 1].isalpha():
            continue
        # Reject \text{} labels that appear INSIDE bra-ket notation.
        # \ket{\text{GS}} or |\text{GS}\rangle marks a quantum state, not an
        # equation identifier. Check the 20 chars preceding the match start.
        before = latex[max(0, m.start() - 20):m.start()]
        if re.search(r'\\(?:ket|bra|braket)\{[^}]*$|\|\s*$', before):
            continue
        # Reject labels used as ordinary subscripts/superscripts.
        if re.search(r'[_^]\{?\s*$', before):
            continue
        # Look at the 25 characters following the \text{} group for ^\dagger or ^*.
        # This handles TRS† where the paper writes \text{TRS}^\dagger outside the
        # braces and the two pieces appear in the LaTeX string adjacently.
        after = latex[m.end():m.end() + 25]
        if re.search(r'\\dagger|†', after):
            label += '†'
        elif re.search(r'\^\*|\*', after[:5]):
            label += '*'
        candidates.append((m.start(), label))

    if not candidates:
        return ""
    # Return the last informative label — rightmost in the equation is usually
    # the annotation tag (e.g. the (TRS) at the end of the line).
    return candidates[-1][1]


def get_theorem_env(table):
    """Check whether the equation lives inside a theorem/lemma/definition block.

    Walks up the DOM looking for 'ltx_theorem_X' class ancestors.  Returns the
    parenthetical name from the title when present (e.g. '(KS Theorem)' from
    'Theorem 1.6 (KS Theorem)').  Purely numeric titles ('Claim 4.2.') are
    suppressed since they carry no semantic content.

    Parameters
    ----------
    table : lxml element

    Returns
    -------
    tuple[str, str]
        (env_type, env_title). Both empty strings when no informative theorem
        environment is found.
    """
    cur = table.getparent()
    while cur is not None:
        cls = cur.get("class") or ""
        for env_type in _THEOREM_ENV_TYPES:
            if f"ltx_theorem_{env_type}" in cls:
                titles = cur.xpath('.//*[contains(@class,"ltx_title")]')
                if not titles:
                    return env_type, ""
                raw = titles[0].text_content().strip()
                # Prefer the parenthetical descriptive name.
                m = re.search(r"\(([^)]{4,80})\)", raw)
                if m:
                    return env_type, m.group(1).strip()
                # Strip "Theorem 1.6" / "Claim 4.2." style prefix.
                clean = re.sub(
                    r"^(?:theorem|lemma|proposition|definition|corollary|"
                    r"remark|conjecture|claim|example)\s*[\d\.]*\.?\s*",
                    "", raw, flags=re.I
                ).strip()
                # Suppress titles that are purely numeric/empty after stripping.
                if not clean or re.fullmatch(r"[\d\.\s]+", clean):
                    return env_type, ""
                return env_type, clean
        cur = cur.getparent()
    return "", ""


def get_section_title(table):
    """Return (name, contained_section) for the nearest enclosing section.

    Parameters
    ----------
    table : lxml element

    Returns
    -------
    tuple[str, str]
    """
    cur = table.getparent()
    while cur is not None:
        cls = cur.get("class") or ""
        if any(c in cls for c in ("ltx_section", "ltx_subsection", "ltx_chapter")):
            titles = cur.xpath('.//*[contains(@class,"ltx_title")]')
            if titles:
                # LaTeXML renders inline math in section titles with alt-text
                # duplication (e.g. "C𝐶Citalic_C"). Replace each math element's
                # rendered text with its `alttext` attribute value (the clean LaTeX
                # source) before extracting plain text.
                from copy import deepcopy as _dcopy
                title_el = _dcopy(titles[0])
                for math_el in title_el.xpath('.//math'):
                    alt = math_el.get("alttext", "")
                    # Clear children and set tail text; use tail so the alt text
                    # is inserted in the text flow where the math element was.
                    for child in list(math_el):
                        math_el.remove(child)
                    math_el.text = alt
                raw = title_el.text_content().strip()
                # Strip leading section number: "1", "A.1", "III.2", etc.
                # [A-Z]\.(?:\d+...)? handles compound labels like "A.1".
                # [\s\u00a0]* covers the non-breaking space LaTeXML sometimes inserts.
                contained = re.sub(
                    r"^\s*(?:[IVXLCDM]+-[A-Z]\.?[\s\u00a0]*"  # VI-B style
                    r"|[IVXLCDM]+\.?[\s\u00a0]*"               # III.2 style
                    r"|[A-Z]\.(?:\d+(?:\.\d+)*\.?)?[\s\u00a0]*"  # A.1 style
                    r"|\d+(?:\.\d+)*\.?[\s\u00a0]*)"           # 3.1 style
                    , "", raw
                ).strip() or raw.strip()
                # Second pass: "A." stripped from "A.1 Title" may leave "1 Title".
                contained = re.sub(
                    r"^\d+(?:\.\d+)*\.?[\s\u00a0]*", "", contained
                ).strip() or contained
                words = contained.split()
                while words and words[-1].lower().rstrip(".") in _SECTION_SUFFIX_WORDS:
                    words.pop()
                # Strip trailing function words left after suffix removal
                # e.g. "Implementing the" after dropping "algorithm".
                _TRAILING_ARTICLES = {"the", "a", "an", "of", "for", "in", "and", "or"}
                while words and words[-1].lower().rstrip(".,") in _TRAILING_ARTICLES:
                    words.pop()
                name = " ".join(words).strip() or contained
                return name, contained
        cur = cur.getparent()
    return "", ""


def get_named_equation(para_text, latex):
    """Match the named equation lexicon against immediate paragraph prose and LaTeX.

    Deliberately restricted to the immediate enclosing paragraph (not the full
    multi-paragraph fallback context) so that incidental mentions of "Hamiltonian"
    in an Introduction section do not fire for unrelated scaling equations.

    Parameters
    ----------
    para_text : str
        Text from the immediate paragraph around the equation only.
    latex : str

    Returns
    -------
    str
        Canonical equation name, or empty string if no match.
    """
    for prose_pat, latex_pat, canon_name in _NAMED_EQ:
        prose_hit = bool(prose_pat and re.search(prose_pat, para_text, re.I))
        latex_hit = bool(latex_pat and re.search(latex_pat, latex))
        if latex_pat:
            # When a LaTeX discriminator is defined, require BOTH signals.
            # Prose alone is too broad — "Hamiltonian" or "anticommute" appear
            # in paragraphs that introduce equations of a different type.
            # LaTeX alone is too broad — \left\{ fires on set notation, \langle
            # fires on bra-ket projectors, \delta fires on Kronecker deltas.
            # Requiring both reduces false positives without needing paper-specific
            # logic (equations that ARE the named type will have both signals).
            if prose_hit and latex_hit:
                return canon_name
        else:
            # No LaTeX pattern (Lindblad, Bell, cost function, etc.) — prose alone.
            if prose_hit:
                return canon_name
    return ""


def _get_immediate_pre_text(table):
    """Return pre-text from the immediate enclosing paragraph only, no fallback.

    Used for named_eq matching to avoid picking up incidental physics keywords
    from distant paragraphs that appeared in the multi-paragraph fallback context.

    Parameters
    ----------
    table : lxml element

    Returns
    -------
    str
    """
    para = _enclosing_para(table)
    if para is None:
        return ""
    children = list(para)
    try:
        idx = next(i for i, c in enumerate(children) if c is table)
    except StopIteration:
        return _strip_markers(_clean_para(para, set()))
    p = deepcopy(para)
    for child in list(p)[idx:]:
        p.remove(child)
    return _strip_markers(_clean_para(p, set()))


def get_pre_text(table):
    """Return cleaned prose from the paragraph section before `table`.

    Parameters
    ----------
    table : lxml element

    Returns
    -------
    str
    """
    para = _enclosing_para(table)
    if para is None:
        return ""
    children = list(para)
    try:
        idx = next(i for i, c in enumerate(children) if c is table)
    except StopIteration:
        return _strip_markers(_clean_para(para, set()))
    p = deepcopy(para)
    for child in list(p)[idx:]:
        p.remove(child)
    text = _nearest_pre_segment(_clean_para(p, set()))
    if len(text.split()) < _MIN_PRE_WORDS:
        prevs = _prev_paras(para)
        if prevs:
            text = (_strip_markers(_clean_para(prevs[-1], set())) + " " + text).strip()
    return text


_POST_DEF_START_RE = re.compile(
    r"^\s*(?:where|with|here|in\s+which|in\s+this\s+notation)\b|"
    r"^\s*\$[^$]{1,120}\$\s*(?:is|are|denotes?|represents?|stands?\s+for|"
    r"refers?\s+to|corresponds?\s+to)\b",
    re.I,
)


def _post_definition_block(text):
    """Keep only the leading local symbol-definition block."""
    result = []
    for sent in _split_sentences(text or ""):
        clean = re.sub(r"\s+", " ", sent).strip()
        if not clean:
            continue
        if not result:
            if _POST_DEF_START_RE.search(clean[:260]):
                result.append(clean)
            else:
                break
        elif _POST_DEF_START_RE.search(clean[:260]):
            result.append(clean)
        else:
            break
    return " ".join(result)


def _nearest_pre_segment(text):
    """Return prose after the last previous display equation marker."""
    if not text:
        return ""
    parts = re.split(r"\[(?:EQ|TARGET)\]", text)
    return _strip_markers(parts[-1] if parts else text)


def _nearest_post_segment(text):
    """Return prose before the next display equation marker."""
    if not text:
        return ""
    parts = re.split(r"\[(?:EQ|TARGET)\]", text, maxsplit=1)
    return _strip_markers(parts[0] if parts else text)


def get_post_text(table):
    """Return cleaned local definition prose immediately following `table`.

    Many quantum physics papers display the equation first and then write
    "where $X$ is... and $Y$ is..." in the next sentence. That clause is
    the richest source of symbol definitions but lies AFTER the equation,
    outside the reach of get_pre_text().

    Only a leading definition block is returned. Ordinary explanatory prose is
    handled separately by get_post_explanation().

    Parameters
    ----------
    table : lxml element

    Returns
    -------
    str
    """
    para = _enclosing_para(table)
    if para is None:
        return ""
    children = list(para)
    try:
        idx = next(i for i, c in enumerate(children) if c is table)
    except StopIteration:
        return ""
    # Extract only the content that follows the equation within this paragraph.
    p = deepcopy(para)
    for child in list(p)[:idx + 1]:
        p.remove(child)
    text = _nearest_post_segment(_clean_para(p, set())).strip()
    # If the same-paragraph tail is thin, also grab the first following paragraph.
    if len(text.split()) < _MIN_PRE_WORDS:
        nexts = _next_paras(para)
        if nexts:
            nxt = _strip_markers(_clean_para(nexts[0], set())).strip()
            text = (text + " " + nxt).strip()
    return _post_definition_block(text)


def extract_abbreviation(pre_text):
    """Extract the most informative abbreviation from pre-table prose via Schwartz-Hearst.

    Deliberately narrowed to the last 2 sentences of pre_text. Using the full
    pre_text caused cross-equation contamination: when 6 sibling equations share
    a paragraph, all 6 inherited the same S-H result (e.g. 'time-reversal
    symmetry') even for equations about PHS and CS. The 2-sentence window stays
    close to the equation being described.

    Parameters
    ----------
    pre_text : str

    Returns
    -------
    str
    """
    if not pre_text or len(pre_text.split()) < _MIN_PRE_WORDS:
        return ""
    # Narrow to the 2 sentences immediately before the equation.
    from context import _split_sentences
    sents = _split_sentences(pre_text)
    narrow = " ".join(sents[-2:]) if len(sents) >= 2 else pre_text
    if len(narrow.split()) < _MIN_PRE_WORDS:
        narrow = pre_text   # fallback to full text when narrow is too short
    try:
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(
            doc_text=narrow, first_definition=True
        )
        if pairs:
            # Return the long-form expansion, not the acronym — so that the meaning
            # string reads "Introduced in the context of Dirac-Heisenberg-Wigner
            # formalism" rather than "Introduced in the context of DHW".
            key = max(pairs, key=lambda k: len(pairs[k]))
            result = pairs[key]
            # Reject prose fragments: Schwartz-Hearst sometimes matches partial
            # sentences ("sample wide basis at best") that contain hedging or
            # negation words — these are not meaningful abbreviation expansions.
            if _ABBREV_REJECT_RE.search(result):
                return ""
            return result
    except Exception:
        pass
    return ""


def _truncate_dangling(sent):
    """Strip a recognized dangling tail from a sentence, returning the setup clause.

    Used as a last resort: when a sentence like "If X, the state becomes"
    would otherwise be discarded, truncating at "becomes" yields "If X" which
    still carries useful setup context.  Returns empty string if the truncated
    result is too short or the match position is near the start.

    Parameters
    ----------
    sent : str

    Returns
    -------
    str
        Truncated sentence, or empty string when truncation yields too little.
    """
    m = _DANGLING_TRUNC_RE.search(sent)
    if m and m.start() > 20:
        fragment = sent[:m.start()].rstrip(" ,:")
        if len(fragment.split()) >= _MIN_INTRO_WORDS:
            return fragment
    return ""


def _scan_sents(sents, min_words):
    """Single-direction scan returning best non-dangling sentence and best dangling.

    Iterates reversed(sents) so the sentence closest to the equation is tried first.
    Returns (intro, best_dangling): intro is a non-dangling sentence with >= min_words;
    best_dangling is the longest dangling sentence >= min_words for truncation fallback.

    Parameters
    ----------
    sents : list[str]
    min_words : int

    Returns
    -------
    tuple[str, str]
    """
    best_dangling = ""
    for sent in reversed(sents):
        cleaned = _clean_intro(sent)
        nwords = len(cleaned.split())
        if nwords < min_words:
            continue
        if _SECTION_OPENER_RE.search(cleaned):
            continue
        if not _DANGLING_RE.search(cleaned):
            return cleaned, best_dangling
        if not best_dangling:
            best_dangling = cleaned
    return "", best_dangling


def extract_intro_sentence(pre_text):
    """Return the last substantive non-dangling sentence from pre_text verbatim.

    Sentences that lead directly into the equation ("can be written as",
    "is given by:", "we have:") are rejected because they produce incomplete
    clauses when the equation itself is stripped out.  Generic section openers
    ("In this section we present...") are also skipped — they describe the
    section, not the specific equation.

    Two-pass strategy:
      Pass 1 (>= _MIN_INTRO_WORDS = 8 words): returns a full-length sentence.
        Uses reversed order so closest sentence is tried first; skips short
        candidates to avoid picking up the 5-word dangling lead-in when a
        good longer sentence exists earlier in the pre-text.
      Pass 2 (>= _MIN_INTRO_WORDS_SHORT = 5 words): only runs when pass 1 finds
        nothing. Recovers short but complete named conditions like "(C5) Additivity
        for direct sum states." that are < 8 words.
      Truncation fallback: when both passes fail to find a non-dangling sentence,
        strips the dangling tail from the best available dangling sentence and
        returns the setup clause prefix.

    Citation placeholders and trailing colons are cleaned before checks.

    Parameters
    ----------
    pre_text : str

    Returns
    -------
    str
        Clean introducing sentence or truncated setup clause, or empty string
        if nothing qualifies.
    """
    if not pre_text:
        return ""
    sents = _split_sentences(pre_text)

    # Pass 1: full-length sentences (>= 8 words).
    intro, best_dangling = _scan_sents(sents, _MIN_INTRO_WORDS)
    if intro:
        return intro

    # Pass 2: short but complete sentences (>= 5 words) — named conditions etc.
    intro, best_dangling_short = _scan_sents(sents, _MIN_INTRO_WORDS_SHORT)
    if intro:
        return intro
    # Merge best_dangling candidates from both passes.
    if not best_dangling:
        best_dangling = best_dangling_short

    # Last resort: truncate the dangling tail from the best available sentence.
    if best_dangling:
        return _truncate_dangling(best_dangling)
    return ""


def extract_lead_in_phrase(pre_text):
    """Return the dangling phrase that leads directly into the equation.

    These are the sentences `extract_intro_sentence` rejects — e.g. "is given
    by:", "takes the form:", "can be written as:".  They are exactly the
    strongest triggers for rule-based meaning synthesis because they tell you
    *how* the equation is being introduced.  We keep only the phrase up to but
    not including the dangling tail itself, so "The sensitivity δB is given by"
    becomes "The sensitivity δB".

    Parameters
    ----------
    pre_text : str

    Returns
    -------
    str
        The full dangling sentence (not truncated), so synthesis rules can
        inspect both the subject and the dangling verb phrase together.
        Empty string when no dangling sentence exists.
    """
    if not pre_text:
        return ""
    sents = _split_sentences(pre_text)
    # Walk from closest sentence outward; return first dangling one found.
    for sent in reversed(sents):
        cleaned = _clean_intro(sent)
        if len(cleaned.split()) < _MIN_INTRO_WORDS_SHORT:
            continue
        if _SECTION_OPENER_RE.search(cleaned):
            continue
        if _DANGLING_RE.search(cleaned):
            return cleaned
    return ""


def get_post_explanation(table):
    """Return the first explanatory sentence after the equation, even without 'where'.

    `get_post_text` already handles 'where $X$ is...' symbol clauses.  This
    function separately captures sentences like "This is a controlled phase
    gate applied to qubit i." that follow an equation but contain no math —
    they carry equation-level meaning, not symbol definitions.

    Parameters
    ----------
    table : lxml element

    Returns
    -------
    str
        First non-where post sentence, or empty string.
    """
    para = _enclosing_para(table)
    if para is None:
        return ""
    children = list(para)
    try:
        idx = next(i for i, c in enumerate(children) if c is table)
    except StopIteration:
        return ""
    # Grab the tail text within this paragraph after the equation.
    p = deepcopy(para)
    for child in list(p)[:idx + 1]:
        p.remove(child)
    in_para = _nearest_post_segment(_clean_para(p, set())).strip()
    # Also try first following paragraph when in-para tail is thin.
    candidates = []
    if in_para:
        candidates.append(in_para)
    nexts = _next_paras(para)
    if nexts:
        candidates.append(_strip_markers(_clean_para(nexts[0], set())).strip())
    for text in candidates:
        sents = _split_sentences(text)
        for sent in sents:
            cleaned = _clean_intro(sent)
            # Must be a complete sentence (not starting with 'where' or math).
            if (len(cleaned.split()) >= 5
                    and not re.match(r"^where\b", cleaned, re.I)
                    and not re.match(r"^\$", cleaned)
                    and not re.match(r"^(?:fig\.?|figure)\b", cleaned, re.I)
                    and not re.search(r"\b(?:plot|plots|shown in Fig\.?|shown in Figure)\b", cleaned, re.I)
                    and not _SECTION_OPENER_RE.search(cleaned)):
                return cleaned
    return ""


def _arxiv_href_variants(eq_id):
    """Return all plausible href values for an equation ID in arXiv LaTeXML HTML.

    arXiv LaTeXML renders equation anchors with IDs like 'S2.E3', 'Sx1.E5',
    'p3.E2', etc. but cross-reference hrefs can use '#S2.E3', '#eq:label',
    '#equation.3', and the bare form. This function generates the candidate
    set so the xpath query covers all formats without false positives.
    """
    variants = {f"#{eq_id}", eq_id}

    # Pattern: Sx\d+.E\d+, S\d+.E\d+, p\d+.E\d+
    # arXiv also uses shortened forms in hrefs: strip leading section prefix.
    m = re.search(r"(?:S(?:x?\d+)?|p\d+)\.(E\d+[a-z]?)", eq_id)
    if m:
        variants.add(f"#{m.group(1).lower()}")  # e.g. #e3
        variants.add(f"#{m.group(0)}")           # e.g. #S2.E3

    # Some papers use eq_id as-is but with different case conventions.
    variants.add(f"#{eq_id.lower()}")

    return variants


def get_cross_ref_context(eq_id, tree):
    """Find sentences elsewhere in the document that reference this equation.

    Handles multiple arXiv LaTeXML href formats: '#S2.E3', '#Sx1.E3',
    '#p3.E2', 'eq:label', and the bare element id. Falls back to the
    first sentence of the referencing paragraph when no exact match is found.

    Parameters
    ----------
    eq_id : str
    tree : lxml tree

    Returns
    -------
    str
    """
    if not eq_id or tree is None:
        return ""

    href_variants = _arxiv_href_variants(eq_id)

    # Build a single xpath that matches any of the href variants.
    conditions = " or ".join(f'@href="{v}"' for v in sorted(href_variants))
    try:
        refs = tree.xpath(f'//a[{conditions}]')
    except Exception:
        refs = []

    for ref in refs:
        para = _enclosing_para(ref)
        if para is None:
            continue
        para_text = _strip_markers(_clean_para(para, set()))
        ref_text = ref.text_content().strip()
        for sent in _split_sentences(para_text):
            if ref_text and ref_text in sent and len(sent.split()) >= 6:
                return _clean_intro(sent)
        sents = _split_sentences(para_text)
        if sents:
            return _clean_intro(sents[0])
    return ""


def extract_meaning_signals(table, latex, eq_id, pre_text, tree):
    """Collect all non-generative meaning signals for one equation.

    Parameters
    ----------
    table : lxml element
    latex : str
        Raw LaTeX of the equation — used for named_eq matching and inline label.
    eq_id : str
    pre_text : str
    tree : lxml tree

    Returns
    -------
    dict
        Signal dict including 'inline_label' key extracted from \\text{...} in
        the equation LaTeX. This is the primary per-equation discriminator when
        sibling equations share section, paragraph, and intro sentence.
    """
    theorem_env, theorem_title = get_theorem_env(table)
    name, contained_section    = get_section_title(table)
    post_text                  = get_post_text(table)
    post_explanation           = get_post_explanation(table)
    # named_eq uses only the immediate paragraph + post "where" clause — NOT the
    # multi-paragraph fallback pre_text — to avoid matching "Hamiltonian" from an
    # unrelated sentence two paragraphs above a scaling equation.
    para_text                  = _get_immediate_pre_text(table)
    named_eq                   = get_named_equation(para_text + " " + post_text, latex)
    # Abbreviation: local direct sentence beats Schwartz-Hearst; S-H is
    # supplemental only (used for audit context, not primary meaning).
    abbrev                     = extract_abbreviation(pre_text)
    intro_sentence             = extract_intro_sentence(pre_text)
    # lead_in_phrase: dangling sentence directly before the equation.
    # These are the phrases extract_intro_sentence rejects ("is given by:",
    # "takes the form:") — they are the strongest synthesis triggers.
    lead_in_phrase             = extract_lead_in_phrase(pre_text)
    cross_ref                  = get_cross_ref_context(eq_id, tree)
    # Inline label: \text{TRS}, \text{PHS†} etc. embedded in the LaTeX itself.
    inline_label               = _extract_inline_label(latex)
    # LHS shape: normalised first token, human-readable name, equation shape.
    lhs_token, lhs_name, shape = _parse_lhs_shape(latex)

    section_is_generic = (
        contained_section.lower().strip() in _GENERIC_SECTIONS
        or bool(re.match(r"^proof\b", contained_section, re.I))
    )

    return {
        "theorem_env":        theorem_env,
        "theorem_title":      theorem_title,
        "named_eq":           named_eq,
        "name":               name,
        "contained_section":  contained_section,
        "section_is_generic": section_is_generic,
        "abbrev":             abbrev,
        "intro_sentence":     intro_sentence,
        "lead_in_phrase":     lead_in_phrase,
        "post_text":          post_text,
        "post_explanation":   post_explanation,
        "cross_ref":          cross_ref,
        "inline_label":       inline_label,
        "lhs_token":          lhs_token,
        "lhs_name":           lhs_name,
        "eq_shape":           shape,
    }


# Symbols safe to name from the LHS alone. Kept small: one-letter symbols like
# F, S, L are too ambiguous and must be inferred from prose context.
_LHS_PHYSICS_NAMES = {
    r"h": "Hamiltonian", r"\\hat{h}": "Hamiltonian",
    r"\\mathcal{h}": "Hamiltonian", r"h_\\mathrm": "Hamiltonian",
    r"\\hat{h}_": "Hamiltonian term",
    r"z": "partition function",
    r"\\rho": "density matrix", r"\\hat{\\rho}": "density matrix",
    r"\\sigma": "density matrix",
    r"p(": "probability", r"\\mathrm{p}(": "probability",
    r"\\mathbb{p}": "probability",
    r"s_\\mathrm{e}": "entropy",
}

# Patterns that identify equation shape from LaTeX structure (checked in order).
# Each entry: (shape_name, regex).  First match wins.
_SHAPE_PATTERNS = [
    # Probability: P(...) = ... or \Pr[...] = ...
    ("probability",
     re.compile(r"^\\(?:mathbb\{P\}|mathrm\{P(?:r)?\}|Pr|P)\s*[\(\[]", re.I)),
    # State evolution: |\psi(t)> = e^{-iHt} |\psi> or \rho(t) = ...
    ("state_evolution",
     re.compile(r"\\(?:ket|bra|psi|Psi)\s*[\{\(].*(?:e\^|\\exp|\\mathrm\{e\})", re.I)),
    # Unitary/time-evolved state
    ("state_evolution",
     re.compile(r"\\(?:ket|bra)\s*\{[^}]*\(t\)", re.I)),
    # State/protocol transformations: ket lines connected by arrows.
    ("state_transform",
     re.compile(r"\\(?:ket|bra)\s*\{.*?(?:\\rightarrow|\\Leftrightarrow|\\to)\b", re.I | re.S)),
    # Bound / inequality: \leq, \geq, <, > as top-level structure
    ("bound_or_inequality",
     re.compile(r"^[^=]*\\(?:leq|geq|le|ge|ll|gg)\b")),
    ("bound_or_inequality",
     re.compile(r"^[^=]*(?:^|[\s\{])(?:<|>)(?![=])\s*[^=]")),
    # Definition: := or \coloneqq or \equiv at top-level (before any = sign)
    ("definition",
     re.compile(r":=|\\coloneqq|\\triangleq|\\stackrel\{\\Delta\}\{=\}")),
    # Hamiltonian decomposition: H = ... + ... (sum of terms)
    ("hamiltonian_decomposition",
     re.compile(r"^(?:H\b|\\hat\{H\}|\\mathcal\{H\}|\\hat\{\\mathcal\{H\}\})")),
    # Operator action: (A \otimes B)|psi> = ...
    ("operator_action",
     re.compile(r"\\(?:otimes|cdot|circ)\s*.*\\(?:ket|bra|rangle|langle)", re.I)),
    # Density matrix evolution (master equation shape): \dot{\rho} or d\rho/dt
    ("master_equation",
     re.compile(r"\\dot\{\\(?:rho|sigma|varrho)\}|\\frac\{d\\(?:rho|sigma)\}", re.I)),
    # Expectation value: \langle ... \rangle = ...
    ("expectation_value",
     re.compile(r"^\\(?:langle|left\\langle|langle\\!)", re.I)),
    # General assignment: LHS = RHS (fallback)
    ("assignment",
     re.compile(r"[^<>!]=(?!=)")),
]


def _parse_lhs_shape(latex):
    """Parse equation LaTeX to extract LHS symbol and classify equation shape.

    Does NOT parse full LaTeX — uses targeted regex patterns to cover the most
    common structures in quantum physics papers without a full parser.

    Parameters
    ----------
    latex : str
        Raw LaTeX of the equation (may be truncated to ~120 chars in signals).

    Returns
    -------
    tuple[str, str, str]
        (lhs_token, lhs_name, shape) where:
        - lhs_token: normalised first token of LHS (e.g. "rho", "H", "psi")
        - lhs_name: human-readable name if recognisable (e.g. "density matrix"),
          else ""
        - shape: one of the _SHAPE_PATTERNS names or "unknown"
    """
    if not latex:
        return "", "", "unknown"

    # Normalise: strip outer whitespace, collapse double-backslashes.
    lat = latex.strip()

    # Extract LHS: everything before the first =, :=, \leq, \geq, < or >.
    # Use a non-greedy match that stops at the first relation symbol.
    lhs_match = re.match(
        r"^(.*?)(?::=|\\coloneqq|\\triangleq|\\leq|\\geq|\\le\b|\\ge\b|(?<![!<>])=(?!=)|<(?!=)|>(?!=))",
        lat
    )
    lhs_raw = lhs_match.group(1).strip() if lhs_match else lat[:40]

    # Normalise lhs_raw to a short token: strip spaces, braces, backslash.
    lhs_token = re.sub(r"[\\{}\s\^\_\(\)]", "", lhs_raw).lower()
    # Truncate to first 20 chars to avoid capturing full expressions.
    lhs_token = lhs_token[:20]

    # Look up human-readable name for known LHS tokens.
    lhs_name = ""
    for pat, name in _LHS_PHYSICS_NAMES.items():
        norm_pat = re.sub(r"[\\{}\s]", "", pat).lower()
        # One-letter priors are only safe on exact match.  Prefix matching made
        # \hat{t}, \hat{Z}, \hat{B} look like H/Hamiltonian.
        if lhs_token == norm_pat or (len(norm_pat) > 2 and lhs_token.startswith(norm_pat)):
            lhs_name = name
            break

    # Classify shape.
    shape = "unknown"
    for shape_name, pat in _SHAPE_PATTERNS:
        if pat.search(lat):
            shape = shape_name
            break

    return lhs_token, lhs_name, shape


def _lhs_matches_prose_symbol(lhs_token, prose_symbol):
    """Check whether a symbol mentioned in prose matches the equation LHS.

    Used as an LHS guard: before using a prose definition like "let X be the
    number operator" as the meaning of an equation, verify that X matches the
    equation LHS.  Prevents using a neighbouring symbol definition for the
    wrong equation.

    Parameters
    ----------
    lhs_token : str
        Normalised LHS token from _parse_lhs_shape (lowercase, no backslash).
    prose_symbol : str
        LaTeX symbol string extracted from the prose (e.g. r"\\hat{n}").

    Returns
    -------
    bool
    """
    if not lhs_token or not prose_symbol:
        return True  # no guard possible — allow
    norm = re.sub(r"[\\{}\s\^\_]", "", prose_symbol).lower()
    # Accept if tokens share a common normalised prefix of length >= 1.
    min_len = max(1, min(len(lhs_token), len(norm)))
    return lhs_token[:min_len] == norm[:min_len]


def _latex_display_name(latex, lhs_token=""):
    """Return a compact human-readable LHS symbol for meaning templates."""
    if not latex:
        return ""
    lhs_match = re.match(
        r"^(.*?)(?::=|\\coloneqq|\\triangleq|\\leq|\\geq|\\le\b|\\ge\b|(?<![!<>])=(?!=)|<(?!=)|>(?!=))",
        latex.strip()
    )
    lhs = lhs_match.group(1).strip() if lhs_match else ""
    if not lhs:
        return ""
    # Keep simple math names readable; avoid emitting long expressions.
    if len(lhs) > 45 or lhs.count("\\") > 4:
        return lhs_token or ""
    return re.sub(r"\s+", " ", lhs).strip(" ,;:")


def _contextual_lhs_meaning(context, latex, lhs_token, eq_shape):
    """No topic-level meaning prior.

    Earlier versions used a large domain-keyword cascade here. That caused the
    pipeline to label equations by nearby topic words instead of by evidence
    attached to the equation. Meaning is now derived only by the generic
    lead-in, post-text, and cross-reference rules in _synthesize_meaning and
    build_meaning.
    """
    return "", ""


def _subject_hint(contained_section, named_eq, inline_label):
    """Return the most specific available label for template subject slots."""
    if inline_label:
        return inline_label
    if named_eq:
        return named_eq
    return "the expression"


def _extract_subject(sent, trigger_match):
    """Extract the subject of a sentence up to the trigger word position.

    Keeps only the last clause before the trigger so that long preambles like
    "For practical reasons, rather than ... the averaged QMI is therefore" yield
    "the averaged QMI" rather than the full preamble.

    Parameters
    ----------
    sent : str
    trigger_match : re.Match

    Returns
    -------
    str
        Subject phrase, or empty string when extraction fails.
    """
    prefix = sent[:trigger_match.start()].strip(" ,;:")
    # Take the last comma-delimited clause as the immediate subject.
    clauses = [c.strip() for c in prefix.split(",") if c.strip()]
    subj = clauses[-1] if clauses else prefix
    # Drop common low-information openers that bleed into the subject.
    subj = re.sub(
        r"^(?:In this (?:section|paper|work)|We (?:now|here|then|also|further|first|next)|"
        r"Note that|Observe that|It (?:follows|is easy to see) that|"
        r"Furthermore|Moreover|However|Therefore|Thus|Hence|Here|As a result)\b[,\s]*",
        "", subj, flags=re.I
    ).strip(" ,;:")
    if _bad_subject(subj):
        return ""
    # When subject is too long (> 10 words), try to take only the final noun
    # phrase by splitting on prepositions that signal clause boundaries.
    # Example: "The master equation for the density matrix of the full system"
    # → split at "for" or "of" → take last segment → "the full system" (too short)
    # → fall back to last 6 words of the original prefix.
    if len(subj.split()) > 10:
        # Try last comma-free clause up to 8 words.
        # First, find if there's a clear NP before a long prepositional chain.
        np_head = re.match(
            r"^(?:the|a|an)?\s*([\w\-][\w\s\-]{1,40}?)(?:\s+(?:for|of|in|at|with|by|to)\b)",
            subj, re.I
        )
        if np_head:
            cand = np_head.group(1).strip()
            if 1 <= len(cand.split()) <= 7:
                return "" if _bad_subject(cand) else cand
        # Fallback: last 6 words of prefix.
        words = subj.split()
        cand = " ".join(words[-6:]) if len(words) > 6 else ""
        return "" if _bad_subject(cand) else cand
    return subj


def _bad_subject(subj):
    """Reject connector/math-fragment subjects produced by broad templates."""
    if not subj:
        return True
    s = re.sub(r"\s+", " ", subj).strip(" ,;:.").lower()
    if not s:
        return True
    if re.fullmatch(r"(?:which|that|this|these|those|following|above|below|same|result|expression|equation|formula)", s):
        return True
    if re.match(r"^(?:which|that|where|such|so that|as a result|the following|following|we see|it|and|or|but|for a system|our results)\b", s):
        return True
    if re.search(r"\b(?:we see that the number|our results also hold|as written in the main text)\b", s):
        return True
    # Math-fragment leftovers from LaTeXML text, e.g. "\mathbf{k} right at time t".
    if "\\" in subj and len(re.findall(r"[A-Za-z]{3,}", subj)) < 2:
        return True
    if re.search(r"right\s*\\?\}?\$?\s+at\s+time", s):
        return True
    return False


def _bad_meaning_text(text):
    """Reject malformed template outputs before they enter final JSON."""
    if not text:
        return True
    t = re.sub(r"\s+", " ", text).strip()
    bad = [
        r"^Gives the (?:it|we\b|our\b|for a system\b)",
        r"^Gives (?:and|it|we\b|our\b)",
        r"^Gives the such\b",
        r"^Gives such\b",
        r"^Gives the .*our results also hold",
        r"^Gives the .*we see that the number",
        r"^Specifies attractive su\.?$",
        r"^Gives defining\b",
        r"^Defines the Here\b",
        r"^Gives a vector network analyzer\b",
        r"^Gives accounting for\b",
        r"^Gives from the\b",
        r"^Gives bose-hubbard system\.?$",
        r"^Gives (?:the )?in dimensionless",
        r"^Bounds .*\$.*bad\b",
    ]
    return any(re.search(p, t, re.I) for p in bad)


def _tpl_introduces(m, evidence, hint):
    """Handles 'studies/examines/considers the X'. Extracts the NP after the verb,
    stopping at the first parenthesis, dollar sign, or preposition boundary."""
    after_verb = evidence[m.end():].strip()
    after_verb = re.sub(r"^(?:the|a|an)\s+", "", after_verb, flags=re.I)
    np_match = re.match(
        r"([\w\s\-]{3,50}?)(?:\s*[\(\$]|\s+(?:between|of|for|with|in|from|and)\b)",
        after_verb, re.I
    )
    if np_match:
        subj = np_match.group(1).strip(" ,;:()")
        if 1 <= len(subj.split()) <= 8:
            return f"Defines the {subj.lower()}."
    return "" if hint == "the expression" else f"Defines {hint}."


def _tpl_presents(m, evidence, hint):
    """Handles 'write down / present / formulate / propose the X'. Extracts the
    object NP after the trigger; prefers physics terms when multiple NPs are found."""
    after_trigger = evidence[m.end():]

    _PHYSICS_PRIORITY = re.compile(
        r"\b(?:equation|equations|hamiltonian|lagrangian|master|lindblad|"
        r"wave function|functional|action|propagator|partition function|"
        r"entropy|fidelity)\b",
        re.I,
    )
    # Step 1: try to grab the NP that starts immediately after the trigger
    # (before any "at / for / in / by" boundary).  This catches "present the"
    # consuming the article so after_trigger = "quantum master equation at time t...".
    direct_np = re.match(
        r"\s*([\w][\w\s\-]{1,50}?)(?:\s*[\(\$\[]|\s+(?:at|for|in|of|with|by|using|to|from|that|which|where|and|more)\b|[,\.:]|$)",
        after_trigger, re.I
    )
    def _clean_np(np_str):
        """Strip leading article."""
        np_str = re.sub(r"^(?:the|a|an)\s+", "", np_str, flags=re.I).strip(" ,;:()")
        return np_str

    _COMMON_OPENERS = re.compile(
        r"^(?:quantum|classical|effective|total|full|generalized|general|"
        r"time-dependent|time|dependent|following|resulting|corresponding|"
        r"above|below|new|final|initial)\b",
        re.I
    )

    def _normalise_np(s):
        """Lowercase first word when it is a common adjective, preserve proper nouns."""
        if not s:
            return s
        words = s.split()
        if _COMMON_OPENERS.match(words[0]):
            return s[0].lower() + s[1:]
        if words[0].isupper():
            return s
        return s

    if direct_np:
        obj = _clean_np(direct_np.group(1))
        if obj and not re.match(r"^(?:same|above|following|problem|derivation)\b", obj, re.I):
            if 1 <= len(obj.split()) <= 7:
                return f"Presents the {_normalise_np(obj)}."

    # Step 2: scan for "the NP" further in the text, prefer physics terms.
    best_obj = None
    for the_np in re.finditer(
        r"\bthe\s+([\w][\w\s\-]{1,50}?)(?:\s*[\(\$\[]|\s+(?:at|for|in|of|with|by|using|to|from|that|which|where|and|more)\b|[,\.:]|$)",
        after_trigger, re.I
    ):
        obj = _clean_np(the_np.group(1))
        if not obj or re.match(r"^(?:same|above|following|problem|derivation)\b", obj, re.I):
            continue
        if 1 <= len(obj.split()) <= 7:
            if best_obj is None:
                best_obj = obj
            if _PHYSICS_PRIORITY.search(obj):
                best_obj = obj
                break
    if best_obj:
        return f"Presents the {_normalise_np(best_obj)}."
    return "" if hint == "the expression" else f"Presents {hint}."


def _tpl_implements(m, evidence, hint):
    """Handles 'implements / leads to / having the Hamiltonian / satisfied by'. Uses
    the physics noun embedded in the trigger when present; otherwise extracts the object NP."""
    trigger_span = m.group(0)
    embedded = re.search(
        r"\b((?:time-dependent\s+)?(?:Hamiltonian|equation|master\s+equation|Lagrangian|model|functional))\s*$",
        trigger_span, re.I
    )
    if embedded:
        return f"Gives the {embedded.group(1).lower()}."

    # Otherwise extract from the text following the trigger.
    after_verb = evidence[m.end():].strip()
    after_verb = re.sub(r"^(?:the|a|an)\s+", "", after_verb, flags=re.I)
    np_match = re.match(
        r"([\w][\w\s\-]{2,50}?)(?:\s*[\(\$\[\{]"
        r"|\s+(?:described|given|governed|modeled|defined|obtained|derived|satisf)\b"
        r"|\s+(?:at|for|in|of|with|by|using|to|from|that|which|where|and)\b"
        r"|[,\.\[]|$)",
        after_verb, re.I
    )
    if np_match:
        obj = np_match.group(1).strip(" ,;:()")
        if 1 <= len(obj.split()) <= 7:
            return f"Gives {obj.lower()}."
    return "" if hint == "the expression" else f"Gives {hint}."


def _tpl_written_as(m, evidence, hint):
    """Handles 'can be expressed/written as' when a clear subject precedes the trigger."""
    subj = _extract_subject(evidence, m)
    if subj and 1 <= len(subj.split()) <= 8:
        if subj[0].isupper() and not subj.split()[0].isupper():
            subj = subj[0].lower() + subj[1:]
        return f"Gives {subj}."
    return "" if hint == "the expression" else f"Gives {hint}."


def _tpl_fitted_with(m, evidence, hint):
    """Handles 'fitted/approximated with Y' and 'used to estimate Z'.
    Extracts the model name or the estimated quantity depending on the sub-pattern."""
    trigger = m.group(0).lower()

    # Sub-case: "used to estimate/compute/extract Z"
    if "used to" in trigger:
        purpose = re.search(
            r"\bused\s+to\s+(?:estimate|extract|compute|determine|calculate|fit|measure)\s+"
            r"(?:the\s+)?([\w][\w\s\-]{2,40}?)(?:\s*[,\.\(\$]|\s+(?:of|from|in|for|at|and|or)\b|$)",
            evidence, re.I
        )
        if purpose:
            obj = purpose.group(1).strip(" ,;:()")
            if 1 <= len(obj.split()) <= 7:
                return f"Gives the {obj.lower()}."

    # Sub-case: "fitted/approximated with/by a Y [to estimate Z]"
    # Extract Y (the function/model name), stopping before "to" clause.
    model = re.search(
        r"(?:fitted?|approximated?|modeled?)\s+(?:with|by|using)\s+(?:a|an|the)?\s*"
        r"([\w][\w\s\-]{2,40}?)(?:\s+to\s+|\s*[,\.\(\$]|$)",
        evidence, re.I
    )
    if model:
        func = model.group(1).strip(" ,;:()")
        if 1 <= len(func.split()) <= 6:
            # Also extract what it estimates when "to estimate Z" is present.
            purpose_m = re.search(
                r"\bto\s+(?:estimate|extract|compute|determine|calculate|fit|measure)\s+"
                r"(?:the\s+)?([\w][\w\s\-]{2,40}?)(?:\s*[,\.\(]|\s+(?:of|from|in|for|at|and)\b|$)",
                evidence, re.I
            )
            if purpose_m:
                qty = purpose_m.group(1).strip(" ,;:()")
                if 1 <= len(qty.split()) <= 6:
                    # Preserve capitalisation of proper-noun function names (Gaussian, Lorentzian).
                    return f"Gives the {func} used to estimate the {qty.lower()}."
            return f"Gives the {func}."

    return "" if hint == "the expression" else f"Gives {hint}."


def _tpl_define(m, evidence, hint):
    """Handles 'we define/denote X as', 'let X be', 'X is defined as'."""
    define_obj = re.search(
        r"(?:we\s+(?:define|denote|introduce|call|write|express)\s+(?:the\s+)?)(.+?)"
        r"(?:\s+as\b|\s+by\b|\s+to\s+be\b)",
        evidence, re.I
    )
    if define_obj:
        obj = define_obj.group(1).strip(" ,;:()")
        # Trim trailing LaTeX artifacts and citation markers.
        obj = re.sub(r"\s*\[\s*[\d,;\s]*\]\s*$", "", obj).strip()
        obj = re.sub(r"^(?:the|a|an)\s+", "", obj, flags=re.I)
        if 2 <= len(obj.split()) <= 8:
            return f"Defines the {obj.lower()}."
    # "X is defined as" / "X denotes"
    subj_def = re.search(
        r"^(.+?)\s+(?:is\s+defined\s+(?:as|by)|denotes|denote|represents?|stands? for)\b",
        evidence, re.I
    )
    if subj_def:
        subj = subj_def.group(1).strip()
        # Strip leading article "The/A/An".
        subj_stripped = re.sub(r"^(?:the|a|an)\s+", "", subj, flags=re.I).strip()
        # Reject overly long subjects that bleed into surrounding prose.
        if 1 <= len(subj_stripped.split()) <= 6:
            # Preserve original capitalisation for proper nouns (Gibbs, Hamiltonian…).
            # Only lowercase the whole thing when all words are lowercase already.
            if subj_stripped[0].isupper() and not subj_stripped.isupper():
                return f"Defines the {subj_stripped}."
            return f"Defines the {subj_stripped.lower()}."
    # "let X be"
    let_be = re.search(r"let\s+(.+?)\s+be\b", evidence, re.I)
    if let_be:
        obj = let_be.group(1).strip()
        if 1 <= len(obj.split()) <= 6:
            return f"Defines {obj}."
    # "we choose/use/adopt the X ordering/representation/convention..."
    choose_obj = re.search(
        r"\bwe\s+(?:choose|use|adopt|make\s+use\s+of)\s+the\s+([\w\s\-]{2,40}?)"
        r"(?:\s+(?:for|in|to|such|where|with|so|that|which)\b|\s*[,\.\$\(]|$)",
        evidence, re.I
    )
    if choose_obj:
        obj = choose_obj.group(1).strip(" ,;:()")
        if 1 <= len(obj.split()) <= 6:
            return f"Defines the {obj.lower()}."
    return "" if hint == "the expression" else f"Defines {hint}."


def _tpl_gives(m, evidence, hint):
    """Handles 'X is therefore/given by'. Extracts the quantity being given."""
    subj = _extract_subject(evidence, m)
    if subj and 1 <= len(subj.split()) <= 8:
        if subj[0].isupper() and not subj.split()[0].isupper():
            subj = subj[0].lower() + subj[1:]
        return f"Gives {subj}."
    return "" if hint == "the expression" else f"Gives {hint}."


def _tpl_matrix_rep(m, evidence, hint):
    """Handles 'has matrix representation' / 'matrix representation of X'."""
    subj = _extract_subject(evidence, m)
    if subj and 1 <= len(subj.split()) <= 6:
        return f"Gives the matrix representation of {subj}."
    return "" if hint == "the expression" else f"Gives the matrix representation of {hint}."


def _tpl_state(m, evidence, hint):
    """Handles 'the state becomes' / 'state of the system after'."""
    if re.search(r"\bgibbs state\b", evidence, re.I):
        return "Defines the Gibbs state at inverse temperature beta."
    # Subject is the state variable; condition is what precedes.
    subj = _extract_subject(evidence, m)
    if subj and 1 <= len(subj.split()) <= 8:
        if re.match(r"^(?:the|a|an)$", subj.strip(), re.I):
            return "Gives the resulting state."
        return f"Gives the resulting state after {subj}."
    return "Gives the resulting state."


def _tpl_probability(m, evidence, hint):
    """Handles 'occurs with probability' / 'probability of measuring'."""
    event = re.search(
        r"probability\s+(?:of\s+(?:measuring\s+|finding\s+|obtaining\s+)?|that\s+)"
        r"([\w\$\\\{\}\^\_ ]{2,50}?)"
        r"(?:\s+(?:is|are|in|at|by|for|given|with|when|if|where)\b|[,\.\(\)]|$)",
        evidence, re.I
    )
    if event:
        ev = event.group(1).strip(" ,;.()")
        ev = re.sub(r"\s*\[\s*[\d,;\s]*\]\s*$", "", ev).strip()
        if 1 <= len(ev.split()) <= 8:
            return f"Gives the probability of {ev}."
    return f"Gives the probability of the outcome."


def _tpl_sum_average(m, evidence, hint):
    """Handles 'average of X over' / 'averaged X as'. Strips trailing 'as' artifact."""
    avg_obj = re.search(
        r"(?:we\s+define\s+the\s+average(?:d)?\s+|average(?:d)?\s+(?:of\s+)?|mean\s+of\s+)"
        r"(.+?)"
        r"(?:\s+as\b|\s+over\b|\s+across\b|\s+for\b|\s+with\b|\.|,|$)",
        evidence, re.I
    )
    if avg_obj:
        obj = avg_obj.group(1).strip(" ,;:()")
        obj = re.sub(r"\s+as$", "", obj, flags=re.I).strip()
        if 1 <= len(obj.split()) <= 8:
            return f"Defines the averaged {obj} over all fractions."
    return "" if hint == "the expression" else f"Defines the averaged {hint}."


def _tpl_specifies(m, evidence, hint):
    """Handles 'where the system is' / 'consider a X' / 'the initial state is'."""
    # Only extract from "consider a/an/the X" pattern — avoids malformed output
    # from "where the system is" trigger which has no extractable object noun.
    np = re.search(
        r"\bconsider\s+(?:a|an|the)\s+([\w\s\-]{3,40}?)(?:\s*[,\.\$\(]|\s+(?:with|for|in|that|which)\b|$)",
        evidence, re.I
    )
    if np:
        subj = np.group(1).strip(" ,;:()")
        # Reject if it starts with a relative/conjunction word.
        if subj and not re.match(r"^(?:where|which|that|who|when)\b", subj, re.I):
            if 1 <= len(subj.split()) <= 6:
                return f"Specifies {subj.lower()}."
    return "" if hint == "the expression" else f"Specifies {hint}."


def _tpl_modeled_as(m, evidence, hint):
    """Handles 'X is modeled as / described by / governed by'. Rejects purely generic
    subjects (dynamics, behavior, system) so named_eq fallback can handle them."""
    _GENERIC_WORDS = {
        "the", "a", "an", "its", "such", "this", "that", "of", "for",
        "dynamics", "dynamic", "dynamical", "behavior", "behaviour",
        "evolution", "process", "system", "such", "entire", "full",
        "general", "overall", "temporal", "time",
    }
    subj = _extract_subject(evidence, m)
    if subj and 1 <= len(subj.split()) <= 8:
        subj_words = set(re.sub(r"[^a-z\s]", "", subj.lower()).split())
        if subj_words and subj_words.issubset(_GENERIC_WORDS):
            return ""  # purely generic subject — let named_eq_fallback handle
        if subj[0].isupper() and not subj.split()[0].isupper():
            subj = subj[0].lower() + subj[1:]
        return f"Gives {subj}."
    return "" if hint == "the expression" else f"Gives {hint}."


def _tpl_takes_form(m, evidence, hint):
    """Handles 'takes the following form' / 'has the form' / 'the X reads'."""
    subj = _extract_subject(evidence, m)
    if subj and 1 <= len(subj.split()) <= 8:
        if subj[0].isupper() and not subj.split()[0].isupper():
            subj = subj[0].lower() + subj[1:]
        # Add "the" when subject starts with a common noun (not already has article).
        if not re.match(r"^(?:the|a|an)\s", subj, re.I):
            return f"Gives the {subj}."
        return f"Gives {subj}."
    return "" if hint == "the expression" else f"Gives {hint}."


def _tpl_computed_from(m, evidence, hint):
    """Handles 'is obtained from / computed from / derived from'."""
    subj = _extract_subject(evidence, m)
    if subj and 1 <= len(subj.split()) <= 8:
        if subj[0].isupper() and not subj.split()[0].isupper():
            subj = subj[0].lower() + subj[1:]
        return f"Gives {subj}."
    return "" if hint == "the expression" else f"Gives {hint}."


# Ordered rule table: (rule_name, trigger_regex, template_fn).
# More specific patterns before more generic ones to avoid early false matches.
_MEANING_RULES = [
    (
        "matrix_rep",
        re.compile(r"\bhas\s+(?:\w+\s+)*matrix\s+representation\b|\bmatrix\s+representation\b", re.I),
        _tpl_matrix_rep,
    ),
    (
        "probability",
        re.compile(
            r"\boccurs?\s+with\s+probability\b|\bprobability\s+of\s+(?:measuring|finding|obtaining|a)\b"
            r"|\bfound\s+(?:with|as)\s+probability\b",
            re.I,
        ),
        _tpl_probability,
    ),
    (
        "state_becomes",
        re.compile(
            r"\bstate\s+becomes\b|\bstate\s+of\s+the\s+system\s+(?:after|becomes)\b"
            r"|\bafter\s+(?:each|the|all)\b.{0,60}\bstate\s+(?:is|becomes|reads)\b",
            re.I,
        ),
        _tpl_state,
    ),
    (
        "average_define",
        re.compile(
            r"\bwe\s+define\s+the\s+average(?:d)?\b|\baveraged?\s+.{0,40}\s+as\b"
            r"|\baverage\s+of\b.{0,60}\bover\b",
            re.I,
        ),
        _tpl_sum_average,
    ),
    (
        "takes_form",
        re.compile(
            r"\btakes?\s+the\s+(?:following\s+)?form\b"
            r"|\bhas\s+the\s+(?:following\s+)?form\b"
            r"|\breads\s*[:\.]?\s*$"
            r"|\bcan\s+be\s+(?:written|expressed)\s+as\b"
            r"|\bbecomes\s*:"
            r"|\b(?:is|are)\s*[,:\.]?\s*$",
            re.I,
        ),
        _tpl_takes_form,
    ),
    (
        "modeled_as",
        re.compile(
            r"\b(?:is|are)\s+(?:modeled|modelled)\s+as\b"
            r"|\b(?:is|are|can\s+be)\s+described\s+by\b"
            r"|\b(?:is|are)\s+governed\s+by\b"
            r"|\b(?:is|are)\s+represented\s+by\b"
            r"|\b(?:is|are)\s+characterized\s+by\b"
            r"|\b(?:is|are)\s+generated\s+by\b"
            r"|\b(?:is|are)\s+(?:quantified|captured|encapsulated)\s+by\b"
            r"|\b(?:is|are)\s+(?:expressed|given)\s+in\s+terms\s+of\b",
            re.I,
        ),
        _tpl_modeled_as,
    ),
    (
        "computed_from",
        re.compile(
            r"\bis\s+obtained\s+(?:from|by)\b"
            r"|\bis\s+computed\s+(?:from|by)\b"
            r"|\bcan\s+be\s+(?:calculated|computed|obtained|derived)\b"
            r"|\bis\s+derived\s+(?:from|by)\b"
            r"|\bis\s+found\s+to\s+be\b"
            r"|\bare\s+found\s+to\s+be\b",
            re.I,
        ),
        _tpl_computed_from,
    ),
    (
        "introduces",
        re.compile(
            r"\b(?:studies|examines|investigates|introduces|"
            r"quantifies|captures|characterises|characterizes)\s+the\b",
            re.I,
        ),
        _tpl_introduces,
    ),
    (
        "presents",
        re.compile(
            r"\b(?:write[s]?\s+down|write[s]?\s+the|present[s]?\s+the|"
            r"formulate[s]?\s+the|formulate[s]?\s+a\b|propose[s]?\s+the|"
            r"propose[s]?\s+a\b|show[s]?\s+the|derive[s]?\s+the)\b",
            re.I,
        ),
        _tpl_presents,
    ),
    (
        "implements",
        re.compile(
            r"\bimplements?\s+(?:a|an|the)\b"
            r"|\bleads?\s+to\s+(?:a\s+tractable|an?\s+\w+\s*)?\s*(?:Hamiltonian|equation|model|formula|expression|functional)\b"
            r"|\bhaving\s+the\s+(?:time-dependent\s+)?(?:Hamiltonian|equation|master\s+equation|Lagrangian)\b"
            r"|\bcharacterized\s+by\s+the\s+(?:Hamiltonian|equation|Lagrangian)\b"
            r"|\bsatisfied\s+by\b"
            r"|\baccounted?\s+for\s+by\b",
            re.I,
        ),
        _tpl_implements,
    ),
    (
        "written_as",
        re.compile(
            r"\bcan\s+be\s+(?:expressed|written|rewritten|cast)\s+as\b"
            r"|\bmay\s+be\s+(?:expressed|written|rewritten)\s+as\b",
            re.I,
        ),
        _tpl_written_as,
    ),
    (
        "fitted_with",
        re.compile(
            r"\bfitted?\s+(?:with|by|using)\b"
            r"|\bapproximated?\s+(?:by|with|using)\b"
            r"|\bmodeled?\s+(?:by|with|using)\s+a\b"
            r"|\bused\s+to\s+(?:estimate|extract|compute|determine|calculate|fit|measure)\b",
            re.I,
        ),
        _tpl_fitted_with,
    ),
    (
        "define",
        re.compile(
            r"\bwe\s+(?:define|denote|introduce|call|write|express)\b"
            r"|\blet\s+\S+\s+be\b"
            r"|\bis\s+defined\s+(?:as|by)\b|\bdenotes?\b"
            r"|\bwe\s+define\b|\bdefined\s+(?:as|by)\b"
            r"|\bwe\s+use\s+the\s+(?:notation|convention)\b"
            r"|\bwe\s+(?:use|adopt|choose)\s+the\b"
            r"|\bwe\s+make\s+use\s+of\s+the\b"
            r"|\bcoloneqq\b",
            re.I,
        ),
        _tpl_define,
    ),
    (
        "gives",
        re.compile(
            r"\bis\s+therefore\b"
            r"|\bis\s+given\s+by\b"
            r"|\bwe\s+(?:obtain|get|find|arrive\s+at)\b"
            r"|\bgenerates?\s+(?:a|an|the)\b"
            r"|\bgives?\s+rise\s+to\b",
            re.I,
        ),
        _tpl_gives,
    ),
    (
        "specifies",
        re.compile(
            r"\bwhere\s+the\s+(?:system|state|model)\s+is\b"
            r"|\bconsider\s+(?:a|an|the)\b"
            r"|\bthe\s+(?:initial|final|output|target)\s+state\s+is\b",
            re.I,
        ),
        _tpl_specifies,
    ),
]


def _mine_post_text_lhs(post_text, latex):
    """Match 'where $SYM$ is DESCRIPTION' in post_text where SYM is the equation LHS.
    Returns the description phrase, or empty string when the pattern does not match."""
    if not post_text or not latex:
        return ""
    lhs_match = re.match(r"\s*([\\]?[A-Za-z{}^_\{\}]+)", latex.strip())
    if not lhs_match:
        return ""
    lhs_raw = lhs_match.group(1)
    lhs_norm = re.sub(r"[\\{}\s^_]", "", lhs_raw).lower()
    if len(lhs_norm) < 1:
        return ""

    for m in re.finditer(
        r"\bwhere\s+\$([^$]{1,40})\$\s+(?:is|are)\s+(?:a|an|the\s+)?([A-Za-z][^\$,\.;]{3,60}?)(?:\s*[,\.;$]|and\s+\$|$)",
        post_text, re.I
    ):
        token_raw = m.group(1)
        description = m.group(2).strip()
        token_norm = re.sub(r"[\\{}\s^_]", "", token_raw).lower()
        if token_norm == lhs_norm or (len(lhs_norm) >= 3 and lhs_norm in token_norm):
            description = re.sub(r"\[\s*[\d,;\s]*\]\s*$", "", description).strip()
            description = re.sub(r"\s+", " ", description)
            if 1 <= len(description.split()) <= 10:
                return description
    return ""


def _synthesize_meaning(intro_sentence, lead_in_phrase, post_text, post_explanation,
                        latex, lhs_token, lhs_name, eq_shape,
                        contained_section, named_eq, inline_label):
    """Convert local evidence into a meaning statement via slot-fill templates.

    Evidence priority: lead_in_phrase > intro_sentence > post_explanation >
    first post_text sentence > post_text LHS mining.
    LHS guard on the "define" rule prevents attributing a neighbouring symbol's
    definition to this equation. Returns ("", "none", "") when nothing matches.

    Parameters
    ----------
    intro_sentence : str
    lead_in_phrase : str
    post_text : str
    post_explanation : str
    latex : str
    lhs_token : str
    lhs_name : str
    eq_shape : str
    contained_section : str
    named_eq : str
    inline_label : str

    Returns
    -------
    tuple[str, str, str]
        (synthesized_meaning, rule_name, evidence_sentence)
    """
    hint = _subject_hint(contained_section, named_eq, inline_label)

    post_sents = _split_sentences(post_text) if post_text else []
    first_post = _clean_intro(post_sents[0]) if post_sents else ""

    # lead_in_phrase first: closest prose signal, contains the exact verb phrase.
    evidence_list = []
    for ev in (lead_in_phrase, intro_sentence, post_explanation, first_post):
        if ev and ev not in evidence_list:
            evidence_list.append(ev)

    combined_context = " ".join(evidence_list)
    contextual, contextual_rule = _contextual_lhs_meaning(
        combined_context, latex, lhs_token, eq_shape
    )
    if contextual:
        return contextual, contextual_rule, combined_context[:120]

    for evidence in evidence_list:
        for rule_name, trigger_re, template_fn in _MEANING_RULES:
            m = trigger_re.search(evidence)
            if not m:
                continue
            # LHS guard: if "define" fires on a neighbouring symbol, skip it.
            if rule_name == "define" and lhs_token:
                sym_m = re.search(r"\$([^$]{1,30})\$", evidence[max(0, m.start()-60):m.end()+60])
                if sym_m:
                    if not _lhs_matches_prose_symbol(lhs_token, sym_m.group(1)):
                        continue  # symbol mismatch — skip this evidence string
            result = template_fn(m, evidence, hint)
            if result and not _bad_meaning_text(result):
                return result, rule_name, evidence[:120]

    lhs_desc = _mine_post_text_lhs(post_text, latex)
    if lhs_desc:
        result = f"Gives the {lhs_desc.lower()}."
        if not _bad_meaning_text(result):
            return result, "post_lhs", post_text[:120]

    return "", "none", ""


def build_meaning(signals, symbol_defs, latex=""):
    """Assemble the meaning string from local evidence.

    Priority order: inline label / theorem env, prose synthesis, proof-step,
    lead-in direct quote, section fallback, then cross-ref. The named-equation
    lexicon is not allowed to invent a meaning by itself.

    Parameters
    ----------
    signals : dict
        Mutable signal dict; audit keys are written into it.
    symbol_defs : dict
        Accepted but unused in meaning assembly.
    latex : str

    Returns
    -------
    str
    """
    theorem_title      = signals.get("theorem_title", "")
    theorem_env        = signals.get("theorem_env", "")
    named_eq           = signals.get("named_eq", "")
    contained_section  = signals.get("contained_section", "")
    section_is_generic = signals.get("section_is_generic", False)
    abbrev             = signals.get("abbrev", "")
    intro_sentence     = signals.get("intro_sentence", "")
    lead_in_phrase     = signals.get("lead_in_phrase", "")
    post_text          = signals.get("post_text", "")
    post_explanation   = signals.get("post_explanation", "")
    cross_ref          = signals.get("cross_ref", "")
    inline_label       = signals.get("inline_label", "")
    lhs_token          = signals.get("lhs_token", "")
    lhs_name           = signals.get("lhs_name", "")
    eq_shape           = signals.get("eq_shape", "unknown")

    section_is_proof = bool(re.match(r"^proof\b", contained_section, re.I))

    # Initialise audit metadata.
    signals["_meaning_rule"]      = "none"
    signals["_meaning_evidence"]  = ""
    signals["_meaning_lhs"]       = lhs_token
    signals["_meaning_shape"]     = eq_shape
    signals["_meaning_source"]    = "none"
    signals["_section_fallback"]  = False

    # ---------- Priority 1: per-equation structural identifiers ----------
    id_clause = ""
    if inline_label and theorem_title:
        env_label = theorem_env if theorem_env else "result"
        id_clause = f"[{inline_label}] As a {env_label}: {theorem_title}."
        signals["_meaning_source"] = "inline_label+theorem"
        signals["_meaning_rule"] = "inline_label_theorem"
        signals["_meaning_evidence"] = f"{inline_label}; {theorem_title}"
    elif inline_label:
        id_clause = f"[{inline_label}]"
        signals["_meaning_source"] = "inline_label"
        signals["_meaning_rule"] = "inline_label"
        signals["_meaning_evidence"] = inline_label
    elif theorem_title:
        env_label = theorem_env if theorem_env else "result"
        if re.fullmatch(r"[IVXLCDM]+\.\d+\.?", theorem_title.strip(), re.I):
            id_clause = ""
        elif theorem_env == "definition":
            id_clause = f"This definition gives the {theorem_title}."
        else:
            id_clause = f"As a {env_label}: {theorem_title}."
        signals["_meaning_source"] = "theorem"
        signals["_meaning_rule"] = "theorem_title"
        signals["_meaning_evidence"] = theorem_title

    # ---------- Priority 2: equation-level semantic synthesis ----------
    synth, rule, evidence = _synthesize_meaning(
        intro_sentence, lead_in_phrase, post_text, post_explanation,
        latex, lhs_token, lhs_name, eq_shape,
        contained_section, named_eq, inline_label,
    )
    if synth or rule != "none" or evidence:
        signals["_meaning_rule"]     = rule
        signals["_meaning_evidence"] = evidence

    if synth:
        signals["_meaning_source"] = (
            "lead_in" if (lead_in_phrase and lead_in_phrase[:80] in evidence)
            else "intro" if (intro_sentence and intro_sentence[:80] in evidence)
            else "post" if evidence
            else "shape"
        )

    # ---------- Priority 3: named equation audit only ----------
    named_clause = ""

    # ---------- Priority 4: proof-step fallback ----------
    proof_clause = ""
    if not synth and not named_clause and section_is_proof and contained_section:
        thm_match = re.search(r"proof\s+of\s+(.+)", contained_section, re.I)
        target = thm_match.group(1).strip() if thm_match else contained_section
        proof_clause = f"Intermediate step in proof of {target}."
        signals["_meaning_rule"]   = "proof_step_fallback"
        signals["_meaning_source"] = "proof_section"
        signals["_meaning_evidence"] = contained_section

    # ---------- Priority 4b: lead-in direct emission ----------
    # When synthesis produced nothing (no prose trigger matched), emit the
    # lead-in phrase or intro sentence verbatim. This is more informative than
    # the section title since it contains the actual introducing verb phrase.
    lead_in_direct = ""
    if not synth and not named_clause and not proof_clause:
        # Prefer lead_in_phrase (dangling sentence ending in a display trigger);
        # fall back to intro_sentence when lead_in is absent.
        for candidate in (lead_in_phrase, intro_sentence):
            if not candidate:
                continue
            cleaned = _clean_intro(candidate)
            if cleaned and len(cleaned.split()) >= 4 and not _bad_meaning_text(cleaned):
                # Accept only if the candidate contains at least one content word
                # beyond pure function words — avoids emitting "We have:" alone.
                non_func = re.sub(r"\b(?:we|the|a|an|is|are|have|this|that|for|in|of|to|by|at|as|on)\b", "", cleaned.lower())
                if len(non_func.split()) >= 2:
                    lead_in_direct = cleaned + ("." if not cleaned.endswith(".") else "")
                    signals["_meaning_rule"]   = "lead_in_direct"
                    signals["_meaning_source"] = "lead_in" if (lead_in_phrase and candidate == lead_in_phrase) else "intro"
                    signals["_meaning_evidence"] = cleaned
                    break

    # ---------- Priority 5: section-title fallback ----------
    section_clause = ""
    if not synth and not named_clause and not proof_clause and not lead_in_direct:
        if contained_section and not section_is_generic:
            section_clause = f"From the '{contained_section}' section."
            signals["_meaning_rule"]  = "section_fallback"
            signals["_meaning_source"] = "section_fallback"
            signals["_section_fallback"] = True
            signals["_meaning_evidence"] = contained_section

    # ---------- Assemble final string ----------
    parts = []
    if id_clause:
        parts.append(id_clause)

    meaning_clause = synth or named_clause or proof_clause or lead_in_direct or section_clause
    if meaning_clause:
        parts.append(meaning_clause)

    # Abbreviation is supplemental context, appended only when synthesis also
    # produced something useful (prevents abbrev replacing empty meaning with noise).
    if abbrev and synth:
        parts.append(f"Introduced in the context of {abbrev}.")
        if signals["_meaning_rule"] == "abbrev":
            pass  # already set
        # Don't overwrite a good rule with "abbrev"

    # Cross-reference last resort — only when absolutely nothing else fired.
    if not parts and cross_ref:
        parts.append(f"Referenced as: {cross_ref}")
        signals["_meaning_rule"]   = "cross_ref"
        signals["_meaning_source"] = "cross_ref"
        signals["_meaning_evidence"] = cross_ref

    return " ".join(parts)


def _strip_markers(text):
    """Remove [EQ] / [TARGET] markers and collapse whitespace."""
    text = re.sub(r"\[(?:EQ|TARGET)\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_intro(sent):
    """Strip trailing citation placeholders, math convention notes, and colons.

    Removes:
    - numeric citation brackets like '[23]', '[1,2]'
    - parenthetical math convention notes like '(with $\\hbar=1$)' or '(\\hbar=1)'
      that authors insert right before the equation as a unit setting
    - trailing colon left by setup phrases like 'can be written as:'

    Parameters
    ----------
    sent : str

    Returns
    -------
    str
    """
    sent = re.sub(r"\[\s*[\d,;\s]*\]\s*[,:]?\s*$", "", sent)
    # Strip trailing parenthetical containing LaTeX (backslash or dollar sign):
    # e.g. "(with $\hbar=1$)", "($\hbar = 1$)", "(\hbar=1)".
    # Limit to 60 chars inside the parens to avoid stripping meaningful clauses.
    sent = re.sub(r"\s*\([^)]*(?:\\|\$)[^)]{0,60}\)\s*$", "", sent)
    # Strip trailing colon or comma left by setup phrases ("can be written as:",
    # "...Lindblad master equation," leading into the equation display).
    sent = re.sub(r"[,:]\s*$", "", sent)
    return re.sub(r"\s+", " ", sent).strip()
