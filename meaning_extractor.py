"""Derive meaning for enumerated equations using a layered non-generative strategy.

Signal extraction priority:
  1. Theorem/Lemma/Definition environment title with a descriptive name.
  2. Section title (when not generic) — primary context clause.
     Named equation lexicon fires as a supplement when the section is specific,
     or as the primary clause when the section is generic/missing.
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
from context_extract import (
    _enclosing_para,
    _prev_paras,
    _next_paras,
    _clean_para,
    _split_sentences,
)

# ---------------------------------------------------------------------------
# Named equation lexicon
# Each entry: (prose_regex, latex_regex, canonical_name).
# prose_regex is case-insensitive and uses \b word boundaries to prevent
# substring false positives (e.g. "interactions" firing "action functional").
# Empty string = skip that field.  Checked in order; first hit wins.
# ---------------------------------------------------------------------------

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
    r'expressed as|represented as|written in the form)\s*[,:]?\s*$'
    r'|\[\s*[\d,;\s]*\]\s*[,:]?\s*$',   # ends with citation + optional colon/comma
    re.I
)

_MIN_PRE_WORDS   = 8
_MIN_INTRO_WORDS = 8

# Phrases that signal the Schwartz-Hearst algorithm returned a prose fragment
# rather than a genuine abbreviation long-form expansion.
_ABBREV_REJECT_RE = re.compile(
    r'\b(?:at best|at least|however|but\b|so that|in order|is not|are not|'
    r'do not|cannot|we note|note that|thus|hence|therefore|although|since|'
    r'valid|available|investigated|possible|presented|applied)\b',
    re.I
)


# ---------------------------------------------------------------------------
# Individual signal extractors
# ---------------------------------------------------------------------------

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
                raw = titles[0].text_content().strip()
                # Strip leading section number: "1", "A.1", "III.2", etc.
                # [A-Z]\.(?:\d+...)? handles compound labels like "A.1".
                # [\s\u00a0]* covers the non-breaking space LaTeXML sometimes inserts.
                contained = re.sub(
                    r"^\s*(?:[IVXLCDM]+\.?[\s\u00a0]*"
                    r"|[A-Z]\.(?:\d+(?:\.\d+)*\.?)?[\s\u00a0]*"
                    r"|\d+(?:\.\d+)*\.?[\s\u00a0]*)"
                    , "", raw
                ).strip() or raw.strip()
                # Second pass: "A." stripped from "A.1 Title" leaves "1 Title".
                contained = re.sub(
                    r"^\d+(?:\.\d+)*\.?[\s\u00a0]*", "", contained
                ).strip() or contained
                words = contained.split()
                while words and words[-1].lower().rstrip(".") in _SECTION_SUFFIX_WORDS:
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
    text = _strip_markers(_clean_para(p, set()))
    if len(text.split()) < _MIN_PRE_WORDS:
        prevs = _prev_paras(para)
        if prevs:
            text = (_strip_markers(_clean_para(prevs[-1], set())) + " " + text).strip()
    return text


def get_post_text(table):
    """Return cleaned 'where...' prose immediately following `table`.

    Many quantum physics papers display the equation first and then write
    "where $X$ is... and $Y$ is..." in the next sentence. That clause is
    the richest source of symbol definitions but lies AFTER the equation,
    outside the reach of get_pre_text().

    Only returns text when 'where' appears in the first 80 characters, so
    unrelated continuation paragraphs are not pulled in.

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
    text = _strip_markers(_clean_para(p, set())).strip()
    # If the same-paragraph tail is thin, also grab the first following paragraph.
    if len(text.split()) < _MIN_PRE_WORDS:
        nexts = _next_paras(para)
        if nexts:
            nxt = _strip_markers(_clean_para(nexts[0], set())).strip()
            text = (text + " " + nxt).strip()
    # Only return if 'where' appears near the start AND the text contains at
    # least one math token ($...$) — confirms this is a symbol definition clause
    # rather than a prose reference like "where the full derivation is in App. A".
    if (text
            and re.search(r"\bwhere\b", text[:80], re.I)
            and re.search(r"\$[^$]+\$", text)):
        return text
    return ""


def extract_abbreviation(pre_text):
    """Extract the most informative abbreviation from pre-table prose via Schwartz-Hearst.

    Parameters
    ----------
    pre_text : str

    Returns
    -------
    str
    """
    if not pre_text or len(pre_text.split()) < _MIN_PRE_WORDS:
        return ""
    try:
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(
            doc_text=pre_text, first_definition=True
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


def extract_intro_sentence(pre_text):
    """Return the last substantive non-dangling sentence from pre_text verbatim.

    Sentences that lead directly into the equation ("can be written as",
    "is given by:", "we have:") are rejected because they produce incomplete
    clauses when the equation itself is stripped out.  Citation placeholders
    and trailing colons are cleaned before the dangling check.

    Parameters
    ----------
    pre_text : str

    Returns
    -------
    str
        Clean introducing sentence, or empty string if none qualifies.
    """
    if not pre_text:
        return ""
    sents = _split_sentences(pre_text)
    for sent in reversed(sents):
        cleaned = _clean_intro(sent)
        if len(cleaned.split()) >= _MIN_INTRO_WORDS and not _DANGLING_RE.search(cleaned):
            return cleaned
    return ""


def get_cross_ref_context(eq_id, tree):
    """Find sentences elsewhere in the document that reference this equation.

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
    for ref in tree.xpath(f'//a[@href="#{eq_id}"]'):
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


# ---------------------------------------------------------------------------
# Signal aggregation
# ---------------------------------------------------------------------------

def extract_meaning_signals(table, latex, eq_id, pre_text, tree):
    """Collect all non-generative meaning signals for one equation.

    Parameters
    ----------
    table : lxml element
    latex : str
    eq_id : str
    pre_text : str
    tree : lxml tree

    Returns
    -------
    dict
    """
    theorem_env, theorem_title = get_theorem_env(table)
    name, contained_section    = get_section_title(table)
    post_text                  = get_post_text(table)
    # named_eq uses only the immediate paragraph + post "where" clause — NOT the
    # multi-paragraph fallback pre_text — to avoid matching "Hamiltonian" from an
    # unrelated sentence two paragraphs above a scaling equation.
    para_text                  = _get_immediate_pre_text(table)
    named_eq                   = get_named_equation(para_text + " " + post_text, latex)
    abbrev                     = extract_abbreviation(pre_text)
    intro_sentence             = extract_intro_sentence(pre_text)
    cross_ref                  = get_cross_ref_context(eq_id, tree)

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
        "post_text":          post_text,
        "cross_ref":          cross_ref,
    }


# ---------------------------------------------------------------------------
# Meaning assembly
# ---------------------------------------------------------------------------

def build_meaning(signals, symbol_defs):
    """Assemble the meaning string from signals in priority order.

    Priority logic:
      - Theorem env title (with descriptive name) → primary clause.
      - Specific section title → primary clause; named_eq fires as supplement.
      - Generic/missing section → named_eq as primary clause.
      - Abbreviation → purpose clause (suppresses intro_sentence to avoid
        redundancy).
      - Intro sentence → included when no abbreviation and sentence is clean.
      - Cross-reference → last resort when both abbrev and intro_sentence absent.

    Symbol definitions are NOT appended here — they belong in the 'symbols'
    field of the JSON schema and repeating them in 'meaning' causes redundancy
    and can propagate contaminated definitions (cross-section leakage) into the
    meaning string.

    Parameters
    ----------
    signals : dict
    symbol_defs : dict
        Accepted but not used in meaning assembly (reserved for callers that
        may need both the string and the dict in one call).

    Returns
    -------
    str
    """
    parts = []

    theorem_title      = signals.get("theorem_title", "")
    theorem_env        = signals.get("theorem_env", "")
    named_eq           = signals.get("named_eq", "")
    name               = signals.get("name", "")
    contained_section  = signals.get("contained_section", "")
    section_is_generic = signals.get("section_is_generic", False)
    abbrev             = signals.get("abbrev", "")
    intro_sentence     = signals.get("intro_sentence", "")
    post_text          = signals.get("post_text", "")
    cross_ref          = signals.get("cross_ref", "")

    # Clause 1: primary identification — phrasing varies by signal type so that
    # all meanings do not open with the same template sentence.
    if theorem_title:
        env_label = theorem_env if theorem_env else "result"
        if theorem_env == "definition":
            parts.append(f"This definition gives the {theorem_title}.")
        else:
            parts.append(f"As a {env_label}: {theorem_title}.")
    elif contained_section and not section_is_generic:
        if named_eq:
            # Specific section with a physics name — combine both.
            parts.append(
                f"This is the {named_eq}, from the '{contained_section}' section."
            )
        elif name and name != contained_section:
            parts.append(f"From '{contained_section}': {name}.")
        else:
            parts.append(f"From the '{contained_section}' section.")
    elif named_eq:
        # Section is generic or absent — physics name is the best primary signal.
        parts.append(f"This is the {named_eq}.")

    # Clause 2: abbreviation context (long-form expansion from Schwartz-Hearst).
    if abbrev:
        parts.append(f"Introduced in the context of {abbrev}.")

    # Clause 3: introducing sentence or post-equation 'where' clause.
    # Pre-text intro sentence is preferred; post_text 'where...' clause is the
    # fallback when the pre-text yields nothing — it still comes verbatim from
    # the paper and satisfies the no-generation rule.
    if intro_sentence and not abbrev:
        parts.append(f"In context: {intro_sentence}")
    elif post_text and not abbrev and not intro_sentence:
        first_post = _split_sentences(post_text)[0] if _split_sentences(post_text) else post_text
        first_post = _clean_intro(first_post)
        if len(first_post.split()) >= _MIN_INTRO_WORDS:
            parts.append(f"In context: {first_post}")

    # Clause 4: cross-reference (only when intro_sentence and post_text also absent).
    if cross_ref and not intro_sentence and not post_text and not abbrev:
        parts.append(f"Referenced as: {cross_ref}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
