"""Extract equation symbols and their text definitions from cached arXiv HTML.

Identifier extraction uses MathML leaves with LaTeX AST fallbacks.
Definition extraction uses a strict post-equation where-clause parser.

Precision over recall: output nothing when the definition is uncertain.
A missing definition is better than a wrong one.


Provenance tags written to _sources when the caller passes the dict:
    'post_where'   — where clause in post_text
    'respectively' — respectively coordinated list
    'pre_explicit' — explicit let/denote/define in last 2 pre_text sentences
"""

import re
from pathlib import Path

from lxml import html as lxml_html
from pylatexenc.latexwalker import (LatexWalker, LatexMacroNode,
                                     LatexCharsNode, LatexGroupNode,
                                     LatexEnvironmentNode)

from context_extract import get_contexts, _split_sentences
from review_equations import extract_equations

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
    "lVert", "rVert", "lvert", "rvert", "Vert", "vert",
    "lfloor", "rfloor", "lceil", "rceil", "lbrace", "rbrace",
    "partial", "limits",
    "bigotimes", "bigoplus", "bigcup", "bigcap", "bigvee", "bigwedge", "biguplus",
    "coprod",
}

IDENTIFIER_STOP = {
    "o",
    "sin", "cos", "tan", "log", "ln", "exp", "tr", "Tr", "trace", "Pr",
    "Re", "Im", "det", "rank", "dim", "mod", "sup", "inf", "limsup",
    "liminf", "argmin", "argmax", "Perm", "Haf", "min", "max", "pi",
}

_BOUND_OPS = {
    "sum", "prod", "int", "oint", "iint", "iiint", "iiiint", "idotsint",
    "lim", "limsup", "liminf", "sup", "inf", "max", "min", "argmin", "argmax",
    "bigcup", "bigcap", "bigvee", "bigwedge", "forall", "exists",
}


def _all_identifiers_in_group(group_node):
    """Return alphabetic bound-variable identifiers from a subscript brace group."""
    found = set()
    raw_chars = []
    for node in group_node.nodelist:
        if isinstance(node, LatexCharsNode):
            raw_chars.append(node.chars)
        elif isinstance(node, LatexMacroNode):
            found.add(node.macroname)
    raw = "".join(raw_chars)
    for part in re.split(r"[,;]", raw):
        m = re.match(r"\s*([A-Za-z]+)", part)
        if m:
            found.add(m.group(1))
    return found


def _collect_bound(nodelist, bound):
    """Recursively collect bound variable names from a pylatexenc node list."""
    if not nodelist:
        return
    i = 0
    while i < len(nodelist):
        node = nodelist[i]
        if isinstance(node, LatexMacroNode):
            if node.macroname in _BOUND_OPS:
                j = i + 1
                while j < len(nodelist):
                    nxt = nodelist[j]
                    if isinstance(nxt, LatexCharsNode) and "_" in nxt.chars:
                        if j + 1 < len(nodelist) and isinstance(nodelist[j + 1], LatexGroupNode):
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
    """Return variable names bound by sum/int/prod/lim etc. in a LaTeX equation.

    A variable is excluded only if it appears as a summation/integration index
    in this equation, not globally. Returns empty set on parse failure.
    """
    try:
        w = LatexWalker(latex)
        nodes, _, _ = w.get_latex_nodes()
        bound = set()
        _collect_bound(nodes, bound)
        return bound
    except Exception:
        return set()


def extract_identifiers(arxiv_id, eq_id, latex):
    """Return normalised identifiers for one equation.

    Combines MathML leaves with LaTeX AST fallbacks and filters out bound
    variables, stop-list entries, and junk tokens. Returns a sorted list
    of no-backslash keys, e.g. ['N', 'mathcal_E', 'rho'].
    """
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


def normalize_identifier(token):
    """Normalize a raw MathML/LaTeX token to a stable symbol key.

    Strips backslashes and braces, maps bare sign subscripts to _plus/_minus,
    and removes non-alphanumeric characters.
    """
    token = token.strip()
    token = token.replace("\\", "")
    token = token.strip("{} ")
    token = re.sub(r"_\+", "_plus", token)
    token = re.sub(r"_-", "_minus", token)
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
            sub_tag  = sub_node.tag if isinstance(sub_node.tag, str) else ""
            sub_text = sub_node.text_content().strip()
            is_bare_sign = (sub_tag.endswith("}mo") or sub_tag == "mo") and sub_text in ("+", "-")
            if is_bare_sign:
                base_tok = _mathml_identifier_token(base_node)
                sign = "plus" if sub_text == "+" else "minus"
                if base_tok:
                    symbols.add(f"{base_tok}_{sign}")
                continue
            if _mathml_node_is_complex(sub_node) or _mathml_node_is_complex(base_node):
                if base_node.tag.endswith("}mi"):
                    mi_list = [base_node]
                else:
                    mi_list = base_node.xpath('.//*[local-name()="mi"]')
                for mi in mi_list:
                    t = mi.text_content().strip()
                    if t:
                        symbols.add(t)
                continue
            base   = _mathml_identifier_token(base_node)
            suffix = _mathml_identifier_token(sub_node)
            combined = _combine_base_subscript(base, suffix)
            if combined:
                symbols.add(combined)

    for mi in root.xpath('.//*[local-name()="mi"]'):
        if mi.xpath('ancestor::*[local-name()="msub" or local-name()="msubsup"]'):
            continue
        parent = mi.getparent()
        if parent is not None:
            ptag = parent.tag if isinstance(parent.tag, str) else ""
            if ptag.endswith("}msup") or ptag == "msup":
                siblings = [c for c in parent if isinstance(c.tag, str)]
                if len(siblings) >= 2 and mi is siblings[1]:
                    continue
        text = mi.text_content().strip()
        if not text:
            continue
        if len(text) == 1:
            prev_sib = mi.getprevious()
            next_sib = mi.getnext()

            def _is_mo_dot(node):
                return (node is not None
                        and isinstance(node.tag, str)
                        and (node.tag == "mo" or node.tag.endswith("}mo"))
                        and node.text_content().strip() == ".")

            if _is_mo_dot(prev_sib) or _is_mo_dot(next_sib):
                continue
        symbols.add(text)

    return symbols


def _mathml_node_is_complex(node):
    """Return True when a MathML node contains operators or multiple identifiers."""
    has_operator = bool(node.xpath('.//*[local-name()="mo"]') or
                        node.tag.endswith("}mo"))
    mi_nodes = node.xpath('.//*[local-name()="mi"]')
    if node.tag.endswith("}mi"):
        mi_nodes = [node] + list(mi_nodes)
    return has_operator or len(mi_nodes) > 1


def _mathml_identifier_token(node):
    """Return a compact token string for a MathML subtree."""
    text = re.sub(r"\s+", "", node.text_content().strip())
    return _clean_tex_token(text)


def _latex_command_identifiers(latex):
    """Collect Greek command identifiers that appear standalone in LaTeX.

    Skips commands used as subscript bases to avoid duplicates with
    _latex_subscript_identifiers (e.g. sigma vs sigma_x).
    """
    symbols = set()
    for cmd in re.findall(r"\\([A-Za-z]+)(?!\s*_)", latex):
        if cmd in GREEK:
            symbols.add(cmd)
    return symbols


def _latex_decorated_identifiers(latex):
    r"""Collect decorated symbols, e.g. \mathcal{D} -> mathcal_D."""
    symbols = set()
    pattern = r"\\(" + "|".join(sorted(DECORATORS, key=len, reverse=True)) + r")\s*\{([^{}]{1,40})\}"
    for deco, body in re.findall(pattern, latex):
        base = _clean_tex_token(body)
        if base and _keep_identifier(base):
            symbols.add(f"{deco}_{base}")
    return symbols


def _latex_subscript_identifiers(latex):
    r"""Collect subscripted identifiers such as p_A, \mathcal{T}_{+}."""
    symbols = set()
    base = r"(?:\\[A-Za-z]+|[A-Za-z])"
    sub  = r"(?:\{[^{}]{1,40}\}|[A-Za-z0-9])"
    for raw_base, raw_sub in re.findall(rf"({base})\s*_\s*({sub})", latex):
        combined = _combine_base_subscript(raw_base, raw_sub)
        if combined:
            symbols.add(combined)

    deco_pat = (
        r"\\(" + "|".join(sorted(DECORATORS, key=len, reverse=True)) + r")"
        r"\s*\{([A-Za-z0-9]{1,10})\}"
        r"\s*_\s*"
        r"(\{[^{}]{1,40}\}|[+\-]|[A-Za-z0-9])"
    )
    for deco, body, raw_sub in re.findall(deco_pat, latex):
        body_clean = _clean_tex_token(body)
        if not body_clean or not _keep_identifier(body_clean):
            continue
        base_key = f"{deco}_{body_clean}"
        raw_sub_strip = raw_sub.strip("{} ")
        if re.search(r"[,\^]", raw_sub_strip):
            continue
        if re.search(r"[+\-]", raw_sub_strip) and len(raw_sub_strip) > 1:
            continue
        if raw_sub_strip in ("+", "-"):
            sign = "plus" if raw_sub_strip == "+" else "minus"
            symbols.add(f"{base_key}_{sign}")
        else:
            sub_clean = _clean_tex_token(raw_sub)
            if sub_clean:
                symbols.add(f"{base_key}_{sub_clean}")

    return symbols


def _latex_simple_identifiers(latex):
    """Collect remaining single-letter identifiers not captured by other passes."""
    symbols = set()
    cleaned = re.sub(r"\\(?:operatorname|mathrm|text|textrm)\{[^{}]*\}", " ", latex)
    cleaned = re.sub(r"\b[Hh]\.c\.", " ", cleaned)
    cleaned = re.sub(r"\^\s*\{[^{}]{1,20}\}", " ", cleaned)
    cleaned = re.sub(r"\^\s*[A-Za-z0-9*†‡]", " ", cleaned)
    deco_group = "|".join(sorted(DECORATORS, key=len, reverse=True))
    cleaned = re.sub(
        rf"\\(?:{deco_group})\s*\{{[^{{}}]{{1,20}}\}}\s*_\s*(?:\{{[^{{}}]*\}}|[A-Za-z0-9+\-])",
        " ", cleaned
    )
    cleaned = re.sub(r"\\[A-Za-z]+\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9])", " ", cleaned)
    cleaned = re.sub(r"[A-Za-z]\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9])", " ", cleaned)
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = re.sub(r"\{[A-Za-z0-9]{1,5}\}", " ", cleaned)
    cleaned = re.sub(r"[A-Za-z]{2,}", " ", cleaned)

    for letter in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", cleaned):
        symbols.add(letter)

    return symbols


def _keep_identifier(symbol, bound=None):
    """Return True when a normalised token is worth keeping as a physics symbol."""
    if bound is None:
        bound = set()
    if not symbol:
        return False
    if symbol in bound:
        return False
    if symbol in COMMAND_STOP:
        return False
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
        if base in IDENTIFIER_STOP or base in COMMAND_STOP:
            return False
        if len(base) > 1 and base.lower() in {s.lower() for s in IDENTIFIER_STOP}:
            return False
        if base in bound:
            return False
    return True


def _looks_like_fused_junk(symbol):
    """Reject tokens that are artefacts of flattening complex expressions."""
    if len(symbol) >= 14 and "_" not in symbol:
        return True
    if len(re.findall(r"[A-Z]", symbol)) >= 4 and len(symbol) >= 8:
        return True
    return False


def _combine_base_subscript(base, suffix):
    """Combine a raw base and subscript token into one normalised symbol key.

    Returns empty string when the subscript is too complex for a clean key.
    """
    raw_suffix = suffix.strip()
    if len(raw_suffix) >= 2 and raw_suffix[0] == "{" and raw_suffix[-1] == "}":
        raw_suffix = raw_suffix[1:-1].strip()
    if re.search(r"[,\^]", raw_suffix):
        return ""
    if "\\" in raw_suffix:
        text_match = re.match(
            r"\\(?:text|mathrm|rm|mathit|mathsf|mathcal|mathbb|mathscr|mathfrak)\{([A-Za-z][A-Za-z0-9]{0,24})\}$",
            raw_suffix.strip(),
        )
        if text_match:
            inner = text_match.group(1)
            deco_m = re.match(r"\\(\w+)\{", raw_suffix.strip())
            deco   = deco_m.group(1) if deco_m else ""
            suffix_key = f"{deco}_{inner}" if deco in {
                "mathcal", "mathbb", "mathscr", "mathfrak"
            } else inner
            if inner not in COMMAND_STOP:
                base_clean = _clean_tex_token(base)
                if base_clean and base_clean not in COMMAND_STOP:
                    return f"{base_clean}_{suffix_key}"
        return ""
    if re.search(r"[+\-]", raw_suffix) and len(raw_suffix.strip()) > 1:
        return ""
    base = _clean_tex_token(base)
    if not base or base in COMMAND_STOP:
        return ""
    if raw_suffix in ("+", "-"):
        sign = "plus" if raw_suffix == "+" else "minus"
        return f"{base}_{sign}"
    suffix = _clean_tex_token(suffix)
    if not suffix or suffix in COMMAND_STOP:
        return ""
    if len(suffix) > 20:
        return ""
    return f"{base}_{suffix}"


def _clean_tex_token(token):
    """Convert a small LaTeX token fragment to a stable symbol fragment."""
    token = token.strip()
    token = re.sub(r"\\(?:rm|mathrm|text|textrm|mathit|mathsf)\b\s*", "", token)
    token = token.strip("{} ")
    token = re.sub(
        r"\\(?:hat|widehat|bar|overline|tilde|vec|bm|boldsymbol|mathcal|mathbb|mathscr|mathfrak|mathbf)\{([^{}]+)\}",
        r"\1", token,
    )
    token = token.replace("\\", "")
    token = re.sub(r"[^A-Za-z0-9]", "", token)
    return token


def _symbol_latex_variants(symbol):
    """Return the set of LaTeX spellings corresponding to one normalised key."""
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


def _symbol_regex(symbol):
    """Build a regex matching a normalised symbol in inline LaTeX.

    Single-char symbols only match inside $...$ to avoid false hits on English
    words. Multi-char symbols also allow bare word matches.
    """
    variants = _symbol_latex_variants(symbol)
    inline = (r"\$\s*(?:" +
              "|".join(sorted(map(re.escape, variants), key=len, reverse=True)) +
              r")\s*\$")
    if len(symbol) == 1:
        return inline
    bare = r"\b" + re.escape(symbol) + r"\b"
    return r"(?:" + inline + r"|" + bare + r")"


_WEAK_DEFS = frozenset({
    "small", "defined as", "given by", "below", "above", "respectively",
    "the following", "as follows", "zero", "one",
})


def _clean_definition(text):
    """Clean and truncate a matched definition span."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,:;-")
    text = re.sub(r"\s*\[(?:TARGET|EQ)\].*$", "", text)
    _SENT_START = (
        r"We|The|This|These|That|Those|Note|Consider|For|In|To|It|Now|"
        r"Thus|Hence|Here|Then|Let|From|As|By|With|Since|Such|One|An|A"
    )
    m_sent = re.search(rf"\.\s*(?={_SENT_START}\b)", text)
    if m_sent:
        text = text[:m_sent.start() + 1]
    text = re.sub(
        r"\s*\(?\s*(?:as defined in|see|cf\.?|in Eq|of Eq)\b.*$",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\s+(?:with|for every|for|such that|where|which|as|are|is|of|"
        r"in|by|to|from|and|or|that|on|if|when)\s*$",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\s+(?:for|in|of|with|by)\s+(?:the|a|an|its|their)\s+\w+\s*$",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\s+(?:is\s+the|are\s+the|is\s+a|are\s+a|the|a|an)\s*$",
        "", text, flags=re.IGNORECASE
    )
    text = text.strip(" ,:;-()")
    text = text.rstrip(".")
    return text[:180].strip()


def _use_definition(desc):
    """Return True when a definition span is worth emitting.

    Rejects clausal fragments, action phrases, and other non-NP spans.
    """
    if not desc:
        return False
    low = desc.lower().strip(" .,:;-")
    if low in _WEAK_DEFS:
        return False
    if len(low.split()) == 1:
        if len(low) <= 4:
            return False
        if re.match(
            r"^(?:the|this|that|these|those|such|some|any|all|both|each|"
            r"other|same|also|very|more|most|less|just|then|when|where|"
            r"which|what|with|from|into|onto|over|under|about|above|"
            r"below|after|before|since|until|while)\b",
            low
        ):
            return False
    if re.match(
        r"^(?:defined\s+as|given\s+by|equal\s+to|expressed\s+as|"
        r"associated\s+with|related\s+to|proportional\s+to|"
        r"obtained\s+by|described\s+by|determined\s+by|"
        r"known\s+as|referred\s+to\s+as|also\s+known|denoted\s+by|"
        r"denoted\s+as|written\s+as)\b",
        low
    ):
        return False
    if len(low.split()) == 1 and "-" in low:
        return False
    if re.search(r"\b(is|are|was|were|equals)\b", low):
        return False
    if re.match(r"^(no\s+longer|not\s+|never\s+|neither\s+)", low):
        return False
    if re.match(
        r"^(using|applying|choosing|taking|making|setting|computing|evaluating)\b", low
    ):
        return False
    if re.match(
        r"^(find|compute|calculate|obtain|derive|ensure|enforce|satisfy|"
        r"require|minimize|maximize|define|consider|describe|determine|"
        r"represent|denote|measure|estimate|solve|build|form|generate)\b", low
    ):
        return False
    return True


_WHERE_START = re.compile(
    r"^(?:where|with|here|in\s+(?:which|this\s+(?:notation|expression|case)))\b",
    re.IGNORECASE,
)

_REL_VERB = re.compile(
    r"\b(?:is|are|denotes?|represents?|stands?\s+for|refers?\s+to|"
    r"describes?|corresponds?\s+to|gives?|counts?|specifies?|means?|"
    r"labels?|indexes?|measures?)\b",
    re.IGNORECASE,
)

_PRE_EXPLICIT_TEMPLATES = [
    r"let\s+{sym}\s+(?:be|denote|represent)\s+(?P<desc>{desc})",
    r"(?:denote|define|write)\s+{sym}\s+(?:as|by|for)\s+(?P<desc>{desc})",
    r"(?:denot\w+|defin\w+)\s+by\s+{sym}\s+(?:the\s+)?(?P<desc>{desc})",
    r"call\s+{sym}\s+the\s+(?P<desc>{desc})",
    r"set\s+{sym}\s+=\s+(?P<desc>{desc})",
]

_DESC_CAPTURE = r"(?:[^.;()\[\]]|\.\s*(?![A-Z\d]))+"


def _protected_split(text):
    """Split on comma/semicolon/'and' while protecting $...$ and (...) tokens."""
    maths = []

    def _protect_math(m):
        maths.append(m.group(0))
        return f"__M{len(maths) - 1}__"

    protected = re.sub(r"\$[^$]{1,400}\$", _protect_math, text)

    parens = []

    def _protect_paren(m):
        parens.append(m.group(0))
        return f"__P{len(parens) - 1}__"

    protected = re.sub(r"\([^)]{1,200}\)", _protect_paren, protected)

    raw_chunks = re.split(r"[,;]\s+|\s+and\s+", protected)

    result = []
    for chunk in raw_chunks:
        chunk = re.sub(r"__M(\d+)__", lambda m: maths[int(m.group(1))], chunk)
        chunk = re.sub(r"__P(\d+)__", lambda m: parens[int(m.group(1))], chunk)
        result.append(chunk.strip())
    return [c for c in result if c]


def _find_where_text(post_text):
    """Return the first where-clause sentence from post_text, or empty string.

    Takes only the first sentence starting with a where/with/here trigger.
    Multi-sentence continuation is intentionally excluded: it causes severe
    false-positive extraction from following prose with incidental relational verbs.
    """
    clean = re.sub(r"\[(?:TARGET|EQ)\]", " ", post_text)
    sents = _split_sentences(clean)
    for sent in sents:
        sent = sent.strip()
        if sent and _WHERE_START.match(sent):
            return sent
    return ""


def _symbol_in_math_token(math_content, symbol):
    """Return True when symbol appears as a token inside a math expression string."""
    variants = _symbol_latex_variants(symbol)
    for v in variants:
        pat = r"(?<![A-Za-z])" + re.escape(v) + r"(?![A-Za-z])"
        if re.search(pat, math_content):
            return True
    return False


def _extract_from_chunk(chunk, symbol):
    """Extract a definition from one where-clause chunk for a given symbol.

    Tries four patterns in priority order: (1) symbol followed by relational
    verb, (2) article + noun phrase before symbol, (3) bare noun after standalone
    symbol, (4) symbol embedded in compound math token.
    """
    sym_pat = _symbol_regex(symbol)

    m = re.search(sym_pat, chunk)
    if m:
        after = chunk[m.end():]
        verb_m = _REL_VERB.search(after)
        if verb_m:
            raw = after[verb_m.end():].strip()
            cleaned = _clean_definition(raw)
            if _use_definition(cleaned):
                return cleaned

        np_m = re.match(r"\s+([a-z][a-z]{3,})\b", after)
        if np_m:
            candidate = np_m.group(1)
            if not re.match(
                r"^(?:is|are|of|in|the|a|an|which|that|to|as|for|from|with|by|"
                r"on|at|and|or|but|after|before|when|while|if|than|then|"
                r"each|every|some|many|most|this|these|thus|also|here|"
                r"just|only|once|more|less|very|very|such|both|given|"
                r"value|values|case|cases|part|parts|form|forms|type|types|"
                r"note|since|under|over|through|between|within|without)\b",
                candidate, re.IGNORECASE
            ):
                cleaned = _clean_definition(candidate)
                if _use_definition(cleaned):
                    return cleaned

    pre_m = re.search(
        rf"(?:the|a|an)\s+([A-Za-z][A-Za-z\s\-]{{2,60}}?)\s*{sym_pat}",
        chunk, re.IGNORECASE
    )
    if pre_m:
        raw_np = pre_m.group(1).rstrip()
        if not raw_np.lower().endswith((" of", " for", " in", " by", " to")):
            cleaned = _clean_definition(raw_np)
            if _use_definition(cleaned):
                return cleaned

    if not re.search(sym_pat, chunk):
        for tok_m in re.finditer(r"\$([^$]{1,400})\$", chunk):
            content = tok_m.group(1)
            if not _symbol_in_math_token(content, symbol):
                continue
            after = chunk[tok_m.end():].strip()
            np_m = re.match(r"([a-z][a-z]{3,})\b", after)
            if np_m:
                candidate = np_m.group(1)
                if not re.match(
                    r"^(?:of|in|the|a|an|which|that|to|as|for|from|with|by|"
                    r"and|or|but|is|are|when|where|while|then|than|each|"
                    r"every|some|both|this|that|these|those|such|here|"
                    r"value|values|case|cases|form|forms|type|types)\b",
                    candidate, re.IGNORECASE
                ):
                    cleaned = _clean_definition(candidate)
                    if _use_definition(cleaned):
                        return cleaned

    return ""


def _parse_respectively(where_text, identifiers):
    """Extract definitions from 'X, Y, ... are A, B, ..., respectively'."""
    m = re.search(
        r"(.+?)\s+are\s+(.+?),?\s+respectively\b",
        where_text, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return {}

    lhs_text = m.group(1)
    rhs_text = m.group(2)

    sym_positions = []
    for ident in identifiers:
        pm = re.search(_symbol_regex(ident), lhs_text)
        if pm:
            sym_positions.append((pm.start(), ident))
    sym_positions.sort()
    syms = [s for _, s in sym_positions]
    if not syms:
        return {}

    defs = _protected_split(rhs_text)
    result = {}
    for sym, defn in zip(syms, defs):
        cleaned = _clean_definition(defn)
        if _use_definition(cleaned):
            result[sym] = cleaned
    return result


def _parse_pre_explicit(pre_text, identifiers):
    """Check the last 2 sentences of pre_text for explicit definition patterns.

    Only fires on: let X be Y, denote X by Y, define X as Y, call X the Y.
    Everything else is rejected for precision.
    """
    if not pre_text:
        return {}

    clean = re.sub(r"\[(?:TARGET|EQ)\]", " ", pre_text)
    sents = _split_sentences(clean)
    search_sents = " ".join(sents[-2:]) if len(sents) >= 2 else " ".join(sents)

    result = {}
    for symbol in identifiers:
        if symbol in result:
            continue
        sym_pat = _symbol_regex(symbol)
        for tmpl in _PRE_EXPLICIT_TEMPLATES:
            pattern = tmpl.format(sym=sym_pat, desc=_DESC_CAPTURE)
            match = re.search(pattern, search_sents, re.IGNORECASE)
            if match:
                if not re.search(sym_pat, match.group(0)):
                    continue
                raw = match.group("desc")
                cleaned = _clean_definition(raw)
                if _use_definition(cleaned):
                    result[symbol] = cleaned
                    break
    return result


def find_symbol_definitions(symbols, context, paper_dict=None, _sources=None):
    """Extract symbol definitions from the equation's surrounding context.

    Three passes in priority order: (1) post-equation where clause, (2)
    respectively coordination within the where clause, (3) explicit
    let/denote/define patterns in the two sentences before the equation.

    paper_dict is accepted for API compatibility but intentionally ignored —
    paper-wide scanning causes cross-symbol contamination.

    _sources is populated in-place with {symbol: provenance} when provided.
    Returns {symbol: definition}; symbols without a confident definition are omitted.
    """
    if not symbols:
        return {}

    parts = context.split("[TARGET]", 1)
    pre_text  = parts[0] if parts else ""
    post_text = parts[1] if len(parts) > 1 else ""

    where_text  = _find_where_text(post_text)
    definitions = {}

    if where_text and "respectively" in where_text.lower():
        resp = _parse_respectively(where_text, symbols)
        for sym, defn in resp.items():
            definitions[sym] = defn
            if _sources is not None:
                _sources[sym] = "respectively"

    remaining = [s for s in symbols if s not in definitions]
    if where_text and remaining:
        chunks = _protected_split(where_text)
        for sym in remaining:
            for chunk in chunks:
                if not re.search(_symbol_regex(sym), chunk):
                    continue
                defn = _extract_from_chunk(chunk, sym)
                if defn:
                    definitions[sym] = defn
                    if _sources is not None:
                        _sources[sym] = "post_where"
                    break

    remaining = [s for s in symbols if s not in definitions]
    if remaining:
        pre_defs = _parse_pre_explicit(pre_text, remaining)
        for sym, defn in pre_defs.items():
            definitions[sym] = defn
            if _sources is not None:
                _sources[sym] = "pre_explicit"

    return definitions
