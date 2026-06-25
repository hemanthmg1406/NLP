"""Extract equation symbols and their text definitions from cached arXiv HTML.

Identifier extraction uses structured LaTeX first, with MathML/legacy fallbacks
only when the structured pass finds nothing. Definition extraction scans local
pre/post/window prose for generic symbol-definition syntax.

Precision over recall: output nothing when the definition is uncertain.
A missing definition is better than a wrong one.


Provenance tags written to _sources when the caller passes the dict:
    'post_nearby' — exact post-equation definition block
    'pre_nearby'  — exact pre-equation local prose
    'window'      — wider context containing explicit definition syntax
    'respectively' — respectively coordinated list from pipeline.py
"""

import re
from pathlib import Path

from lxml import html as lxml_html
from pylatexenc.latexwalker import (LatexWalker, LatexMacroNode,
                                     LatexCharsNode, LatexGroupNode,
                                     LatexEnvironmentNode)

from context import get_contexts, _split_sentences
from equations import extract_equations

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
    "mathcal", "mathbb", "mathscr", "mathfrak", "mathbf", "mathsf"
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
    "dot", "ddot", "dddot",
    "bigotimes", "bigoplus", "bigcup", "bigcap", "bigvee", "bigwedge", "biguplus",
    "coprod",
}

IDENTIFIER_STOP = {
    "o",
    "sin", "cos", "tan", "log", "ln", "exp", "tr", "Tr", "trace", "Pr",
    "Re", "Im", "det", "rank", "dim", "mod", "sup", "inf", "limsup",
    "liminf", "argmin", "argmax", "Perm", "Haf", "min", "max", "pi", "e",
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

    Extracts structured LaTeX symbols first, preserving decorations, simple
    subscripts, superscripts, and function arguments. MathML and older LaTeX
    fallbacks are used only as extra recall, after the structured pass has had
    the chance to keep symbols such as mathsf_L_x, gamma_x_jminus, and H_x_LS
    intact.
    """
    bound = get_bound_variables(latex)

    symbols = set()
    symbols.update(_latex_structured_identifiers(latex))
    symbols.update(_latex_compact_identifiers(latex))
    if not symbols:
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


_SCRIPT_COMMAND_MAP = {
    "uparrow": "up",
    "downarrow": "down",
    "rightarrow": "to",
    "leftarrow": "from",
    "prime": "prime",
    "dagger": "dagger",
    "ddagger": "ddagger",
    "ast": "star",
}

_SKIP_GROUP_COMMANDS = {
    "text", "textrm", "mathrm", "operatorname", "rm", "mbox",
}

_NON_SYMBOL_SUPERSCRIPTS = {
    "dagger", "ddagger", "prime", "star", "ast", "T", "top",
}


def _latex_structured_identifiers(latex):
    """Extract first-class equation symbols from raw LaTeX.

    The older extractor flattened complex nodes like sigma^x_{k,j} into sigma.
    This pass keeps the rendered symbol shape as the key. It is intentionally
    syntactic: it does not infer physics, it only preserves what the equation
    actually writes.
    """
    if not latex:
        return set()
    text = _strip_comments(latex)
    out = set()
    i = 0
    while i < len(text):
        matched = _read_symbol_base(text, i)
        if matched is None:
            i += 1
            continue
        base, end = matched
        if not base:
            i = max(end, i + 1)
            continue
        subs, sups, args, end = _read_symbol_tail(text, end)
        key = _assemble_structured_key(base, subs, sups, args)
        if key and _keep_identifier(key):
            out.add(key)
        i = max(end, i + 1)
    return out


def _strip_comments(latex):
    """Remove LaTeX comments and layout commands that do not change symbols."""
    latex = re.sub(r"%[^\n]*", " ", latex)
    latex = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)\b", " ", latex)
    return latex


def _read_symbol_base(text, pos):
    """Read one symbol base at text[pos], returning (key, end) or None."""
    ch = text[pos]
    if ch == "\\":
        m = re.match(r"\\([A-Za-z]+)", text[pos:])
        if not m:
            return None
        cmd = m.group(1)
        end = pos + len(m.group(0))
        if cmd in DECORATORS:
            body, body_end = _read_balanced_group(text, end)
            if body is None:
                return "", end
            base = _script_fragment_key(body)
            if not base:
                return "", body_end
            return f"{cmd}_{base}", body_end
        if cmd in GREEK:
            return cmd, end
        if cmd in _SKIP_GROUP_COMMANDS:
            _, group_end = _read_balanced_group(text, end)
            return "", group_end if group_end > end else end
        if cmd in COMMAND_STOP or cmd in IDENTIFIER_STOP:
            return "", end
        # Unknown commands are usually package macros or operators. Keep short
        # alphabetic ones only when they look like named variables.
        if len(cmd) <= 3 and cmd not in COMMAND_STOP:
            return cmd, end
        return "", end
    if ch.isalpha():
        prev_ok = pos == 0 or not text[pos - 1].isalpha()
        next_ok = pos + 1 >= len(text) or not text[pos + 1].isalpha()
        if ch.isupper() and prev_ok:
            return ch, pos + 1
        if prev_ok and next_ok:
            return ch, pos + 1
    return None


def _read_symbol_tail(text, pos):
    """Read scripts and a simple function argument after a base symbol."""
    subs = []
    sups = []
    i = pos
    while True:
        i = _skip_ws(text, i)
        if i >= len(text) or text[i] not in "_^":
            break
        kind = text[i]
        raw, i = _read_script_arg(text, i + 1)
        part = _script_fragment_key(raw)
        if part:
            if kind == "_":
                subs.append(part)
            else:
                sups.append(part)
    args = []
    arg, end = _read_function_arg(text, i)
    if arg:
        args.append(arg)
        i = end
    return subs, sups, args, i


def _skip_ws(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _read_script_arg(text, pos):
    """Read a subscript or superscript argument."""
    pos = _skip_ws(text, pos)
    if pos >= len(text):
        return "", pos
    if text[pos] == "{":
        body, end = _read_balanced_group(text, pos)
        return body or "", end
    if text[pos] == "[":
        end = text.find("]", pos + 1)
        if end != -1:
            return text[pos + 1:end], end + 1
    if text[pos] == "\\":
        m = re.match(r"\\[A-Za-z]+(?:\{[^{}]{0,40}\})?", text[pos:])
        if m:
            return m.group(0), pos + len(m.group(0))
    return text[pos], pos + 1


def _read_balanced_group(text, pos):
    """Read a brace group starting at pos or just after optional whitespace."""
    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != "{":
        return None, pos
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1:i], i + 1
    return None, pos + 1


def _read_function_arg(text, pos):
    """Read one simple function argument such as (t), (k), or (nu_i)."""
    i = _skip_ws(text, pos)
    if text.startswith(r"\left", i):
        i = _skip_ws(text, i + len(r"\left"))
    if i >= len(text) or text[i] != "(":
        return "", pos
    depth = 0
    for j in range(i, min(len(text), i + 80)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                body = text[i + 1:j].strip()
                key = _function_arg_key(body)
                return key, j + 1
    return "", pos


def _function_arg_key(body):
    """Return a compact key for a simple function argument."""
    if not body or "," in body or "=" in body or len(body) > 40:
        return ""
    body = re.sub(r"\\(?:left|right)\b", " ", body)
    parts = []
    for item in re.finditer(r"\\[A-Za-z]+(?:_\{?[^{}\s,]+\}?)?|[A-Za-z](?:_\{?[^{}\s,]+\}?)?", body):
        token = item.group(0)
        key = _script_fragment_key(token)
        if key and key not in IDENTIFIER_STOP and key not in COMMAND_STOP:
            parts.append(key)
    if not parts or len(parts) > 3:
        return ""
    return "_".join(parts)


def _script_fragment_key(raw):
    """Convert a small script or grouped symbol fragment into a key part."""
    if raw is None:
        return ""
    raw = raw.strip()
    if not raw:
        return ""
    raw = re.sub(
        r"\\(?:text|textrm|mathrm|rm|mathit|mathsf)\{([^{}]{1,40})\}",
        r"\1",
        raw,
    )
    text_m = re.fullmatch(
        r"\\(?:text|textrm|mathrm|rm|mathit|mathsf)\{([^{}]{1,40})\}",
        raw,
    )
    if text_m:
        raw = text_m.group(1)
    raw = raw.strip("{}[]() ")
    deco_m = re.fullmatch(
        r"\\(hat|widehat|bar|overline|tilde|vec|bm|boldsymbol|mathcal|mathbb|mathscr|mathfrak|mathbf|mathsf)\{([^{}]{1,40})\}",
        raw,
    )
    if deco_m:
        return f"{deco_m.group(1)}_{_script_fragment_key(deco_m.group(2))}"

    def repl_cmd(m):
        name = m.group(1)
        if name in GREEK:
            return f"_{name}_"
        return f"_{_SCRIPT_COMMAND_MAP.get(name, name)}_"

    raw = re.sub(r"\\([A-Za-z]+)", repl_cmd, raw)
    raw = raw.replace("+", "plus").replace("-", "minus")
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        return ""
    if raw in COMMAND_STOP or raw in IDENTIFIER_STOP:
        return ""
    return raw


def _assemble_structured_key(base, subs, sups, args):
    """Assemble base, scripts, and simple function args into one symbol key."""
    if not base or base in COMMAND_STOP or base in IDENTIFIER_STOP:
        return ""
    parts = [base]
    parts.extend(p for p in subs if p)
    for sup in sups:
        if not sup or sup in _NON_SYMBOL_SUPERSCRIPTS:
            continue
        if sup.isdigit() or re.fullmatch(r"(?:minus|plus)?\d+", sup):
            continue
        parts.append(sup)
    parts.extend(p for p in args if p)
    key = "_".join(parts)
    key = normalize_identifier(key)
    return key


def _latex_compact_identifiers(latex):
    """Collect compact products that papers define as one token, such as dt."""
    found = set()
    for m in re.finditer(r"(?<![A-Za-z\\])d([A-Za-z])(?![A-Za-z])", latex or ""):
        found.add(f"d{m.group(1)}")
    return found


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
        if base in DECORATORS:
            return True
        if base in IDENTIFIER_STOP or base in COMMAND_STOP:
            return False
        if base.islower() and len(base) > 1 and base.lower() in {s.lower() for s in IDENTIFIER_STOP}:
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
    "the following", "as follows", "zero", "one", "defined", "property",
    "by definition", "condition", "conditions",
})


def _clean_definition(text):
    """Clean and truncate a matched definition span."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,:;-")
    text = re.sub(r"\s*\[(?:TARGET|EQ)\].*$", "", text)
    text = re.sub(r"^(?:for|as)\s+(?=\w)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^being\s+(?:the\s+|a\s+|an\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s+\b(?:with\s+respect(?:\s+to)?|such\s+that|if\s+and\s+only\s+if|"
        r"for\s+all|there\s+exists)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
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
    if re.search(r"\brespectively\b", low):
        return False
    if re.search(
        r"\b(?:if\s+and\s+only\s+if|such\s+that|with\s+respect(?:\s+to)?|"
        r"for\s+all|there\s+exists|exists\s+a|"
        r"the\s+following\s+holds|called\s+\w+\s+if)\b",
        low,
    ):
        return False
    if low.endswith((" if and only", " with respect", " such that")):
        return False
    if len(low) > 150 and re.search(r"\$|\\[A-Za-z]+|:", low):
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
    # Reject bare verb phrases (the whole span is a copula or equality), but
    # allow noun phrases that happen to contain a relational verb embedded
    # deeper in the text (e.g. "density operator that is Hermitian" is fine).
    words = low.split()
    if words and words[0] in {"is", "are", "was", "were", "equals"}:
        return False
    if re.match(r"^(no\s+longer|not\s+|never\s+|neither\s+)", low):
        return False
    if re.match(
        r"^(using|applying|choosing|taking|making|setting|computing|evaluating)\b", low
    ):
        return False
    if re.match(
        r"^(?:resonant|proportional|dependent|independent|linear|nonlinear)\s+"
        r"(?:with|to|on|in|as)\b",
        low,
    ):
        return False
    if re.match(r"^(?:and\s+)?therefore\b|^nd\s+therefore\b", low):
        return False
    if re.search(
        r"\b(?:induces?|requires?|commutes?|absorbs?|emits?|activates?|"
        r"introduces?|reduces?|increases?|decreases?|correctable)\b",
        low,
    ):
        return False
    if re.search(
        r"\b(?:must|should|can|could|would|will|may|might)\s+be\b",
        low,
    ):
        return False
    if re.match(r"^(?:be\s+used|used\s+to|can\s+be\s+used|has\s+the\s+property)\b", low):
        return False
    if " by definition" in low and len(low.split()) <= 8:
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


_WHERE_CONTINUE = re.compile(
    r"^\s*(?:and\s+)?(?:where|with|here|in\s+which)\b"
    r"|^\s*(?:[A-Z]?\$[^$]{1,40}\$\s+(?:is|are|denotes?|represents?)\b)",
    re.IGNORECASE,
)


def _find_where_text(post_text):
    """Return the where-clause block from post_text, collecting multiple sentences.

    Starts at the first sentence matching _WHERE_START. Continues collecting
    subsequent sentences that either (a) begin with another where/with/here
    trigger or (b) open with an inline math token followed by a relational verb,
    which signals a continuation definition list. Stops at the first sentence
    that matches neither criterion, preventing bleed into unrelated prose.
    """
    clean = re.sub(r"\[(?:TARGET|EQ)\]", " ", post_text)
    sents = _split_sentences(clean)
    result = []
    inside = False
    for sent in sents:
        s = sent.strip()
        if not s:
            continue
        if not inside:
            if _WHERE_START.match(s):
                inside = True
                result.append(s)
        else:
            if _WHERE_START.match(s) or _WHERE_CONTINUE.match(s):
                result.append(s)
            else:
                break
    return " ".join(result)


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
    """Check up to the last 5 sentences of pre_text for explicit definition patterns.

    Only fires on: let X be Y, denote X by Y, define X as Y, call X the Y.
    Five sentences covers multi-sentence notation introductions while staying
    precise enough to avoid false matches from unrelated prose.
    """
    if not pre_text:
        return {}

    clean = re.sub(r"\[(?:TARGET|EQ)\]", " ", pre_text)
    sents = _split_sentences(clean)
    search_sents = " ".join(sents[-5:]) if len(sents) >= 5 else " ".join(sents)

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


_spacy_nlp = None


def _get_spacy_nlp():
    """Load en_core_sci_sm on first call; return None when unavailable."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp if _spacy_nlp is not False else None
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_sci_sm", disable=["ner"])
    except (OSError, ImportError):
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except (OSError, ImportError):
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp is not False else None


def _spacy_definition_arcs(where_text, symbols):
    """Extract symbol definitions via spaCy dependency arcs in the where-clause.

    For each symbol token found in the text, walks nsubj/appos/attr arcs to
    find the head noun phrase that names the physical quantity. This catches
    patterns like "where rho is the density operator" that template matching
    misses when the copula structure is unusual. Returns {symbol: definition}.

    Only fires when en_core_sci_sm (or en_core_web_sm fallback) is available.
    Results go through _use_definition for the same precision gate as other passes.
    """
    nlp = _get_spacy_nlp()
    if nlp is None or not where_text:
        return {}

    result = {}
    # Strip inline math markers so spaCy tokenises surrounding prose correctly.
    # Replace $X$ with the normalised symbol key so the token appears literally.
    cleaned = re.sub(r"\$([^$]{1,60})\$", lambda m: m.group(1).replace("\\", "").replace("{", "").replace("}", "").strip() or "SYM", where_text)
    cleaned = re.sub(r"\[(?:TARGET|EQ)\]", " ", cleaned)

    doc = nlp(cleaned[:800])  # limit to avoid slow parses on long chunks

    for sym in symbols:
        if sym in result:
            continue
        sym_variants = {sym, sym.replace("_", "").lower()}
        for token in doc:
            tok_low = re.sub(r"[^a-z0-9]", "", token.text.lower())
            if tok_low not in sym_variants and not any(tok_low.startswith(v[:2]) and len(tok_low) <= len(v) + 2 for v in sym_variants):
                continue

            # Walk dependency arcs: nsubj (token is subject of a copula),
            # appos (appositional modifier), attr (predicate nominal).
            head = token.head
            for dep_tok in list(doc):
                if dep_tok.dep_ in {"nsubj", "nsubjpass"} and dep_tok.head is head:
                    if any(re.sub(r"[^a-z0-9]", "", dep_tok.text.lower()).startswith(v[:2]) for v in sym_variants):
                        # Head is the verb; find the attr complement (the definition NP).
                        for child in head.children:
                            if child.dep_ in {"attr", "acomp"} and child.pos_ in {"NOUN", "PROPN", "ADJ"}:
                                np = " ".join(t.text for t in child.subtree
                                              if t.dep_ not in {"punct", "cc"})
                                cleaned_def = _clean_definition(np)
                                if _use_definition(cleaned_def):
                                    result[sym] = cleaned_def
                                break
                        break

            if sym in result:
                break

            # Direct appositive: "rho, the density matrix"
            for dep_tok in list(doc):
                if dep_tok.dep_ == "appos":
                    gov_low = re.sub(r"[^a-z0-9]", "", dep_tok.head.text.lower())
                    if gov_low in sym_variants:
                        np = " ".join(t.text for t in dep_tok.subtree
                                      if t.dep_ not in {"punct", "cc"})
                        cleaned_def = _clean_definition(np)
                        if _use_definition(cleaned_def):
                            result[sym] = cleaned_def
                        break

    return result


_MATH_PLACEHOLDER_RE = re.compile(r"@@M(\d+)@@")


def _split_definition_context(context):
    """Split the caller context into exact pre, exact post, and wider window."""
    if not context:
        return "", "", ""
    main, window = (context.split("[WINDOW]", 1) + [""])[:2] if "[WINDOW]" in context else (context, "")
    if "[TARGET]" in main:
        pre_text, post_text = main.split("[TARGET]", 1)
    else:
        pre_text, post_text = "", main
    return pre_text.strip(), post_text.strip(), window.strip()


def _candidate_definition_sentences(pre_text, post_text, window_text):
    """Return ordered local sentences with provenance labels."""
    out = []
    seen = set()

    def add(label, text, reverse=False, limit=8):
        sents = _split_sentences(text or "")
        if reverse:
            sents = list(reversed(sents))
        for sent in sents[:limit]:
            clean = re.sub(r"\s+", " ", sent).strip()
            if not clean or clean in seen:
                continue
            if label == "window" and not re.search(
                r"^\s*(?:where|with|here|in\s+which)\b|"
                r"\b(?:denotes?|represents?|stands?\s+for|refers?\s+to|defined)\b",
                clean[:240],
                re.I,
            ):
                continue
            seen.add(clean)
            out.append((label, clean))

    add("post_nearby", post_text, limit=8)
    add("pre_nearby", pre_text, reverse=True, limit=8)
    add("window", window_text, limit=12)
    return out


def _protect_inline_math(text):
    """Replace inline math with placeholders and record their structured symbols."""
    raw = []
    key_sets = []

    def repl(m):
        idx = len(raw)
        body = m.group(1)
        raw.append(body)
        key_sets.append(_latex_structured_identifiers(body) | _latex_compact_identifiers(body))
        return f"@@M{idx}@@"

    protected = re.sub(r"\$([^$]{1,500})\$", repl, text)
    return protected, raw, key_sets


def _restore_math_placeholders(text, raw_math):
    """Restore math placeholders as compact inline LaTeX snippets."""
    def repl(m):
        idx = int(m.group(1))
        if 0 <= idx < len(raw_math):
            return f"${raw_math[idx]}$"
        return ""
    return _MATH_PLACEHOLDER_RE.sub(repl, text)


def _symbol_aliases(symbol):
    """Return comparable aliases for one canonical symbol key."""
    aliases = {symbol}
    aliases.add(symbol.replace("_", ""))
    aliases.add(re.sub(r"_(\d+)$", r"\1", symbol))
    parts = symbol.split("_")
    if len(parts) >= 2 and len(parts[-1]) == 1 and parts[-1].islower():
        root_parts = parts[:-1]
        root = "_".join(root_parts)
        root_base = root_parts[0] if root_parts else ""
        # Treat a trailing one-letter argument as optional only for decorated
        # notation (mathcal_E_t -> mathcal_E). Do not collapse plain symbols
        # such as P_d and P_r into P, because those subscripts usually carry
        # different meanings.
        if root_base in DECORATORS:
            aliases.add(root)
    if parts and parts[0] in DECORATORS and len(parts) >= 2:
        aliases.add("_".join(parts[1:]))
    return {a for a in aliases if a}


def _symbols_equivalent(wanted, found):
    """Return True when two symbol keys describe the same local notation."""
    wanted_aliases = _symbol_aliases(wanted)
    found_aliases = _symbol_aliases(found)
    return bool(wanted_aliases & found_aliases)


def _math_token_definitions(raw_math, symbols):
    """Extract definitions written inside one inline math token, e.g. X=Y."""
    result = {}
    eq_pat = r":=|\\coloneqq|\\equiv|(?<![<>])=(?!=)"
    if not re.search(eq_pat, raw_math):
        return result
    if re.search(r"\\iff|\\Leftrightarrow|\\leq|\\geq|≤|≥|<|>", raw_math):
        return result
    parts = re.split(eq_pat, raw_math, maxsplit=1)
    if len(parts) != 2:
        return result
    lhs, rhs = parts[0].strip(), parts[1].strip()
    lhs_keys = _latex_structured_identifiers(lhs) | _latex_compact_identifiers(lhs)
    if len(lhs_keys) != 1:
        return result
    rhs_clean = _clean_definition(f"${rhs}$")
    if not (_use_definition(rhs_clean) or _use_formula_definition(rhs_clean)):
        return result
    for sym in symbols:
        if any(_symbols_equivalent(sym, key) for key in lhs_keys):
            result[sym] = rhs_clean
    return result


def _use_formula_definition(desc):
    """Allow compact formula RHS values when the LHS is an exact single symbol."""
    if not desc:
        return False
    text = desc.strip()
    if not (text.startswith("$") and text.endswith("$")):
        return False
    inner = text.strip("$").strip()
    if not inner or len(inner) > 140:
        return False
    if re.search(r"\\iff|\\Leftrightarrow|\\leq|\\geq|≤|≥|<|>", inner):
        return False
    return bool(re.search(r"[=+\-*/^_]|\\(?:frac|sqrt|sum|prod|int|operatorname|textrm|mathrm)", inner))


def _mention_spans(protected, raw_math, key_sets, symbol):
    """Find placeholder or bare-text mentions of a symbol in protected text."""
    spans = []
    for m in _MATH_PLACEHOLDER_RE.finditer(protected):
        idx = int(m.group(1))
        keys = key_sets[idx] if 0 <= idx < len(key_sets) else set()
        if any(_symbols_equivalent(symbol, key) for key in keys):
            spans.append((m.start(), m.end()))

    aliases = sorted(_symbol_aliases(symbol), key=len, reverse=True)
    for alias in aliases:
        if len(alias) == 1 and not alias.isupper():
            continue
        bare = re.escape(alias).replace("_", r"[_\s]?")
        if len(alias) == 1:
            pat = rf"(?<![A-Za-z0-9_+\-]){bare}(?![A-Za-z0-9_+\-])"
        else:
            pat = rf"(?<![A-Za-z0-9]){bare}(?![A-Za-z0-9])"
        for m in re.finditer(pat, protected):
            if any(max(a, m.start()) < min(b, m.end()) for a, b in spans):
                continue
            spans.append((m.start(), m.end()))
    return sorted(spans)


def _span_allows_definition_before(protected, span, raw_math, key_sets, symbol):
    """Return True when noun-before-symbol evidence is safe to use."""
    token = protected[span[0]:span[1]]
    m = re.fullmatch(r"@@M(\d+)@@", token)
    if not m:
        return True
    idx = int(m.group(1))
    keys = key_sets[idx] if 0 <= idx < len(key_sets) else set()
    matching = {key for key in keys if _symbols_equivalent(symbol, key)}
    if len(keys) == 1 and matching:
        return True

    raw = raw_math[idx] if 0 <= idx < len(raw_math) else ""
    lhs = re.split(
        r"(?<![<>])=(?!=)|:=|\\coloneqq|\\propto|\\approx|\\simeq|\\sim",
        raw,
        maxsplit=1,
    )[0]
    lhs_keys = _latex_structured_identifiers(lhs) | _latex_compact_identifiers(lhs)
    if not any(_symbols_equivalent(symbol, key) for key in lhs_keys):
        return False
    before = protected[:span[0]]
    return bool(re.search(
        r"(?:is|are|was|were)\s+(?:given|defined|represented|denoted|written)"
        r"(?:\s+to\s+\w+\s+order)?\s+by\s*$",
        before,
        re.I,
    ))


def _span_allows_definition_after(protected, span, raw_math, key_sets, symbol):
    """Return True when a mention can own text that follows it."""
    token = protected[span[0]:span[1]]
    m = re.fullmatch(r"@@M(\d+)@@", token)
    if not m:
        return True
    idx = int(m.group(1))
    keys = key_sets[idx] if 0 <= idx < len(key_sets) else set()
    if len(keys) == 1 and any(_symbols_equivalent(symbol, key) for key in keys):
        return True

    raw = raw_math[idx] if 0 <= idx < len(raw_math) else ""
    if not re.search(r"(?<![<>])=(?!=)|:=|\\coloneqq|\\equiv", raw):
        return False
    lhs = re.split(r"(?<![<>])=(?!=)|:=|\\coloneqq|\\equiv", raw, maxsplit=1)[0]
    lhs_keys = _latex_structured_identifiers(lhs) | _latex_compact_identifiers(lhs)
    return any(_symbols_equivalent(symbol, key) for key in lhs_keys)


def _allow_single_char_definition(protected, span, raw_math, key_sets, symbol, desc):
    """Single-character prose definitions require exact inline-math ownership."""
    if len(symbol) != 1:
        return True
    if desc.strip().startswith("$"):
        return True
    token = protected[span[0]:span[1]]
    m = re.fullmatch(r"@@M(\d+)@@", token)
    if not m:
        return False
    idx = int(m.group(1))
    keys = key_sets[idx] if 0 <= idx < len(key_sets) else set()
    return len(keys) == 1 and any(_symbols_equivalent(symbol, key) for key in keys)


def _placeholder_symbol_map(placeholders, key_sets, symbols):
    """Map math placeholders in a coordinated group to requested symbols."""
    pairs = []
    used = set()
    for idx in placeholders:
        keys = key_sets[idx] if 0 <= idx < len(key_sets) else set()
        for sym in symbols:
            if sym in used:
                continue
            if len(sym) == 1 and len(keys) > 1:
                continue
            if any(_symbols_equivalent(sym, key) for key in keys):
                pairs.append((idx, sym))
                used.add(sym)
                break
    return pairs


def _split_respectively_descriptions(desc, count):
    """Split the right side of a respectively construction into count parts."""
    if count <= 0:
        return []
    desc = re.sub(r"\s+", " ", desc or "").strip(" ,;")
    if not desc:
        return []

    pieces = _protected_split(desc)
    if len(pieces) == count:
        prefix = ""
        m = re.match(
            r"(?P<prefix>.*\b(?:of|onto|on|in|for|to|by|from|along)\s+)"
            r"(?:@@M\d+@@|[^,;]+?)\s*$",
            pieces[0],
            re.I,
        )
        if m:
            prefix = m.group("prefix")
        out = []
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if i and prefix and not re.match(r"^(?:the|a|an)\s+", piece, re.I):
                piece = prefix + piece
            out.append(piece)
        return out

    if count == 2:
        m = re.match(
            r"(?P<prefix>.+?\b(?:of|onto|on|in|for|to|by|from|along)\s+)"
            r"(?P<a>@@M\d+@@|[A-Za-z][^,;]+?)\s+and\s+"
            r"(?P<b>@@M\d+@@|[A-Za-z][^,;]+)$",
            desc,
            re.I,
        )
        if m:
            prefix = m.group("prefix")
            return [prefix + m.group("a").strip(), prefix + m.group("b").strip()]

    return pieces if len(pieces) == count else []


def _capture_definition_tail(text):
    """Capture a definition tail until the next symbol definition begins."""
    boundary = re.search(
        r"(?:[,;]\s*(?:and\s+)?)?(?:@@M\d+@@|[A-Z][A-Za-z0-9_]{0,20})\s+"
        r"(?:is|are|denotes?|represents?|stands?\s+for|refers?\s+to)\b",
        text,
    )
    if boundary and boundary.start() > 0:
        text = text[:boundary.start()]
    boundary = re.search(
        r"\s+and\s+(?:the\s+)?[A-Za-z][A-Za-z\s\-]{2,80}\s+"
        r"(?:is|are|was|were|denotes?|represents?|stands?\s+for|"
        r"refers?\s+to|given|defined)\b",
        text,
        re.I,
    )
    if boundary and boundary.start() > 0:
        text = text[:boundary.start()]
    boundary = re.search(
        r"[,;]\s*(?:and\s+|with\s+)?@@M\d+@@\s+(?:the|a|an|[A-Za-z])",
        text,
    )
    if boundary and boundary.start() > 0:
        text = text[:boundary.start()]
    boundary = re.search(r"\s+(?:and|while|whereas)\s+@@M\d+@@\b", text)
    if boundary and boundary.start() > 0:
        text = text[:boundary.start()]
    return text


def _definition_after(protected, span, raw_math):
    """Extract a definition appearing after one symbol mention."""
    before = protected[:span[0]].lower()
    tail = protected[span[1]:].strip()
    tail = re.sub(r"^[,;:\s]+", "", tail)

    eq_m = re.match(r"(?:=|:=|\\coloneqq)\s*(.+)$", tail)
    if eq_m:
        return _restore_math_placeholders(_capture_definition_tail(eq_m.group(1)), raw_math)

    verb_m = re.match(
        r"(?:is|are|denotes?|represent(?:s)?|stand(?:s)?\s+for|refer(?:s)?\s+to|"
        r"describe(?:s)?|correspond(?:s)?\s+to|account(?:s)?\s+for|being)\s+(.+)$",
        tail,
        re.I,
    )
    if verb_m:
        return _restore_math_placeholders(_capture_definition_tail(verb_m.group(1)), raw_math)

    if re.search(r"\b(?:where|with|here|in which|in this notation)\b", before[-120:], re.I):
        np_m = re.match(
            r"(?:(?:the|a|an)\s+)?([A-Za-z][A-Za-z0-9\s\-/]{2,120})"
            r"(?:[,;.]|$|\s+and\s+@@M\d+@@)",
            tail,
            re.I,
        )
        if np_m:
            return np_m.group(1)
    return ""


def _definition_before(protected, span):
    """Extract a noun phrase before a symbol mention."""
    before = protected[:span[0]].strip()
    m = re.search(
        r"(?:the|a|an)\s+([A-Za-z][A-Za-z\s\-]{2,80}?)\s+"
        r"(?:is|are|was|were)\s+(?:given|defined|represented|denoted|written)"
        r".{0,50}\s+by\s*$",
        before,
        re.I,
    )
    if m:
        phrase = m.group(1).strip()
        if re.search(r"\band\s+the\b", phrase, re.I):
            phrase = re.split(r"\band\s+the\b", phrase, flags=re.I)[-1].strip()
        return phrase
    m = re.search(r"(?:the|a|an)\s+([A-Za-z][A-Za-z\s\-]{2,80}?)\s*$", before, re.I)
    if m:
        phrase = m.group(1).strip()
        if re.search(r"\bthe\b", phrase, re.I):
            phrase = re.split(r"\bthe\b", phrase, flags=re.I)[-1].strip()
        low = phrase.lower()
        if " as a function" in low or " function of" in low:
            return ""
        if re.search(
            r"\b(?:induces?|depends?|requires?|commutes?|absorbs?|emits?|"
            r"activates?|introduces?|reduces?|increases?|decreases?|"
            r"modulates?|contains?|couples?|correctable)\b",
            low,
        ):
            return ""
        if len(phrase.split()) <= 8 and " and " not in low and not low.endswith("and"):
            return phrase
    return ""


def _coordinated_group_definitions(protected, raw_math, key_sets, symbols):
    """Handle lists such as X, Y, and Z are unitary operators."""
    result = {}
    group_re = re.compile(
        r"(?P<group>@@M\d+@@(?:\s*,\s*@@M\d+@@)*(?:\s*,?\s*and\s*@@M\d+@@)?)"
        r"\s+(?:are|denote|denotes|represent|represents|stand\s+for|stands\s+for)\s+"
        r"(?P<desc>[^.;]+)",
        re.I,
    )
    for m in group_re.finditer(protected):
        if re.search(r"\brespectively\b", m.group("desc"), re.I):
            continue
        desc = _clean_definition(_restore_math_placeholders(m.group("desc"), raw_math))
        if not _use_definition(desc):
            continue
        placeholders = [int(x) for x in re.findall(r"@@M(\d+)@@", m.group("group"))]
        if len(placeholders) < 2:
            continue
        for idx in placeholders:
            keys = key_sets[idx] if 0 <= idx < len(key_sets) else set()
            for sym in symbols:
                if sym in result:
                    continue
                if len(sym) == 1 and len(keys) > 1:
                    continue
                if any(_symbols_equivalent(sym, key) for key in keys):
                    result[sym] = desc
    return result


def _respectively_group_definitions(protected, raw_math, key_sets, symbols):
    """Handle paired lists such as X and Y denote A and B, respectively."""
    result = {}
    group = r"@@M\d+@@(?:\s*,\s*@@M\d+@@)*(?:\s*,?\s*and\s*@@M\d+@@)?"
    resp_re = re.compile(
        rf"(?P<group>{group})\s+"
        r"(?:(?:are|is|denote|denotes|represent|represents|stand\s+for|"
        r"stands\s+for|refer\s+to|refers\s+to)\s+)?"
        r"(?P<desc>[^.;]*?)\s*,?\s*respectively\b",
        re.I,
    )
    for m in resp_re.finditer(protected):
        placeholders = [int(x) for x in re.findall(r"@@M(\d+)@@", m.group("group"))]
        pairs = _placeholder_symbol_map(placeholders, key_sets, symbols)
        if len(pairs) < 2:
            continue
        desc_parts = _split_respectively_descriptions(m.group("desc"), len(pairs))
        if len(desc_parts) != len(pairs):
            continue
        for (_, sym), desc in zip(pairs, desc_parts):
            if sym in result:
                continue
            cleaned = _clean_definition(_restore_math_placeholders(desc, raw_math))
            if _use_definition(cleaned):
                result[sym] = cleaned
    return result


def _sentence_definitions(sentence, symbols):
    """Extract symbol definitions from one local sentence."""
    protected, raw_math, key_sets = _protect_inline_math(sentence)
    result = {}

    for raw in raw_math:
        for sym, defn in _math_token_definitions(raw, symbols).items():
            result.setdefault(sym, defn)

    for sym, defn in _respectively_group_definitions(
        protected, raw_math, key_sets, symbols
    ).items():
        result.setdefault(sym, defn)

    for sym, defn in _coordinated_group_definitions(protected, raw_math, key_sets, symbols).items():
        result.setdefault(sym, defn)

    for sym in symbols:
        if sym in result:
            continue
        for span in _mention_spans(protected, raw_math, key_sets, sym):
            defn = ""
            if _span_allows_definition_after(protected, span, raw_math, key_sets, sym):
                defn = _definition_after(protected, span, raw_math)
            if not defn and _span_allows_definition_before(protected, span, raw_math, key_sets, sym):
                defn = _definition_before(protected, span)
            cleaned = _clean_definition(defn)
            if _use_definition(cleaned) and _allow_single_char_definition(
                protected, span, raw_math, key_sets, sym, cleaned
            ):
                result[sym] = cleaned
                break
    return result


def find_symbol_definitions(symbols, context, paper_dict=None, _sources=None):
    """Extract symbol definitions from local equation evidence.

    The matcher is equation-first: only symbols already found in the equation
    are searched. It then scans exact nearby pre/post text and a wider context
    window for generic definition patterns. No paper-topic knowledge is used.
    """
    if not symbols:
        return {}

    symbols = list(dict.fromkeys(symbols))
    pre_text, post_text, window_text = _split_definition_context(context)
    definitions = {}

    for source, sent in _candidate_definition_sentences(pre_text, post_text, window_text):
        local_defs = _sentence_definitions(sent, [s for s in symbols if s not in definitions])
        for sym, defn in local_defs.items():
            if sym in definitions:
                continue
            if defn.lower() in {"set of all", "the set of all"}:
                continue
            sym_base = re.escape(sym.split("_", 1)[-1])
            tuple_pat = (
                r"\b[Ll]et\s*(?:\(|\$?\s*\\left\()\s*[^.\n]{0,160}\b" +
                sym_base +
                r"\b[^.\n]{0,160}(?:\)|\\right\))\s*(?:\$?\s*)be\b"
            )
            if re.search(tuple_pat, sent):
                continue
            definitions[sym] = defn
            if _sources is not None:
                _sources[sym] = source

    return definitions
