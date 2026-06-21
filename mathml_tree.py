"""Convert Presentation MathML (from LaTeXML HTML) to Node trees for tree edit distance.

LaTeXML outputs pMathML (visual layout trees), not Content MathML (semantic trees).
We normalize away pure formatting nodes before computing TED so that spacing and
grouping differences do not inflate the distance between structurally equivalent
equations.

Node labeling convention
------------------------
- Structural nodes (internal):  tag name only  e.g. "mfrac", "msup"
- Leaf nodes (content-bearing): tag:text       e.g. "mi:H", "mo:+", "mn:2"

Normalization rules applied during tree construction
-----------------------------------------------------
- _DROP_TAGS (mspace, malignmark, maligngroup): removed entirely.
- _COLLAPSIBLE_TAGS (mrow, mstyle, mpadded, mphantom) with a single child:
  collapsed to that child to avoid depth inflation from grouping wrappers.
- <semantics> wrappers: skipped; first non-annotation child is used directly.
"""

import re
import zss


# Tags with no semantic content — removed during tree construction.
_DROP_TAGS = frozenset({"mspace", "malignmark", "maligngroup", "mprescripts"})

# Transparent grouping tags — collapsed when they have exactly one child.
_COLLAPSIBLE_TAGS = frozenset({"mrow", "mstyle", "mpadded", "mphantom", "merror"})

# Tags that carry text content as leaf nodes.
_LEAF_TAGS = frozenset({"mi", "mo", "mn", "mtext", "ms"})


class Node:
    """Expression tree node compatible with the zss simple_distance interface.

    Parameters
    ----------
    label : str
        Used by ZSS for rename cost computation. Structural nodes use the MathML
        tag name; leaf nodes use 'tag:text' (e.g. 'mi:H', 'mo:+').
    children : list of Node, optional
    """

    def __init__(self, label, children=None):
        self.label = label
        self.children = list(children or [])

    @staticmethod
    def get_children(node):
        """ZSS interface: return child list."""
        return node.children

    @staticmethod
    def get_label(node):
        """ZSS interface: return node label."""
        return node.label

    def __repr__(self):
        return f"Node({self.label!r}, n_children={len(self.children)})"


def _local_tag(el):
    """Return the local tag name, stripping any XML namespace prefix.

    Parameters
    ----------
    el : lxml element

    Returns
    -------
    str
        Local tag name, or empty string for non-element nodes (comments, PI).
    """
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _leaf_text(el):
    """Return normalized text content of a leaf MathML element."""
    raw = (el.text or "").strip()
    return re.sub(r"\s+", " ", raw)


def _build_node(el):
    """Recursively convert one MathML element to a Node tree.

    Parameters
    ----------
    el : lxml element

    Returns
    -------
    Node or None
        None when the element is a pure formatting node with no semantic content.
    """
    tag = _local_tag(el)

    if not tag or tag in _DROP_TAGS:
        return None

    # Leaf nodes: encode text content in the label.
    if tag in _LEAF_TAGS:
        text = _leaf_text(el)
        return Node(f"{tag}:{text}" if text else tag)

    # Structural nodes: recurse into children.
    children = [_build_node(c) for c in el]
    children = [c for c in children if c is not None]

    # Collapse single-child grouping nodes — they add depth without semantics.
    if tag in _COLLAPSIBLE_TAGS:
        if len(children) == 1:
            return children[0]
        if len(children) == 0:
            return None

    return Node(tag, children)


def mathml_to_tree(table):
    """Build a normalized expression Node tree from a LaTeXML equation table.

    Locates the first <math> element in the table, skips the <semantics>
    wrapper if present, and recursively builds a Node tree from the pMathML
    content.

    Parameters
    ----------
    table : lxml element
        The ltx_equation table element from the parsed HTML document.

    Returns
    -------
    Node or None
        Root of the expression tree, or None when no MathML is found or the
        tree would be empty after normalization.
    """
    if table is None:
        return None

    # lxml's HTML parser strips MathML namespace, so tags appear without prefix.
    math_els = table.xpath(".//math")
    if not math_els:
        return None

    math_el = math_els[0]

    # LaTeXML wraps pMathML in <semantics>; first child is the content element,
    # subsequent children are <annotation> nodes we discard.
    semantics = math_el.find("semantics")
    if semantics is not None:
        for child in semantics:
            if _local_tag(child) != "annotation":
                return _build_node(child)
        return None

    # No semantics wrapper — build directly from math element's children.
    children = [_build_node(c) for c in math_el]
    children = [c for c in children if c is not None]

    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return Node("math", children)


def _count_nodes(node):
    """Count total nodes in a tree (recursive DFS).

    Parameters
    ----------
    node : Node

    Returns
    -------
    int
    """
    return 1 + sum(_count_nodes(c) for c in node.children)


def _label_dist(label_a, label_b):
    """ZSS label comparison: 0 if identical, 1 otherwise.

    Parameters
    ----------
    label_a : str
    label_b : str

    Returns
    -------
    int
    """
    return 0 if label_a == label_b else 1


def tree_edit_distance(tree_a, tree_b):
    """Compute normalized tree edit distance similarity in [0, 1].

    Uses the Zhang-Shasha algorithm (zss.simple_distance). Raw TED is an
    integer edit count; we normalize by max(nodes(A), nodes(B)) to produce a
    bounded similarity score:

        tree_sim = 1 - TED(A, B) / max(nodes(A), nodes(B))

    Returning 0.0 for parse failures (None trees) prevents two failed parses
    from being classified as 'strong' due to a spurious similarity of 1.0.

    Parameters
    ----------
    tree_a : Node or None
    tree_b : Node or None

    Returns
    -------
    float
        Similarity in [0, 1]. 0.0 if either tree is None.
    """
    if tree_a is None or tree_b is None:
        return 0.0

    size_a = _count_nodes(tree_a)
    size_b = _count_nodes(tree_b)
    max_size = max(size_a, size_b)

    if max_size == 0:
        return 0.0

    # Guard against extremely large trees that would make ZSS slow (O(n^2 m^2)).
    # Equations with > 200 nodes are likely multi-line arrays; skip TED for them.
    if max_size > 200:
        return 0.0

    ted = zss.simple_distance(
        tree_a, tree_b,
        get_children=Node.get_children,
        get_label=Node.get_label,
        label_dist=_label_dist,
    )

    return max(0.0, 1.0 - ted / max_size)
