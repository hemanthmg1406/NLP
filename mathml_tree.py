"""Convert Presentation MathML (from LaTeXML HTML) to Node trees for tree edit distance.

LaTeXML outputs pMathML (visual layout trees), not Content MathML. Formatting
nodes are normalized away before TED so that spacing and grouping differences
do not inflate the distance between structurally equivalent equations.

Structural nodes are labeled by tag name (e.g. "mfrac", "msup"). Leaf nodes
carry their text content (e.g. "mi:H", "mo:+", "mn:2"). During construction:
_DROP_TAGS are removed entirely, single-child _COLLAPSIBLE_TAGS are collapsed,
and semantics wrappers are skipped in favour of the first content child.
"""

import re
import zss

_DROP_TAGS = frozenset({"mspace", "malignmark", "maligngroup", "mprescripts"})
_COLLAPSIBLE_TAGS = frozenset({"mrow", "mstyle", "mpadded", "mphantom", "merror"})
_LEAF_TAGS = frozenset({"mi", "mo", "mn", "mtext", "ms"})


class Node:
    """Expression tree node compatible with the zss simple_distance interface."""

    def __init__(self, label, children=None):
        self.label = label
        self.children = list(children or [])

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    def __repr__(self):
        return f"Node({self.label!r}, n_children={len(self.children)})"


def _local_tag(el):
    """Return the local tag name, stripping any XML namespace prefix."""
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _leaf_text(el):
    """Return normalized whitespace-collapsed text of a leaf MathML element."""
    return re.sub(r"\s+", " ", (el.text or "").strip())


def _build_node(el):
    """Recursively convert one MathML element to a Node tree.

    Returns None for pure formatting nodes with no semantic content.
    """
    tag = _local_tag(el)
    if not tag or tag in _DROP_TAGS:
        return None

    if tag in _LEAF_TAGS:
        text = _leaf_text(el)
        return Node(f"{tag}:{text}" if text else tag)

    children = [c for c in (_build_node(c) for c in el) if c is not None]

    if tag in _COLLAPSIBLE_TAGS:
        if len(children) == 1:
            return children[0]
        if len(children) == 0:
            return None

    return Node(tag, children)


def mathml_to_tree(table):
    """Build a normalized expression Node tree from a LaTeXML equation table.

    Returns the root Node, or None when no MathML is found or the tree is
    empty after normalization.
    """
    if table is None:
        return None

    math_els = table.xpath(".//math")
    if not math_els:
        return None

    math_el = math_els[0]

    semantics = math_el.find("semantics")
    if semantics is not None:
        for child in semantics:
            if _local_tag(child) != "annotation":
                return _build_node(child)
        return None

    children = [c for c in (_build_node(c) for c in math_el) if c is not None]
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return Node("math", children)


def _count_nodes(node):
    """Return total node count via recursive DFS."""
    return 1 + sum(_count_nodes(c) for c in node.children)


def _label_dist(label_a, label_b):
    """ZSS label cost: 0 if identical, 1 otherwise."""
    return 0 if label_a == label_b else 1


def tree_edit_distance(tree_a, tree_b):
    """Compute normalized TED similarity in [0, 1].

    Uses Zhang-Shasha (zss.simple_distance). Raw TED is normalized by
    max(nodes(A), nodes(B)) to give a bounded similarity score:
        sim = 1 - TED(A, B) / max(nodes(A), nodes(B))

    Returns 0.0 for None trees or trees larger than 200 nodes (multi-line
    arrays where ZSS would be O(n^2 m^2) and results are unreliable).
    """
    if tree_a is None or tree_b is None:
        return 0.0

    size_a = _count_nodes(tree_a)
    size_b = _count_nodes(tree_b)
    max_size = max(size_a, size_b)

    if max_size == 0 or max_size > 200:
        return 0.0

    ted = zss.simple_distance(
        tree_a, tree_b,
        get_children=Node.get_children,
        get_label=Node.get_label,
        label_dist=_label_dist,
    )
    return max(0.0, 1.0 - ted / max_size)
