import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from xml.etree.ElementTree import ElementTree, Element, SubElement, indent

from models.tree import TreeNode, TreeUtils


DIFFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "diffs")


@dataclass
class EditOperation:
    operation: str
    node_label: str
    path: list[str]
    target_label: Optional[str] = None
    is_content: bool = False
    string_edit_distance: Optional[int] = None
    position: Optional[int] = None
    subtree_xml: Optional[str] = None


@dataclass
class EditScript:
    source_country: str
    target_country: str
    ted_score: int
    operations: list[EditOperation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.operations)



def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(m):
        cur = [i + 1] + [0] * n
        for j in range(n):
            cost = 0 if a[i] == b[j] else 1
            cur[j + 1] = min(cur[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = cur
    return prev[n]


def _rename_cost(n1: TreeNode, n2: TreeNode) -> int:
    if n1.label == n2.label:
        return 0
    if n1.is_content != n2.is_content:
        return 10000
    if not n1.is_content and not n2.is_content:
        return 10000
    lev = levenshtein_distance(n1.label, n2.label)
    return min(lev, _delete_cost(n1) + _insert_cost(n2))


def _insert_cost(_: TreeNode) -> int:
    return 1


def _delete_cost(_: TreeNode) -> int:
    return 1



def _compute_leftmost_leaves(postorder_nodes: list[TreeNode]) -> list[int]:
    return [TreeUtils.get_leftmost_leaf_index(node, postorder_nodes) for node in postorder_nodes]


def _compute_keyroots(postorder_nodes: list[TreeNode], leftmost: list[int]) -> list[int]:
    seen: dict[int, int] = {}
    for i, ll in enumerate(leftmost):
        seen[ll] = i
    return sorted(seen.values())


def _zhang_shasha(
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
) -> tuple[list[list[int]], dict[tuple[int, int], list[list[int]]]]:
    n = len(nodes1)
    m = len(nodes2)

    td = [[0] * m for _ in range(n)]
    fd_store: dict[tuple[int, int], list[list[int]]] = {}

    kr1 = _compute_keyroots(nodes1, lm1)
    kr2 = _compute_keyroots(nodes2, lm2)

    for i in kr1:
        for j in kr2:
            fd = _compute_forest_distance(i, j, nodes1, nodes2, lm1, lm2, td)
            fd_store[(i, j)] = fd
            td[i][j] = fd[i - lm1[i] + 1][j - lm2[j] + 1]

    return td, fd_store


def _compute_forest_distance(
    i: int,
    j: int,
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
    td: list[list[int]],
) -> list[list[int]]:
    i_offset = lm1[i]
    j_offset = lm2[j]
    rows = i - i_offset + 2
    cols = j - j_offset + 2

    fd = [[0] * cols for _ in range(rows)]

    for x in range(1, rows):
        fd[x][0] = fd[x - 1][0] + _delete_cost(nodes1[i_offset + x - 1])
    for y in range(1, cols):
        fd[0][y] = fd[0][y - 1] + _insert_cost(nodes2[j_offset + y - 1])

    for x in range(1, rows):
        for y in range(1, cols):
            node1 = nodes1[i_offset + x - 1]
            node2 = nodes2[j_offset + y - 1]
            i_idx = i_offset + x - 1
            j_idx = j_offset + y - 1

            cost_delete = fd[x - 1][y] + _delete_cost(node1)
            cost_insert = fd[x][y - 1] + _insert_cost(node2)

            if lm1[i_idx] == lm1[i] and lm2[j_idx] == lm2[j]:
                cost_rename = fd[x - 1][y - 1] + _rename_cost(node1, node2)
                fd[x][y] = min(cost_delete, cost_insert, cost_rename)
                td[i_idx][j_idx] = fd[x][y]
            else:
                mapped_x = lm1[i_idx] - i_offset
                mapped_y = lm2[j_idx] - j_offset
                cost_subtree = fd[mapped_x][mapped_y] + td[i_idx][j_idx]
                fd[x][y] = min(cost_delete, cost_insert, cost_subtree)

    return fd




def _extract_mapping(
    td: list[list[int]],
    fd_store: dict[tuple[int, int], list[list[int]]],
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    root1 = len(nodes1) - 1
    root2 = len(nodes2) - 1
    _backtrack_mapping(root1, root2, fd_store, nodes1, nodes2, lm1, lm2, td, mapping)
    return mapping


def _backtrack_mapping(
    i: int,
    j: int,
    fd_store: dict[tuple[int, int], list[list[int]]],
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
    td: list[list[int]],
    mapping: dict[int, int],
) -> None:
    if (i, j) not in fd_store:
        return

    fd = fd_store[(i, j)]
    i_offset = lm1[i]
    j_offset = lm2[j]

    x = i - i_offset + 1
    y = j - j_offset + 1

    while x > 0 and y > 0:
        i_idx = i_offset + x - 1
        j_idx = j_offset + y - 1
        node1 = nodes1[i_idx]
        node2 = nodes2[j_idx]

        cost_delete = fd[x - 1][y] + _delete_cost(node1)
        cost_insert = fd[x][y - 1] + _insert_cost(node2)

        if lm1[i_idx] == lm1[i] and lm2[j_idx] == lm2[j]:
            cost_rename = fd[x - 1][y - 1] + _rename_cost(node1, node2)
            if fd[x][y] == cost_rename:
                mapping[i_idx] = j_idx
                x -= 1
                y -= 1
            elif fd[x][y] == cost_delete:
                x -= 1
            else:
                y -= 1
        else:
            mapped_x = lm1[i_idx] - i_offset
            mapped_y = lm2[j_idx] - j_offset
            cost_subtree = fd[mapped_x][mapped_y] + td[i_idx][j_idx]

            if fd[x][y] == cost_subtree:
                _backtrack_mapping(
                    i_idx, j_idx, fd_store,
                    nodes1, nodes2, lm1, lm2, td, mapping,
                )
                x = mapped_x
                y = mapped_y
            elif fd[x][y] == cost_delete:
                x -= 1
            else:
                y -= 1




def _mapping_to_operations(
    mapping: dict[int, int],
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
) -> list[EditOperation]:
    operations: list[EditOperation] = []
    matched_source = set(mapping.keys())
    matched_target = set(mapping.values())
    reverse_map = {j: i for i, j in mapping.items()}

    for src_idx, tgt_idx in mapping.items():
        n1 = nodes1[src_idx]
        n2 = nodes2[tgt_idx]
        if n1.label == n2.label:
            continue
        if n1.is_content and n2.is_content:
            operations.append(EditOperation(
                operation="RENAME",
                node_label=n1.label,
                path=TreeUtils.get_path(n1),
                target_label=n2.label,
                is_content=True,
                string_edit_distance=levenshtein_distance(n1.label, n2.label),
            ))

    for src_idx, n1 in enumerate(nodes1):
        if src_idx not in matched_source:
            operations.append(EditOperation(
                operation="DELETE",
                node_label=n1.label,
                path=TreeUtils.get_path(n1),
                is_content=n1.is_content,
            ))

    for tgt_idx, n2 in enumerate(nodes2):
        if tgt_idx in matched_target:
            continue

        parent_t2 = n2.parent
        position = parent_t2.children.index(n2) if parent_t2 else 0

        if parent_t2 is not None and parent_t2.postorder_index in matched_target:
            src_parent_idx = reverse_map[parent_t2.postorder_index]
            parent_path = TreeUtils.get_path(nodes1[src_parent_idx])
        else:
            parent_path = TreeUtils.get_path(parent_t2) if parent_t2 else []

        is_topmost = (parent_t2 is not None and parent_t2.postorder_index in matched_target)
        subtree_xml = None
        if is_topmost and not n2.is_content and n2.children:
            if _all_descendants_unmatched(n2, matched_target):
                subtree_xml = _serialize_subtree_to_xml(n2)

        operations.append(EditOperation(
            operation="INSERT",
            node_label=n2.label,
            path=parent_path,
            is_content=n2.is_content,
            position=position,
            subtree_xml=subtree_xml,
        ))

    return operations


def _all_descendants_unmatched(node: TreeNode, matched_target: set[int]) -> bool:
    if node.postorder_index in matched_target:
        return False
    return all(_all_descendants_unmatched(c, matched_target) for c in node.children)



def _serialize_subtree_to_xml(node: TreeNode) -> str:
    el = _subtree_to_element(node)
    tree = ElementTree(el)
    indent(tree, space="  ")
    buf = io.StringIO()
    tree.write(buf, encoding="unicode")
    return buf.getvalue()


def _subtree_to_element(node: TreeNode) -> Element:
    el = Element("node")
    el.set("label", node.label)
    el.set("is_content", str(node.is_content))
    for child in node.children:
        el.append(_subtree_to_element(child))
    return el


def element_to_subtree(el: Element) -> TreeNode:
    node = TreeNode(
        label=el.get("label", ""),
        is_content=el.get("is_content", "False") == "True",
    )
    for child_el in el.findall("node"):
        child = element_to_subtree(child_el)
        node.add_child(child)
    return node




def _serialize_edit_script(script: EditScript) -> ElementTree:
    root = Element("edit_script")
    root.set("source", script.source_country)
    root.set("target", script.target_country)
    root.set("ted_score", str(script.ted_score))
    root.set("operation_count", str(len(script.operations)))

    for op in script.operations:
        op_el = SubElement(root, "operation")
        op_el.set("type", op.operation)
        op_el.set("node_label", op.node_label)
        op_el.set("path", "/".join(op.path))
        op_el.set("is_content", str(op.is_content))
        if op.target_label is not None:
            op_el.set("target_label", op.target_label)
        if op.string_edit_distance is not None:
            op_el.set("string_edit_distance", str(op.string_edit_distance))
        if op.position is not None:
            op_el.set("position", str(op.position))
        if op.subtree_xml is not None:
            subtree_wrapper = SubElement(op_el, "subtree")
            subtree_el = ET.fromstring(op.subtree_xml)
            subtree_wrapper.append(subtree_el)

    tree = ElementTree(root)
    indent(tree, space="  ")
    return tree


def save_edit_script(script: EditScript) -> str:
    os.makedirs(DIFFS_DIR, exist_ok=True)
    filename = (
        f"{script.source_country.lower().replace(' ', '_')}_"
        f"{script.target_country.lower().replace(' ', '_')}.xml"
    )
    filepath = os.path.join(DIFFS_DIR, filename)
    xml_tree = _serialize_edit_script(script)
    xml_tree.write(filepath, encoding="unicode", xml_declaration=True)
    return filepath



def compute_ted(t1: TreeNode, t2: TreeNode, source: str = "T1", target: str = "T2") -> EditScript:
    nodes1 = TreeUtils.postorder(t1)
    nodes2 = TreeUtils.postorder(t2)

    lm1 = _compute_leftmost_leaves(nodes1)
    lm2 = _compute_leftmost_leaves(nodes2)

    td, fd_store = _zhang_shasha(nodes1, nodes2, lm1, lm2)

    ted_score = td[len(nodes1) - 1][len(nodes2) - 1]
    mapping = _extract_mapping(td, fd_store, nodes1, nodes2, lm1, lm2)
    operations = _mapping_to_operations(mapping, nodes1, nodes2)

    return EditScript(
        source_country=source,
        target_country=target,
        ted_score=ted_score,
        operations=operations,
    )
