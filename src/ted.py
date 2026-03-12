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
    postorder_index: int
    target_label: Optional[str] = None
    is_content: bool = False


@dataclass
class EditScript:
    source_country: str
    target_country: str
    ted_score: int
    operations: list[EditOperation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.operations)


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def _rename_cost(n1: TreeNode, n2: TreeNode) -> int:
    if n1.label == n2.label:
        return 0
    return 1


def _insert_cost(_: TreeNode) -> int:
    return 1


def _delete_cost(_: TreeNode) -> int:
    return 1


# ---------------------------------------------------------------------------
# Zhang-Shasha DP
# ---------------------------------------------------------------------------

def _compute_leftmost_leaves(postorder_nodes: list[TreeNode]) -> list[int]:
    leftmost = []
    for node in postorder_nodes:
        leftmost.append(TreeUtils.get_leftmost_leaf_index(node, postorder_nodes))
    return leftmost


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
) -> list[list[int]]:
    n = len(nodes1)
    m = len(nodes2)

    td = [[0] * m for _ in range(n)]

    kr1 = _compute_keyroots(nodes1, lm1)
    kr2 = _compute_keyroots(nodes2, lm2)

    for i in kr1:
        for j in kr2:
            fd = _compute_forest_distance(i, j, nodes1, nodes2, lm1, lm2, td)
            td[i][j] = fd[i - lm1[i] + 1][j - lm2[j] + 1]

    return td


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


# ---------------------------------------------------------------------------
# Edit script extraction
# ---------------------------------------------------------------------------

def _extract_operations(
    td: list[list[int]],
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
) -> list[EditOperation]:
    operations: list[EditOperation] = []
    _backtrack(len(nodes1) - 1, len(nodes2) - 1, td, nodes1, nodes2, lm1, lm2, operations)
    return operations


def _backtrack(
    i: int,
    j: int,
    td: list[list[int]],
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
    operations: list[EditOperation],
) -> None:
    if i < 0 and j < 0:
        return
    if i < 0:
        for y in range(j, -1, -1):
            operations.append(EditOperation(
                operation="INSERT",
                node_label=nodes2[y].label,
                postorder_index=y,
                is_content=nodes2[y].is_content,
            ))
        return
    if j < 0:
        for x in range(i, -1, -1):
            operations.append(EditOperation(
                operation="DELETE",
                node_label=nodes1[x].label,
                postorder_index=x,
                is_content=nodes1[x].is_content,
            ))
        return

    node1 = nodes1[i]
    node2 = nodes2[j]
    current = td[i][j]

    delete_cost = td[i - 1][j] + _delete_cost(node1) if i > 0 else float("inf")
    insert_cost = td[i][j - 1] + _insert_cost(node2) if j > 0 else float("inf")
    rename_cost = (td[i - 1][j - 1] if i > 0 and j > 0 else 0) + _rename_cost(node1, node2)

    if current == rename_cost:
        if node1.label != node2.label:
            operations.append(EditOperation(
                operation="RENAME",
                node_label=node1.label,
                postorder_index=i,
                target_label=node2.label,
                is_content=node1.is_content,
            ))
        _backtrack(i - 1, j - 1, td, nodes1, nodes2, lm1, lm2, operations)

    elif current == delete_cost:
        operations.append(EditOperation(
            operation="DELETE",
            node_label=node1.label,
            postorder_index=i,
            is_content=node1.is_content,
        ))
        _backtrack(i - 1, j, td, nodes1, nodes2, lm1, lm2, operations)

    else:
        operations.append(EditOperation(
            operation="INSERT",
            node_label=node2.label,
            postorder_index=j,
            is_content=node2.is_content,
        ))
        _backtrack(i, j - 1, td, nodes1, nodes2, lm1, lm2, operations)


# ---------------------------------------------------------------------------
# Edit script serialization
# ---------------------------------------------------------------------------

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
        op_el.set("postorder_index", str(op.postorder_index))
        op_el.set("is_content", str(op.is_content))
        if op.target_label is not None:
            op_el.set("target_label", op.target_label)

    tree = ElementTree(root)
    indent(tree, space="  ")
    return tree


def save_edit_script(script: EditScript) -> str:
    os.makedirs(DIFFS_DIR, exist_ok=True)
    filename = f"{script.source_country.lower().replace(' ', '_')}_{script.target_country.lower().replace(' ', '_')}.xml"
    filepath = os.path.join(DIFFS_DIR, filename)
    xml_tree = _serialize_edit_script(script)
    xml_tree.write(filepath, encoding="unicode", xml_declaration=True)
    return filepath


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ted(t1: TreeNode, t2: TreeNode, source: str = "T1", target: str = "T2") -> EditScript:
    nodes1 = TreeUtils.postorder(t1)
    nodes2 = TreeUtils.postorder(t2)

    lm1 = _compute_leftmost_leaves(nodes1)
    lm2 = _compute_leftmost_leaves(nodes2)

    td = _zhang_shasha(nodes1, nodes2, lm1, lm2)

    ted_score = td[len(nodes1) - 1][len(nodes2) - 1]
    operations = _extract_operations(td, nodes1, nodes2, lm1, lm2)

    return EditScript(
        source_country=source,
        target_country=target,
        ted_score=ted_score,
        operations=operations,
    )