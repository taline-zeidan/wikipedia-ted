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


@dataclass
class EditScript:
    source_country: str
    target_country: str
    ted_score: int
    operations: list[EditOperation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.operations)


def levenshtein_distance(a: str, b: str) -> int:
    """Unit-cost Levenshtein (insert/delete/substitute) on Unicode code points."""
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
    if not n1.is_content and not n2.is_content:
        return 10000
    # Never charge more than delete+insert for a leaf replacement, so TED prefers RENAME
    # over spurious DELETE+INSERT when strings differ a lot.
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

def _extract_operations(
    td: list[list[int]],
    fd_store: dict[tuple[int, int], list[list[int]]],
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
) -> list[EditOperation]:
    raw_ops: list[tuple[str, int, Optional[int], bool]] = []

    kr1 = _compute_keyroots(nodes1, lm1)
    kr2 = _compute_keyroots(nodes2, lm2)

    for i in reversed(kr1):
        for j in reversed(kr2):
            if (i, j) not in fd_store:
                continue
            fd = fd_store[(i, j)]
            _backtrack_fd(i, j, fd, nodes1, nodes2, lm1, lm2, td, fd_store, raw_ops)

    seen: set[tuple] = set()
    operations: list[EditOperation] = []

    for op_type, idx1, idx2, is_content in raw_ops:

        if op_type == "RENAME":
            node1 = nodes1[idx1]
            node2 = nodes2[idx2]
            key = ("RENAME", idx1)
            if key not in seen:
                seen.add(key)
                operations.append(EditOperation(
                    operation="RENAME",
                    node_label=node1.label,
                    path=TreeUtils.get_path(node1),
                    target_label=node2.label,
                    is_content=node1.is_content,
                    string_edit_distance=levenshtein_distance(node1.label, node2.label),
                ))
        elif op_type == "DELETE":
            node1 = nodes1[idx1]
            key = ("DELETE", idx1)
            if key not in seen:
                seen.add(key)
                operations.append(EditOperation(
                    operation="DELETE",
                    node_label=node1.label,
                    path=TreeUtils.get_path(node1),
                    is_content=node1.is_content,
                ))
        elif op_type == "INSERT":
            node2 = nodes2[idx2]
            parent_path = TreeUtils.get_path(node2.parent) if node2.parent else []
            key = ("INSERT", idx2)
            if key not in seen:
                seen.add(key)
                operations.append(EditOperation(
                    operation="INSERT",
                    node_label=node2.label,
                    path=parent_path,
                    is_content=node2.is_content,
                ))

    return operations


def _backtrack_fd(
    i: int,
    j: int,
    fd: list[list[int]],
    nodes1: list[TreeNode],
    nodes2: list[TreeNode],
    lm1: list[int],
    lm2: list[int],
    td: list[list[int]],
    fd_store: dict[tuple[int, int], list[list[int]]],
    raw_ops: list,
) -> None:
    i_offset = lm1[i]
    j_offset = lm2[j]

    x = i - i_offset + 1
    y = j - j_offset + 1

    while x > 0 or y > 0:
        if x == 0:
            y -= 1
            j_idx = j_offset + y
            raw_ops.append(("INSERT", None, j_idx, nodes2[j_idx].is_content))
        elif y == 0:
            x -= 1
            i_idx = i_offset + x
            raw_ops.append(("DELETE", i_idx, None, nodes1[i_idx].is_content))
        else:
            i_idx = i_offset + x - 1
            j_idx = j_offset + y - 1
            node1 = nodes1[i_idx]
            node2 = nodes2[j_idx]

            cost_delete = fd[x - 1][y] + _delete_cost(node1)
            cost_insert = fd[x][y - 1] + _insert_cost(node2)

            if lm1[i_idx] == lm1[i] and lm2[j_idx] == lm2[j]:
                cost_rename = fd[x - 1][y - 1] + _rename_cost(node1, node2)
                if fd[x][y] == cost_rename:
                    if node1.label != node2.label:
                        raw_ops.append(("RENAME", i_idx, j_idx, node1.is_content))
                    x -= 1
                    y -= 1
                elif fd[x][y] == cost_delete:
                    raw_ops.append(("DELETE", i_idx, None, node1.is_content))
                    x -= 1
                else:
                    raw_ops.append(("INSERT", None, j_idx, node2.is_content))
                    y -= 1
            else:
                mapped_x = lm1[i_idx] - i_offset
                mapped_y = lm2[j_idx] - j_offset
                cost_subtree = fd[mapped_x][mapped_y] + td[i_idx][j_idx]

                if fd[x][y] == cost_subtree:
                    if (i_idx, j_idx) in fd_store:
                        _backtrack_fd(
                            i_idx, j_idx, fd_store[(i_idx, j_idx)],
                            nodes1, nodes2, lm1, lm2, td, fd_store, raw_ops,
                        )
                    x = mapped_x
                    y = mapped_y
                elif fd[x][y] == cost_delete:
                    raw_ops.append(("DELETE", i_idx, None, node1.is_content))
                    x -= 1
                else:
                    raw_ops.append(("INSERT", None, j_idx, node2.is_content))
                    y -= 1

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

def _clean_operations(operations: list[EditOperation], t1: TreeNode, t2: TreeNode) -> list[EditOperation]:
    clean: list[EditOperation] = []

    for op in operations:
        if op.operation == "RENAME":
            node = TreeUtils.get_node_by_path(t1, op.path)
            if node is None or node.label != op.node_label:
                continue
            clean.append(op)

        elif op.operation == "DELETE":
            node = TreeUtils.get_node_by_path(t1, op.path)
            if node is None:
                continue
            clean.append(op)

        elif op.operation == "INSERT":
            parent = TreeUtils.get_node_by_path(t1, op.path)
            if parent is None:
                continue
            clean.append(op)

    return clean

def _sweep_missing_deletes(
    operations: list[EditOperation], t1: TreeNode, t2: TreeNode
) -> list[EditOperation]:
    already_deleted = {tuple(op.path) for op in operations if op.operation == "DELETE"}
    already_renamed = {tuple(op.path) for op in operations if op.operation == "RENAME"}

    new_deletes: list[EditOperation] = []

    def walk(node: TreeNode) -> None:
        path = TreeUtils.get_path(node)
        t2_node = TreeUtils.get_node_by_path(t2, path)
        if (
            t2_node is None
            and tuple(path) not in already_deleted
            and tuple(path) not in already_renamed
        ):
            ancestor_deleted = any(
                tuple(path[:i]) in already_deleted
                for i in range(1, len(path))
            )
            if not ancestor_deleted:
                new_deletes.append(EditOperation(
                    operation="DELETE",
                    node_label=node.label,
                    path=path,
                    is_content=node.is_content,
                ))
        for child in node.children:
            walk(child)

    walk(t1)
    return operations + new_deletes


def compute_ted(t1: TreeNode, t2: TreeNode, source: str = "T1", target: str = "T2") -> EditScript:
    nodes1 = TreeUtils.postorder(t1)
    nodes2 = TreeUtils.postorder(t2)

    lm1 = _compute_leftmost_leaves(nodes1)
    lm2 = _compute_leftmost_leaves(nodes2)

    td, fd_store = _zhang_shasha(nodes1, nodes2, lm1, lm2)

    ted_score = td[len(nodes1) - 1][len(nodes2) - 1]
    raw_operations = _extract_operations(td, fd_store, nodes1, nodes2, lm1, lm2)

    raw_rename_count = sum(1 for op in raw_operations if op.operation == "RENAME")
    raw_delete_count = sum(1 for op in raw_operations if op.operation == "DELETE")
    raw_insert_count = sum(1 for op in raw_operations if op.operation == "INSERT")

    print("RAW Renames:", raw_rename_count)
    print("RAW Deletes:", raw_delete_count)
    print("RAW Inserts:", raw_insert_count)

    operations = _clean_operations(raw_operations, t1, t2)

    return EditScript(
        source_country=source,
        target_country=target,
        ted_score=ted_score,
        operations=operations,
    )