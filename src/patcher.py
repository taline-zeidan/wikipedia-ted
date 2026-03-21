import os
import copy
import xml.etree.ElementTree as ET

from models.tree import TreeNode, TreeUtils
from src.ted import EditOperation, EditScript


DIFFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "diffs")


def _load_edit_script(filepath: str) -> EditScript:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Diff file not found: {filepath}")

    xml_tree = ET.parse(filepath)
    root = xml_tree.getroot()

    source = root.get("source", "")
    target = root.get("target", "")
    ted_score = int(root.get("ted_score", "0"))

    operations: list[EditOperation] = []
    for op_el in root.findall("operation"):
        path_str = op_el.get("path", "")
        path = path_str.split("/") if path_str else []
        operations.append(EditOperation(
            operation=op_el.get("type", ""),
            node_label=op_el.get("node_label", ""),
            path=path,
            target_label=op_el.get("target_label"),
            is_content=op_el.get("is_content", "False") == "True",
        ))

    return EditScript(
        source_country=source,
        target_country=target,
        ted_score=ted_score,
        operations=operations,
    )


def _get_node(root: TreeNode, path: list[str]) -> TreeNode | None:
    return TreeUtils.get_node_by_path(root, path)


def _get_parent(root: TreeNode, path: list[str]) -> TreeNode | None:
    if len(path) < 2:
        return None
    return TreeUtils.get_node_by_path(root, path[:-1])


def _apply_rename(root: TreeNode, operation: EditOperation) -> None:
    node = _get_node(root, operation.path)
    if node is None:
        return
    node.label = operation.target_label


def _apply_delete(root: TreeNode, operation: EditOperation) -> None:
    node = _get_node(root, operation.path)
    if node is None:
        return
    parent = node.parent
    if parent is None:
        return
    if node in parent.children:
        parent.children.remove(node)
        node.parent = None


def _apply_insert(root: TreeNode, operation: EditOperation) -> None:
    parent = _get_node(root, operation.path)
    if parent is None:
        return
    new_node = TreeNode(label=operation.node_label, is_content=operation.is_content)
    parent.add_child(new_node)


def _prune_empty_structural_nodes(root: TreeNode) -> None:
    changed = True
    while changed:
        changed = False
        nodes = TreeUtils.postorder(root)
        for node in nodes:
            if (
                not node.is_content
                and node.parent is not None
                and len(node.children) == 0
            ):
                node.parent.children.remove(node)
                node.parent = None
                changed = True
                break


def _apply_operations(root: TreeNode, operations: list[EditOperation]) -> TreeNode:
    renames = [op for op in operations if op.operation == "RENAME"]
    deletes = [op for op in operations if op.operation == "DELETE"]
    inserts = [op for op in operations if op.operation == "INSERT"]

    for op in renames:
        _apply_rename(root, op)

    for op in sorted(deletes, key=lambda op: len(op.path), reverse=True):
        _apply_delete(root, op)

    _prune_empty_structural_nodes(root)

    for op in inserts:
        _apply_insert(root, op)

    return root


def patch(tree: TreeNode, edit_script: EditScript) -> TreeNode:
    return _apply_operations(tree, edit_script.operations)


def patch_from_file(tree: TreeNode, filepath: str) -> TreeNode:
    edit_script = _load_edit_script(filepath)
    return _apply_operations(tree, edit_script.operations)


def patch_countries(source_country: str, target_country: str, tree: TreeNode) -> TreeNode:
    filename = (
        f"{source_country.lower().replace(' ', '_')}_"
        f"{target_country.lower().replace(' ', '_')}.xml"
    )
    filepath = os.path.join(DIFFS_DIR, filename)
    return patch_from_file(tree, filepath)