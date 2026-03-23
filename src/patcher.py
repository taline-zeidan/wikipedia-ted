import os
import copy
import xml.etree.ElementTree as ET

from models.tree import TreeNode, TreeUtils
from src.ted import EditOperation, EditScript, element_to_subtree


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
        sed = op_el.get("string_edit_distance")
        pos = op_el.get("position")

        subtree_xml = None
        subtree_wrapper = op_el.find("subtree")
        if subtree_wrapper is not None:
            node_el = subtree_wrapper.find("node")
            if node_el is not None:
                subtree_xml = ET.tostring(node_el, encoding="unicode")

        operations.append(EditOperation(
            operation=op_el.get("type", ""),
            node_label=op_el.get("node_label", ""),
            path=path,
            target_label=op_el.get("target_label"),
            is_content=op_el.get("is_content", "False") == "True",
            string_edit_distance=int(sed) if sed is not None else None,
            position=int(pos) if pos is not None else None,
            subtree_xml=subtree_xml,
        ))

    return EditScript(
        source_country=source,
        target_country=target,
        ted_score=ted_score,
        operations=operations,
    )


def _get_node(root: TreeNode, path: list[str]) -> TreeNode | None:
    return TreeUtils.get_node_by_path(root, path)


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

    if operation.subtree_xml is not None:
        subtree_el = ET.fromstring(operation.subtree_xml)
        new_node = element_to_subtree(subtree_el)
    else:
        new_node = TreeNode(label=operation.node_label, is_content=operation.is_content)

    new_node.parent = parent
    pos = operation.position
    if pos is not None and 0 <= pos <= len(parent.children):
        parent.children.insert(pos, new_node)
    else:
        parent.children.append(new_node)


def _apply_operations(root: TreeNode, operations: list[EditOperation]) -> TreeNode:
    renames = [op for op in operations if op.operation == "RENAME"]
    deletes = [op for op in operations if op.operation == "DELETE"]
    inserts = [op for op in operations if op.operation == "INSERT"]

    for op in renames:
        _apply_rename(root, op)

    for op in sorted(deletes, key=lambda op: len(op.path), reverse=True):
        _apply_delete(root, op)

    subtree_inserts = [op for op in inserts if op.subtree_xml is not None]
    leaf_inserts = [op for op in inserts if op.subtree_xml is None]

    already_covered: set[str] = set()
    for op in sorted(subtree_inserts, key=lambda op: len(op.path)):
        key = "/".join(op.path) + "/" + op.node_label
        _apply_insert(root, op)
        already_covered.add(key)

    for op in sorted(leaf_inserts, key=lambda op: len(op.path)):
        parent_key = "/".join(op.path)
        is_child_of_subtree = False
        for covered in already_covered:
            if parent_key.startswith(covered):
                is_child_of_subtree = True
                break
        if is_child_of_subtree:
            continue
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
