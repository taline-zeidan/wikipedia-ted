import os
import xml.etree.ElementTree as ET

from models.tree import TreeNode, TreeUtils
from src.ted import EditOperation, EditScript


DIFFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "diffs")


# ---------------------------------------------------------------------------
# Edit script deserialization
# ---------------------------------------------------------------------------

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
        operations.append(EditOperation(
            operation=op_el.get("type", ""),
            node_label=op_el.get("node_label", ""),
            postorder_index=int(op_el.get("postorder_index", "0")),
            target_label=op_el.get("target_label"),
            is_content=op_el.get("is_content", "False") == "True",
        ))

    return EditScript(
        source_country=source,
        target_country=target,
        ted_score=ted_score,
        operations=operations,
    )


# ---------------------------------------------------------------------------
# Operation handlers
# ---------------------------------------------------------------------------

def _apply_rename(nodes: list[TreeNode], operation: EditOperation) -> None:
    index = operation.postorder_index
    if index < 0 or index >= len(nodes):
        return
    nodes[index].label = operation.target_label


def _apply_delete(nodes: list[TreeNode], operation: EditOperation) -> None:
    index = operation.postorder_index
    if index < 0 or index >= len(nodes):
        return

    node = nodes[index]
    parent = node.parent

    if parent is None:
        return

    if node in parent.children:
        parent.children.remove(node)
        node.parent = None


def _apply_insert(root: TreeNode, nodes: list[TreeNode], operation: EditOperation) -> None:
    index = operation.postorder_index
    new_node = TreeNode(label=operation.node_label, is_content=operation.is_content)

    if not nodes:
        return

    if index < len(nodes):
        sibling = nodes[index]
        parent = sibling.parent if sibling.parent else root
        insert_position = parent.children.index(sibling) if sibling in parent.children else len(parent.children)
        parent.children.insert(insert_position, new_node)
        new_node.parent = parent
    else:
        root.add_child(new_node)


# ---------------------------------------------------------------------------
# Core patch logic
# ---------------------------------------------------------------------------

def _apply_operations(root: TreeNode, operations: list[EditOperation]) -> TreeNode:
    renames = [op for op in operations if op.operation == "RENAME"]
    deletes = sorted(
        [op for op in operations if op.operation == "DELETE"],
        key=lambda op: op.postorder_index,
        reverse=True,
    )
    inserts = sorted(
        [op for op in operations if op.operation == "INSERT"],
        key=lambda op: op.postorder_index,
    )

    nodes = TreeUtils.postorder(root)
    for op in renames:
        _apply_rename(nodes, op)

    nodes = TreeUtils.postorder(root)
    for op in deletes:
        _apply_delete(nodes, op)
        nodes = TreeUtils.postorder(root)

    for op in inserts:
        nodes = TreeUtils.postorder(root)
        _apply_insert(root, nodes, op)

    return root


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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