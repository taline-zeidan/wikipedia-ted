import copy
from models.tree import TreeNode


def get_available_fields(root: TreeNode) -> list[str]:
    return [
        child.label
        for child in root.children
        if not child.is_content and child.label != "name"
    ]


def filter_tree(root: TreeNode, selected_fields: list[str]) -> TreeNode:
    filtered_root = TreeNode(label=root.label, is_content=root.is_content)

    for child in root.children:
        if child.label == "name" or child.label in selected_fields:
            filtered_root.add_child(copy.deepcopy(child))

    return filtered_root