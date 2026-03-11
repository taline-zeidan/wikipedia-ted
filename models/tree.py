from __future__ import annotations
from typing import List, Optional


class TreeNode:
    def __init__(self, label: str, is_content: bool = False) -> None:
        self.label: str = label
        self.is_content: bool = is_content
        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.postorder_index: Optional[int] = None

    def add_child(self, child: TreeNode) -> None:
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def __repr__(self) -> str:
        kind = "content" if self.is_content else "structural"
        return f"TreeNode(label={self.label!r}, type={kind}, children={len(self.children)})"


class TreeUtils:
    @staticmethod
    def postorder(root: TreeNode) -> List[TreeNode]:
        result: List[TreeNode] = []

        def traverse(node: TreeNode) -> None:
            for child in node.children:
                traverse(child)
            node.postorder_index = len(result)
            result.append(node)

        traverse(root)
        return result

    @staticmethod
    def leftmost_leaf(node: TreeNode) -> TreeNode:
        current = node
        while current.children:
            current = current.children[0]
        return current

    @staticmethod
    def get_leftmost_leaf_index(node: TreeNode, postorder_nodes: List[TreeNode]) -> int:
        leftmost = TreeUtils.leftmost_leaf(node)
        return leftmost.postorder_index

    @staticmethod
    def compute_keyroots(postorder_nodes: List[TreeNode]) -> List[int]:
        leftmost_map: dict[int, int] = {}
        for i, node in enumerate(postorder_nodes):
            leftmost_index = TreeUtils.get_leftmost_leaf_index(node, postorder_nodes)
            leftmost_map[leftmost_index] = i

        return sorted(leftmost_map.values())

    @staticmethod
    def size(root: TreeNode) -> int:
        return len(TreeUtils.postorder(root))

    @staticmethod
    def depth(node: TreeNode) -> int:
        count = 0
        current = node
        while current.parent is not None:
            current = current.parent
            count += 1
        return count

    @staticmethod
    def pretty_print(root: TreeNode, indent: int = 0) -> None:
        prefix = "  " * indent
        tag = "[content]" if root.is_content else "[structural]"
        print(f"{prefix}{tag} {root.label}")
        for child in root.children:
            TreeUtils.pretty_print(child, indent + 1)