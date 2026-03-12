import os
import xml.etree.ElementTree as ET

from models.tree import TreeNode


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _build_tree(element: ET.Element) -> TreeNode:
    node = TreeNode(label=element.tag, is_content=False)

    for attr_name in sorted(element.attrib.keys()):
        attr_node = TreeNode(label=attr_name, is_content=False)
        attr_value_node = TreeNode(label=element.attrib[attr_name], is_content=True)
        attr_node.add_child(attr_value_node)
        node.add_child(attr_node)

    if element.text and element.text.strip():
        for token in _tokenize(element.text.strip()):
            token_node = TreeNode(label=token, is_content=True)
            node.add_child(token_node)

    for child_element in element:
        child_node = _build_tree(child_element)
        node.add_child(child_node)

    return node


def _tokenize(text: str) -> list[str]:
    tokens = []
    current = []

    for char in text:
        if char in (" ", "\t", "\n", ",", ";", ".", "!", "?", ":", "/", "-"):
            if current:
                tokens.append("".join(current))
                current = []
        elif char.isupper() and current and current[-1].islower():
            tokens.append("".join(current))
            current = [char]
        else:
            current.append(char)

    if current:
        tokens.append("".join(current))

    return [t for t in tokens if t]


def load_tree(country_name: str) -> TreeNode:
    filename = country_name.lower().replace(" ", "_") + ".xml"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No XML file found for country: {country_name} at {filepath}")

    xml_tree = ET.parse(filepath)
    root_element = xml_tree.getroot()

    return _build_tree(root_element)


def load_tree_from_file(filepath: str) -> TreeNode:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    xml_tree = ET.parse(filepath)
    root_element = xml_tree.getroot()

    return _build_tree(root_element)