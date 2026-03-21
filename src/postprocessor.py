import io
import os
import re
from xml.etree.ElementTree import Element, ElementTree, indent

from models.tree import TreeNode, TreeUtils


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
INFOBOX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "infoboxes")


_RESERVED_TAGS = {"name", "type", "id", "class"}

def _sanitize_tag(label: str) -> str:
    tag = re.sub(r"[^\w]", "_", label.strip())
    tag = re.sub(r"_+", "_", tag).strip("_")
    if not tag:
        return "node"
    if tag[0].isdigit():
        tag = "field_" + tag
    if len(tag) == 1 and tag.isalpha():
        tag = tag + "_node"
    if tag in _RESERVED_TAGS:
        tag = "field_" + tag
    return tag


def _tree_to_element(node: TreeNode) -> Element:
    tag = _sanitize_tag(node.label)
    element = Element(tag)

    content_children = [c for c in node.children if c.is_content]
    structural_children = [c for c in node.children if not c.is_content]

    if content_children:
        element.text = " ".join(c.label for c in content_children)

    for child in structural_children:
        element.append(_tree_to_element(child))

    return element


def tree_to_xml_string(root: TreeNode) -> str:
    element = _tree_to_element(root)
    xml_tree = ElementTree(element)
    indent(xml_tree, space="  ")

    buffer = io.StringIO()
    xml_tree.write(buffer, encoding="unicode", xml_declaration=True)
    return buffer.getvalue()

def _flatten_to_fields(node: TreeNode, prefix: str = "") -> dict[str, str]:
    fields: dict[str, str] = {}

    content_children = [c for c in node.children if c.is_content]
    structural_children = [c for c in node.children if not c.is_content]

    if content_children and not structural_children:
        fields[prefix] = " ".join(c.label for c in content_children)
    else:
        for child in structural_children:
            key = f"{prefix}__{child.label}" if prefix else child.label
            fields.update(_flatten_to_fields(child, key))

    return fields


def tree_to_infobox_string(root: TreeNode) -> str:
    country_name = ""
    for child in root.children:
        if child.label == "name" and child.children:
            country_name = child.children[0].label
            break

    fields = _flatten_to_fields(root)
    lines = ["{{Infobox country"]

    if country_name:
        lines.append(f"| conventional_long_name = {country_name}")

    for key, value in fields.items():
        if key in ("name", "country"):
            continue
        display_key = key.replace("__", "_")
        lines.append(f"| {display_key} = {value}")

    lines.append("}}")
    return "\n".join(lines)

def save_as_xml(root: TreeNode, country_name: str, output_dir: str = DATA_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = country_name.lower().replace(" ", "_") + ".xml"
    filepath = os.path.join(output_dir, filename)

    xml_string = tree_to_xml_string(root)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(xml_string)

    return filepath


def save_as_infobox(root: TreeNode, country_name: str, output_dir: str = INFOBOX_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = country_name.lower().replace(" ", "_") + ".txt"
    filepath = os.path.join(output_dir, filename)

    infobox_string = tree_to_infobox_string(root)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(infobox_string)

    return filepath


def postprocess(root: TreeNode, country_name: str, fmt: str = "xml") -> str:
    if fmt == "xml":
        return save_as_xml(root, country_name)
    elif fmt == "infobox":
        return save_as_infobox(root, country_name)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Choose 'xml' or 'infobox'.")