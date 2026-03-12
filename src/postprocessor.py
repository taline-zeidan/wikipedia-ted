import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree, indent

from models.tree import TreeNode, TreeUtils


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
INFOBOX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "infoboxes")


# ---------------------------------------------------------------------------
# Tree → XML Element
# ---------------------------------------------------------------------------

def _is_attribute_node(node: TreeNode) -> bool:
    return (
        not node.is_content
        and len(node.children) == 1
        and node.children[0].is_content
    )


def _tree_to_element(node: TreeNode) -> Element:
    element = Element(node.label)

    attribute_children = [c for c in node.children if _is_attribute_node(c)]
    structural_children = [c for c in node.children if not _is_attribute_node(c)]

    for attr_node in sorted(attribute_children, key=lambda n: n.label):
        element.set(attr_node.label, attr_node.children[0].label)

    content_children = [c for c in structural_children if c.is_content]
    element_children = [c for c in structural_children if not c.is_content]

    if content_children:
        element.text = " ".join(c.label for c in content_children)

    for child in element_children:
        element.append(_tree_to_element(child))

    return element


def tree_to_xml_string(root: TreeNode) -> str:
    element = _tree_to_element(root)
    xml_tree = ElementTree(element)
    indent(xml_tree, space="  ")

    import io
    buffer = io.StringIO()
    xml_tree.write(buffer, encoding="unicode", xml_declaration=True)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Tree → Wikipedia infobox wikitext
# ---------------------------------------------------------------------------

def _collect_infobox_fields(node: TreeNode) -> dict[str, str]:
    fields: dict[str, str] = {}

    for child in node.children:
        if child.is_content:
            continue

        if _is_attribute_node(child):
            continue

        content_tokens = [c.label for c in child.children if c.is_content]
        sub_elements = [c for c in child.children if not c.is_content and not _is_attribute_node(c)]

        if content_tokens and not sub_elements:
            fields[child.label] = " ".join(content_tokens)
        elif sub_elements:
            nested = _collect_infobox_fields(child)
            for sub_key, sub_value in nested.items():
                fields[f"{child.label}__{sub_key}"] = sub_value

    return fields


def tree_to_infobox_string(root: TreeNode) -> str:
    country_name = ""
    for child in root.children:
        if child.label == "name" and child.children:
            country_name = child.children[0].label
            break

    fields = _collect_infobox_fields(root)
    lines = [f"{{{{Infobox country"]

    if country_name:
        lines.append(f"| conventional_long_name = {country_name}")

    for key, value in fields.items():
        if key == "name":
            continue
        display_key = key.replace("__", "_")
        lines.append(f"| {display_key} = {value}")

    lines.append("}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save to file
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def postprocess(root: TreeNode, country_name: str, fmt: str = "xml") -> str:
    if fmt == "xml":
        return save_as_xml(root, country_name)
    elif fmt == "infobox":
        return save_as_infobox(root, country_name)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Choose 'xml' or 'infobox'.")