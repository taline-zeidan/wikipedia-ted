import os
import re
import xml.etree.ElementTree as ET

from models.tree import TreeNode


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Fields stored as a single atomic content node
ATOMIC_FIELDS = {
    "national_motto", "englishmotto", "government_type", "demonym",
    "sovereignty_type", "languages_type", "languages2_type",
    "conventional_long_name", "native_name", "capital", "largest_city",
    "legislature", "upper_house", "lower_house", "admin_center",
    "admin_center_type", "currency", "currency_code", "calling_code",
    "cctld", "iso3166code", "time_zone", "time_zone_dst", "date_format",
    "drives_on", "patron_saint", "common_name",
}

# Fields that contain percentage-based lists parsed into subtrees
LIST_FIELDS = {
    "religion", "ethnic_groups",
}

# Fields that contain comma-separated name (pct) entries
LANGUAGE_FIELDS = {
    "languages", "languages2", "official_languages", "regional_languages",
}

# Fields excluded entirely from the tree
EXCLUDED_FIELDS = {
    "religion_ref",
}

# Numeric fields kept as a single token preserving the full value
NUMERIC_FIELDS = {
    "area_km2", "area_sq_mi", "area_rank", "percent_water",
    "population_estimate", "population_census", "population_density_km2",
    "population_density_sq_mi", "population_rank", "population_estimate_rank",
    "population_census_rank", "population_density_rank",
    "gdp_ppp", "gdp_ppp_rank", "gdp_ppp_year", "gdp_ppp_per_capita",
    "gdp_ppp_per_capita_rank", "gdp_nominal", "gdp_nominal_rank",
    "gdp_nominal_year", "gdp_nominal_per_capita", "gdp_nominal_per_capita_rank",
    "gini", "gini_year", "gini_rank",
    "hdi", "hdi_year", "hdi_rank",
    "utc_offset", "utc_offset_dst",
}

def _build_tree(element: ET.Element) -> TreeNode:
    if element.tag in EXCLUDED_FIELDS:
        return None

    node = TreeNode(label=element.tag, is_content=False)

    for attr_name in sorted(element.attrib.keys()):
        attr_node = TreeNode(label=attr_name, is_content=False)
        attr_value_node = TreeNode(label=element.attrib[attr_name], is_content=True)
        attr_node.add_child(attr_value_node)
        node.add_child(attr_node)

    if element.text and element.text.strip():
        text = element.text.strip()
        if element.tag in NUMERIC_FIELDS:
            node.add_child(TreeNode(label=_clean_numeric(text), is_content=True))
        elif element.tag in ATOMIC_FIELDS or element.tag in LIST_FIELDS or element.tag in LANGUAGE_FIELDS:
            node.add_child(TreeNode(label=text, is_content=True))
        else:
            for token in _tokenize(text):
                node.add_child(TreeNode(label=token, is_content=True))

    for child_element in element:
        child_node = _build_tree(child_element)
        if child_node is not None:
            node.add_child(child_node)

    return node


def _clean_numeric(text: str) -> str:
    cleaned = re.sub(r"\[.*?\]", "", text.strip())
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text.strip()


def _tokenize(text: str) -> list[str]:
    tokens = []
    current = []

    for char in text:
        if char in (" ", "\t", "\n", ",", ";", "!", "?", ":", "/"):
            if current:
                tokens.append("".join(current))
                current = []
        elif char == "." and current and not any(c.isdigit() for c in current):
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


def _find_child(root: TreeNode, label: str) -> TreeNode | None:
    for child in root.children:
        if child.label == label:
            return child
    return None


def _detach(root: TreeNode, node: TreeNode) -> None:
    if node in root.children:
        root.children.remove(node)
        node.parent = None


def _collect_numbered(root: TreeNode, prefix: str) -> dict[int, TreeNode]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    result: dict[int, TreeNode] = {}
    for child in root.children:
        match = pattern.match(child.label)
        if match:
            result[int(match.group(1))] = child
    return result


def _make_node(label: str, children_from: TreeNode | None = None) -> TreeNode:
    node = TreeNode(label=label, is_content=False)
    if children_from is not None:
        for child in children_from.children:
            child.parent = node
        node.children = list(children_from.children)
    return node


def _attach(root: TreeNode, node: TreeNode) -> None:
    node.parent = root
    root.children.append(node)


def _group_establishment(root: TreeNode) -> None:
    events = _collect_numbered(root, "established_event")
    dates = _collect_numbered(root, "established_date")

    all_numbers = sorted(set(events.keys()) | set(dates.keys()))
    if not all_numbers:
        return

    for node in list(events.values()) + list(dates.values()):
        _detach(root, node)

    establishment = TreeNode(label="establishment", is_content=False)

    for n in all_numbers:
        event_node = TreeNode(label=f"event{n}", is_content=False)
        if n in events:
            for child in events[n].children:
                event_node.add_child(child)
        if n in dates:
            for child in dates[n].children:
                event_node.add_child(child)
        establishment.add_child(event_node)

    _attach(root, establishment)


def _group_leaders(root: TreeNode) -> None:
    titles = _collect_numbered(root, "leader_title")
    names = _collect_numbered(root, "leader_name")

    all_numbers = sorted(set(titles.keys()) | set(names.keys()))
    if not all_numbers:
        return

    for node in list(titles.values()) + list(names.values()):
        _detach(root, node)

    leaders = TreeNode(label="leaders", is_content=False)

    for n in all_numbers:
        leader_node = TreeNode(label=f"leader{n}", is_content=False)
        if n in titles:
            title_node = _make_node("title")
            title_val = " ".join(c.label for c in titles[n].children if c.is_content)
            title_node.add_child(TreeNode(label=title_val, is_content=True))
            leader_node.add_child(title_node)
        if n in names:
            name_node = _make_node("name")
            name_val = " ".join(c.label for c in names[n].children if c.is_content)
            name_node.add_child(TreeNode(label=name_val, is_content=True))
            leader_node.add_child(name_node)
        leaders.add_child(leader_node)

    _attach(root, leaders)


def _group_establishment(root: TreeNode) -> None:
    events = _collect_numbered(root, "established_event")
    dates = _collect_numbered(root, "established_date")

    all_numbers = sorted(set(events.keys()) | set(dates.keys()))
    if not all_numbers:
        return

    for node in list(events.values()) + list(dates.values()):
        _detach(root, node)

    establishment = TreeNode(label="establishment", is_content=False)

    for n in all_numbers:
        event_node = TreeNode(label=f"event{n}", is_content=False)

        if n in events:
            event_val = " ".join(c.label for c in events[n].children if c.is_content)
            event_node.add_child(TreeNode(label=event_val, is_content=True))

        if n in dates:
            date_val = " ".join(c.label for c in dates[n].children if c.is_content)
            date_node = TreeNode(label="date", is_content=False)
            date_node.add_child(TreeNode(label=date_val, is_content=True))
            event_node.add_child(date_node)

        establishment.add_child(event_node)

    _attach(root, establishment)


def _group_gdp(root: TreeNode, prefix: str, group_label: str) -> None:
    fields = {
        "total": f"{prefix}",
        "rank": f"{prefix}_rank",
        "year": f"{prefix}_year",
        "per_capita": f"{prefix}_per_capita",
        "per_capita_rank": f"{prefix}_per_capita_rank",
    }

    found = {key: _find_child(root, tag) for key, tag in fields.items()}
    if not any(found.values()):
        return

    for node in found.values():
        if node is not None:
            _detach(root, node)

    gdp_node = TreeNode(label=group_label, is_content=False)

    for key, node in found.items():
        if node is not None:
            gdp_node.add_child(_make_node(key, node))

    _attach(root, gdp_node)


def _group_population(root: TreeNode) -> None:
    fields = {
        "estimate": "population_estimate",
        "estimate_year": "population_estimate_year",
        "estimate_rank": "population_estimate_rank",
        "census": "population_census",
        "census_year": "population_census_year",
        "census_rank": "population_census_rank",
        "density_km2": "population_density_km2",
        "density_sq_mi": "population_density_sq_mi",
        "density_rank": "population_density_rank",
        "rank": "population_rank",
    }

    found = {key: _find_child(root, tag) for key, tag in fields.items()}
    if not any(found.values()):
        return

    for node in found.values():
        if node is not None:
            _detach(root, node)

    population = TreeNode(label="population", is_content=False)

    for key, node in found.items():
        if node is not None:
            population.add_child(_make_node(key, node))

    _attach(root, population)


def _group_index(root: TreeNode, prefix: str, group_label: str) -> None:
    fields = {
        "value": prefix,
        "year": f"{prefix}_year",
        "rank": f"{prefix}_rank",
        "change": f"{prefix}_change",
    }

    found = {key: _find_child(root, tag) for key, tag in fields.items()}
    if not any(found.values()):
        return

    for node in found.values():
        if node is not None:
            _detach(root, node)

    index_node = TreeNode(label=group_label, is_content=False)

    for key, node in found.items():
        if node is not None:
            index_node.add_child(_make_node(key, node))

    _attach(root, index_node)



def _parse_percentage_list(text: str) -> list[tuple[str, str]]:
    entries = []
    pattern = re.compile(r'(\d+\.?\d*)\s*%\s*([^0-9%]+?)(?=\s*\d+\.?\d*\s*%|$)')
    for match in pattern.finditer(text):
        pct = match.group(1).strip() + "%"
        name = re.sub(r'\s+', ' ', match.group(2)).strip().strip("—–-").strip()
        if name:
            entries.append((name, pct))
    return entries


def _group_percentage_field(root: TreeNode, field_label: str) -> None:
    node = _find_child(root, field_label)
    if node is None:
        return

    raw = " ".join(c.label for c in node.children if c.is_content)
    if not raw:
        return

    entries = _parse_percentage_list(raw)
    if not entries:
        return

    _detach(root, node)

    group = TreeNode(label=field_label, is_content=False)
    for name, pct in entries:
        entry_node = TreeNode(label=name, is_content=False)
        entry_node.add_child(TreeNode(label=pct, is_content=True))
        group.add_child(entry_node)

    _attach(root, group)


def _group_language_field(root: TreeNode, field_label: str) -> None:
    node = _find_child(root, field_label)
    if node is None:
        return

    raw = " ".join(c.label for c in node.children if c.is_content)
    if not raw:
        return

    _detach(root, node)

    group = TreeNode(label=field_label, is_content=False)

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        entry_node = TreeNode(label=part, is_content=True)
        group.add_child(entry_node)

    _attach(root, group)


def _post_process(root: TreeNode) -> TreeNode:
    _group_population(root)
    _group_gdp(root, "gdp_ppp", "gdp_ppp")
    _group_gdp(root, "gdp_nominal", "gdp_nominal")
    _group_index(root, "gini", "gini")
    _group_index(root, "hdi", "hdi")
    _group_establishment(root)
    _group_leaders(root)
    for field in list(LIST_FIELDS):
        _group_percentage_field(root, field)
    for field in list(LANGUAGE_FIELDS):
        _group_language_field(root, field)
    return root


def load_tree(country_name: str) -> TreeNode:
    filename = country_name.lower().replace(" ", "_") + ".xml"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No XML file found for country: {country_name} at {filepath}")

    xml_tree = ET.parse(filepath)
    root_element = xml_tree.getroot()

    tree = _build_tree(root_element)
    return _post_process(tree)


def load_tree_from_file(filepath: str) -> TreeNode:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    xml_tree = ET.parse(filepath)
    root_element = xml_tree.getroot()

    tree = _build_tree(root_element)
    return _post_process(tree)