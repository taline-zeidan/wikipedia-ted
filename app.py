import sys
import os
import copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from src.collector import collect_country, UN_MEMBER_STATES
from src.preprocessor import load_tree
from src.ted import compute_ted, save_edit_script
from src.patcher import patch
from src.postprocessor import tree_to_xml_string
from src.filter import get_available_fields, filter_tree
from models.tree import TreeNode, TreeUtils
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(
    page_title="Country Infobox Comparator",
    page_icon="🌍",
    layout="wide",
)

import re as _re

FIELD_CATEGORIES = {
    "Identity": [
        "common_name", "conventional_long_name", "native_name", "official_name",
        "name", "demonym", "cctld", "calling_code", "iso3166code",
        "englishmotto", "national_motto", "englishname",
    ],
    "Geography": [
        "area_km2", "area_sq_mi", "area_rank", "area_magnitude", "area_label",
        "percent_water", "coordinates", "latd", "latm", "lats", "latns",
        "longd", "longm", "longs", "longew", "admin_center", "admin_center_type",
        "highest_point", "lowest_point", "largest_city", "capital",
    ],
    "Population": [
        "population_estimate", "population_census", "population_density_km2",
        "population_density_sq_mi", "population_rank", "population_estimate_year",
        "population_estimate_rank", "population_census_year", "population_census_rank",
        "population_density_rank",
    ],
    "Government": [
        "government_type", "legislature", "upper_house", "lower_house",
        "sovereignty_type", "sovereignty_note",
    ],
    "Economy": [
        "gdp_ppp", "gdp_ppp_rank", "gdp_ppp_year", "gdp_ppp_per_capita",
        "gdp_ppp_per_capita_rank", "gdp_nominal", "gdp_nominal_rank",
        "gdp_nominal_year", "gdp_nominal_per_capita", "gdp_nominal_per_capita_rank",
        "gini", "gini_year", "gini_rank", "gini_change",
        "hdi", "hdi_year", "hdi_rank", "hdi_change",
        "currency", "currency_code",
    ],
    "Society": [
        "religion", "religion_year", "religion_ref",
        "languages_type", "languages", "languages2_type", "languages2",
        "official_languages", "regional_languages",
        "ethnic_groups", "ethnic_groups_year",
        "drives_on", "date_format", "time_zone", "utc_offset",
        "time_zone_dst", "utc_offset_dst", "patron_saint",
    ],
}

FIELD_PATTERNS: dict[str, list] = {
    "Government": [
        _re.compile(r"^leader_title\d+$"),
        _re.compile(r"^leader_name\d+$"),
        _re.compile(r"^established_date\d+$"),
        _re.compile(r"^established_event\d+$"),
        _re.compile(r"^year_leader\d+$"),
    ],
    "Population": [
        _re.compile(r"^population_census\d+$"),
        _re.compile(r"^population_density\d+$"),
    ],
}

def _categorize_fields(available: list[str]) -> dict[str, list[str]]:
    categorized: dict[str, list[str]] = {}
    assigned: set[str] = set()

    for category, fields in FIELD_CATEGORIES.items():
        matched = [f for f in fields if f in available]
        categorized[category] = matched
        assigned.update(matched)

    for field in available:
        if field in assigned:
            continue
        for category, patterns in FIELD_PATTERNS.items():
            if any(p.match(field) for p in patterns):
                categorized[category].append(field)
                assigned.add(field)
                break

    other = sorted([f for f in available if f not in assigned])
    if other:
        categorized["Other"] = other

    return {k: v for k, v in categorized.items() if v}

def _ensure_collected(country: str) -> bool:
    filepath = os.path.join("data", "raw", country.lower().replace(" ", "_") + ".xml")
    if not os.path.exists(filepath):
        try:
            collect_country(country)
        except Exception as e:
            st.error(f"Could not collect data for {country}: {e}")
            return False
    return True

def _normalized_ted(raw_score: int, t1_size: int, t2_size: int) -> float:
    denominator = t1_size + t2_size
    if denominator == 0:
        return 1.0
    return round(1 - (raw_score / denominator), 4)

def _is_meaningful_label(label: str) -> bool:
    stripped = label.strip()
    if not stripped:
        return False
    if all(c in r"{}|!@#$%^&*()<>/\=+" for c in stripped):
        return False
    if len(stripped) <= 1 and not stripped.isalpha():
        return False
    return True

def _build_agraph(root: TreeNode, prefix: str = "") -> tuple[list[Node], list[Edge]]:
    nodes = []
    edges = []
    counter = [0]

    def walk(node: TreeNode, parent_id: str | None):
        counter[0] += 1
        node_id = f"{prefix}_{counter[0]}_{node.label}"
        color = "#4A90E2" if node.is_content else "#27AE60"
        font_color = "#ffffff"
        size = 14 if node.is_content else 18
        shape = "ellipse" if node.is_content else "box"
        nodes.append(Node(
            id=node_id,
            label=node.label,
            size=size,
            color=color,
            font={"color": font_color, "size": 11},
            shape=shape,
        ))
        if parent_id is not None:
            edges.append(Edge(source=parent_id, target=node_id, color="#B4B2A9", width=1))
        for child in node.children:
            walk(child, node_id)

    walk(root, None)
    return nodes, edges

st.markdown(
    "<h1 style='margin-bottom:0'>🌍 Country Infobox Comparator</h1>"
    "<p style='color:#888; margin-top:4px;'>Compare Wikipedia country infoboxes using Tree Edit Distance</p>",
    unsafe_allow_html=True,
)
st.divider()

st.markdown("### Select Countries")
col1, col2 = st.columns(2)
country1 = col1.selectbox("Country 1", UN_MEMBER_STATES, index=UN_MEMBER_STATES.index("Lebanon"))
country2 = col2.selectbox("Country 2", UN_MEMBER_STATES, index=UN_MEMBER_STATES.index("Switzerland"))

if country1 == country2:
    st.warning("Please select two different countries.")
    st.stop()

load_btn = st.button("Load Fields", type="secondary")

if "fields_loaded" not in st.session_state:
    st.session_state.fields_loaded = False
if "available_fields" not in st.session_state:
    st.session_state.available_fields = {}
if "last_countries" not in st.session_state:
    st.session_state.last_countries = (None, None)
if "result" not in st.session_state:
    st.session_state.result = None

countries_changed = st.session_state.last_countries != (country1, country2)

if load_btn or (st.session_state.fields_loaded and countries_changed):
    with st.spinner("Loading country data..."):
        ok1 = _ensure_collected(country1)
        ok2 = _ensure_collected(country2)
    if ok1 and ok2:
        t1 = load_tree(country1)
        t2 = load_tree(country2)
        all_fields = sorted(set(get_available_fields(t1)) | set(get_available_fields(t2)))
        st.session_state.available_fields = _categorize_fields(all_fields)
        st.session_state.fields_loaded = True
        st.session_state.last_countries = (country1, country2)
        st.session_state.result = None

if not st.session_state.fields_loaded:
    st.stop()

st.markdown("### Select Fields to Compare")
st.caption("Fields are grouped by category. Deselect any category or individual field to exclude it.")

selected_fields: list[str] = []
categorized = st.session_state.available_fields

cols = st.columns(3)
col_index = 0

for category, fields in categorized.items():
    with cols[col_index % 3]:
        st.markdown(f"**{category}**")
        select_all = st.checkbox("Select all", value=True, key=f"all_{category}")
        chosen = st.multiselect(
            label=category,
            options=fields,
            default=fields if select_all else [],
            key=f"fields_{category}",
            label_visibility="collapsed",
        )
        selected_fields.extend(chosen)
    col_index += 1

compare_btn = st.button("Compare →", type="primary", disabled=len(selected_fields) == 0)

if len(selected_fields) == 0:
    st.warning("Select at least one field to compare.")

if compare_btn:
    with st.spinner("Running TED pipeline..."):
        t1 = load_tree(country1)
        t2 = load_tree(country2)

        ft1 = filter_tree(t1, selected_fields)
        ft2 = filter_tree(t2, selected_fields)

        nodes1 = TreeUtils.postorder(ft1)
        nodes2 = TreeUtils.postorder(ft2)

        edit_script = compute_ted(ft1, ft2, source=country1, target=country2)
        save_edit_script(edit_script)

        normalized = _normalized_ted(edit_script.ted_score, len(nodes1), len(nodes2))

        patched = patch(copy.deepcopy(ft1), edit_script)
        patched_xml = tree_to_xml_string(patched)

    st.session_state.result = {
        "edit_script": edit_script,
        "normalized": normalized,
        "ft1": ft1,
        "ft2": ft2,
        "patched_xml": patched_xml,
        "n1": len(nodes1),
        "n2": len(nodes2),
    }

if st.session_state.result is None:
    st.stop()

result = st.session_state.result
edit_script = result["edit_script"]
normalized = result["normalized"]
ft1 = result["ft1"]
ft2 = result["ft2"]
patched_xml = result["patched_xml"]

st.markdown("### Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Similarity Score", f"{normalized:.4f}", help="1.0 = identical, 0.0 = completely different")
c2.metric("Raw TED Score", edit_script.ted_score)
c3.metric(f"{country1} nodes", result["n1"])
c4.metric(f"{country2} nodes", result["n2"])

tab1, tab2, tab3, tab4 = st.tabs(["🌳 Trees", "📄 Edit Script", "🔧 Patched Output", "ℹ️ Raw XML"])

with tab1:
    config = Config(
        width=1300,
        height=800,
        directed=True,
        hierarchical=True,
        physics=False,
        fit=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7C1C1",
        collapsible=True,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": False},
        layout={
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "nodeSpacing": 160,
                "levelSeparation": 120,
                "treeSpacing": 200,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        },
    )

    st.markdown(f"#### {country1}")
    nodes1, edges1 = _build_agraph(ft1, prefix="t1")
    agraph(nodes=nodes1, edges=edges1, config=config)

    st.divider()

    st.markdown(f"#### {country2}")
    nodes2, edges2 = _build_agraph(ft2, prefix="t2")
    agraph(nodes=nodes2, edges=edges2, config=config)

    st.caption("🟢 Structural node (tag name)   🔵 Content node (value)")

with tab2:
    renames = [op for op in edit_script.operations if op.operation == "RENAME"]
    inserts = [op for op in edit_script.operations if op.operation == "INSERT"]
    deletes = [op for op in edit_script.operations if op.operation == "DELETE"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Operations", len(edit_script.operations))
    mc2.metric("🟡 Renames", len(renames))
    mc3.metric("🟢 Inserts", len(inserts))
    mc4.metric("🔴 Deletes", len(deletes))

    if renames:
        st.markdown("#### 🟡 Renames")
        meaningful_renames = [
            op for op in renames
            if _is_meaningful_label(op.node_label) and _is_meaningful_label(op.target_label or "")
        ]
        if meaningful_renames:
            for op in meaningful_renames:
                kind = "content" if op.is_content else "structural"
                st.markdown(
                    f'<div style="font-family:monospace; font-size:13px; margin:4px 0;">'
                    f'<span style="background:#fff3cd; padding:2px 6px; border-radius:4px;">{op.node_label}</span>'
                    f' → '
                    f'<span style="background:#d4edda; padding:2px 6px; border-radius:4px;">{op.target_label}</span>'
                    f' <span style="color:#aaa; font-size:11px;">({kind})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No meaningful renames to display.")

    if inserts:
        st.markdown("#### 🟢 Inserts")
        meaningful_inserts = [op for op in inserts if _is_meaningful_label(op.node_label) and not op.is_content]
        if meaningful_inserts:
            for op in meaningful_inserts:
                st.markdown(
                    f'<div style="font-family:monospace; font-size:13px; margin:4px 0;">'
                    f'<span style="background:#d4edda; padding:2px 6px; border-radius:4px;">+ {op.node_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No meaningful inserts to display.")

    if deletes:
        st.markdown("#### 🔴 Deletes")
        meaningful_deletes = [op for op in deletes if _is_meaningful_label(op.node_label) and not op.is_content]
        if meaningful_deletes:
            for op in meaningful_deletes:
                st.markdown(
                    f'<div style="font-family:monospace; font-size:13px; margin:4px 0;">'
                    f'<span style="background:#f8d7da; padding:2px 6px; border-radius:4px;">− {op.node_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No meaningful deletes to display.")

with tab3:
    st.markdown(f"##### {country1} patched → {country2}")
    st.caption("This is the result of applying the edit script to transform the source tree into the target.")
    st.code(patched_xml, language="xml")

    with st.expander("🔍 Debug: Operation details"):
        st.markdown("**All operations applied:**")
        for op in edit_script.operations:
            path_str = "/".join(op.path)
            st.markdown(
                f'`{op.operation}` &nbsp; `{path_str}` &nbsp;→&nbsp; `{op.target_label}`',
                unsafe_allow_html=True,
            )

with tab4:
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"##### {country1}")
        st.code(tree_to_xml_string(ft1), language="xml")
    with rc2:
        st.markdown(f"##### {country2}")
        st.code(tree_to_xml_string(ft2), language="xml")