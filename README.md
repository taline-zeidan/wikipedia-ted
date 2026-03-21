# Wikipedia Infobox Comparison using Tree Edit Distance

A Python pipeline and web application for collecting, comparing, and differencing Wikipedia country infobox documents using Tree Edit Distance (TED). Given any two UN-recognized countries, the system computes their structural and semantic similarity, extracts a minimal edit script describing their differences, applies it to transform one country's infobox into the other's, and visualizes the results through an interactive Streamlit interface.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Modules](#pipeline-modules)
- [Output Files](#output-files)
- [Design Decisions](#design-decisions)

---

## Overview

This project was developed as part of the Intelligent Data Processing and Applications course (COE 543/743) at the Lebanese American University, Spring 2026.

The pipeline consists of five core stages plus a web interface:

1. **Data Collection** — Scrape Wikipedia infoboxes for all 193 UN member states using the MediaWiki API and `mwparserfromhell`
2. **Pre-processing** — Parse XML documents into rooted ordered labeled trees with semantic grouping of related fields
3. **TED + Differencing** — Compute label-preserving Tree Edit Distance using Zhang-Shasha (adapted by Nierman & Jagadish) and extract a path-based edit script
4. **Patching** — Apply the edit script to transform one tree into another using stable path-based node addressing
5. **Post-processing** — Reconstruct the patched tree as XML or Wikipedia wikitext
6. **Web UI** — Interactive Streamlit application for country comparison, tree visualization, and result exploration

---

## Project Structure

```
wikipedia-ted/
│
├── data/
│   ├── raw/                  # Scraped XML infoboxes (one per country)
│   ├── diffs/                # Edit scripts (one per compared pair)
│   └── infoboxes/            # Wikitext output files
│
├── src/
│   ├── collector.py          # Module 1: Wikipedia scraper
│   ├── preprocessor.py       # Module 2: XML → TreeNode with semantic grouping
│   ├── ted.py                # Module 3: TED algorithm + path-based edit script
│   ├── patcher.py            # Module 4: Path-based tree patching
│   ├── postprocessor.py      # Module 5: TreeNode → XML / wikitext
│   └── filter.py             # Field filter for selective comparison
│
├── models/
│   └── tree.py               # TreeNode class and TreeUtils helpers
│
├── app.py                    # Streamlit web interface
├── main.py                   # CLI entry point
├── requirements.txt
└── report/
    └── report.docx           # Technical report
```

---

## Requirements

- Python 3.11+
- Internet connection (for data collection)

---

## Installation

```bash
git clone https://github.com/your-username/wikipedia-ted.git
cd wikipedia-ted
pip install -r requirements.txt
```

---

## Usage

### Web Interface (recommended)

```bash
streamlit run app.py
```

The UI guides you through three steps:

1. Select two countries from a dropdown of all 193 UN member states
2. Click **Load Fields** — the app scrapes both countries if not already cached, then presents fields grouped by category (Identity, Geography, Population, Government, Economy, Society)
3. Select which fields to include and click **Compare**

Results are displayed across four tabs: interactive tree visualization, edit script breakdown, patched XML output, and raw XML for both countries.

### CLI

```bash
python main.py "Lebanon" "Switzerland"
```

| Flag               | Description                               | Default |
| ------------------ | ----------------------------------------- | ------- |
| `--format xml`     | Output patched tree as XML                | `xml`   |
| `--format infobox` | Output patched tree as Wikipedia wikitext | —       |
| `--overwrite`      | Re-scrape even if XML files already exist | `False` |

To collect infoboxes for all 193 UN member states at once:

```python
from src.collector import collect_all
collect_all()
```

---

## Pipeline Modules

### `models/tree.py`

Defines the `TreeNode` class and `TreeUtils` helpers used by all modules. Each node carries a `label` and an `is_content` flag distinguishing structural nodes (XML tag names) from content nodes (text values). `TreeUtils` provides postorder traversal, leftmost leaf computation, keyroot identification, and stable path-based navigation via `get_path()` and `get_node_by_path()`.

### `src/collector.py`

Fetches raw wikitext from the MediaWiki API and parses it using `mwparserfromhell` to cleanly extract infobox key-value pairs. Strips nested templates, wikilinks, and HTML. Excludes decorative fields (image filenames, captions, alt text). Writes one XML file per country to `data/raw/`.

### `src/preprocessor.py`

Parses XML into a `TreeNode` hierarchy. Fields are classified as atomic (stored as single content nodes), numeric (comma-stripped), or tokenized (split on whitespace and punctuation boundaries). Related flat fields are grouped into semantic subtrees during post-processing: `population`, `gdp_ppp`, `gdp_nominal`, `gini`, `hdi`, `establishment` (with dated events), and `leaders` (with title/name pairs). List fields (`religion`, `ethnic_groups`) are parsed into percentage-entry subtrees. Language fields are split into comma-separated leaf nodes.

### `src/ted.py`

Implements the Zhang-Shasha algorithm adapted by Nierman and Jagadish with label-preserving costs: structural node renames cost 10000 (effectively infinity), content node renames cost 1, inserts and deletes cost 1. Stores all forest distance tables during the forward pass for correct backtracking. After extraction, `_clean_operations()` applies seven filters to remove phantom matches from the DP, and `_sweep_missing_deletes()` catches any nodes present in T1 but absent in T2 that backtracking missed. Operations store stable tree paths (`["country", "population", "estimate"]`) rather than postorder indices.

### `src/patcher.py`

Applies an `EditScript` to a source tree using path-based node navigation. Renames are applied first, then deletes deepest-first (longest path first), then inserts. After deletion, `_prune_empty_structural_nodes()` removes any structural nodes left childless. Path stability means no index recomputation is needed between operations.

### `src/postprocessor.py`

Converts a `TreeNode` tree back to XML using a purely recursive element builder with reserved-tag handling. Also supports Wikipedia wikitext output via `_flatten_to_fields()` which recursively flattens nested paths to `key → value` pairs.

### `src/filter.py`

Extracts available top-level field names from a tree and returns a filtered copy containing only selected fields. Used by the UI to enable selective comparison by category.

---

## Output Files

| Path                                              | Description                                  |
| ------------------------------------------------- | -------------------------------------------- |
| `data/raw/{country}.xml`                          | Scraped and sanitized infobox XML            |
| `data/diffs/{source}_{target}.xml`                | Path-based edit script between two countries |
| `data/infoboxes/{source}_patched_to_{target}.txt` | Wikitext output (if `--format infobox`)      |

---

## Design Decisions

**Label-preserving TED:** Structural node renames are assigned a cost of 10000, effectively preventing the algorithm from matching nodes of different types across fields. This directly implements the project requirement to distinguish between document structure and content, and is consistent with the Nierman & Jagadish formulation.

**Semantic tree grouping:** Related flat infobox fields (e.g. `population_estimate`, `population_density_km2`, `population_rank`) are grouped into a single `population` subtree during preprocessing. This makes TED structurally meaningful — two countries with similar population data produce similar subtrees and lower edit distance, rather than having unrelated flat fields matched by proximity.

**Atomic fields:** Fields containing structured list data (religion breakdowns, language distributions, government type) are stored as single atomic content nodes. Tokenizing these would produce semantically meaningless fragments and inflate TED scores artificially.

**Path-based edit script:** Operations store the full path from root to the target node rather than a postorder index. This eliminates index drift after insertions and deletions, making the patcher stable and correct regardless of tree depth or structural differences between countries.

**mwparserfromhell for scraping:** Wikipedia infobox values contain nested MediaWiki templates, wikilinks, and HTML that simple regex stripping cannot handle reliably. `mwparserfromhell` uses a proper wikitext parser and its `strip_code()` method cleanly removes all markup, leaving only plain text values.

**Flat file storage:** Country data is stored as individual XML files rather than in a database. This aligns with the XML-centric nature of the project and avoids unnecessary infrastructure complexity for a research prototype.
