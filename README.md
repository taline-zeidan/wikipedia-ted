# Wikipedia Infobox Comparison using Tree Edit Distance

A Python pipeline for collecting, comparing, and differencing Wikipedia country infobox documents using Tree Edit Distance (TED). Given any two UN-recognized countries, the system computes their structural and semantic similarity, extracts a minimal edit script describing their differences, and applies it to transform one country's infobox into the other's.

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

The pipeline consists of five stages:

1. **Data Collection** — Scrape Wikipedia infoboxes for all 193 UN member states
2. **Pre-processing** — Parse XML documents into rooted ordered labeled trees
3. **TED + Differencing** — Compute Tree Edit Distance and extract an edit script
4. **Patching** — Apply the edit script to transform one tree into another
5. **Post-processing** — Reconstruct the patched tree as XML or Wikipedia wikitext

---

## Project Structure

```
wikipedia-ted/
│
├── data/
│   ├── raw/                  # Scraped XML infoboxes (one per country)
│   └── diffs/                # Edit scripts (one per compared pair)
│
├── src/
│   ├── collector.py          # Wikipedia scraper
│   ├── preprocessor.py       # XML → TreeNode
│   ├── ted.py                # TED algorithm + edit script
│   ├── patcher.py            # Tree patching
│   └── postprocessor.py      # TreeNode → XML / wikitext
│
├── models/
│   └── tree.py               # Shared TreeNode class and TreeUtils
│
├── main.py                   # CLI entry point
└── requirements.txt
```

---

## Requirements

- Python 3.11+
- Please make sure you are connected to the internet!

---

## Installation

```bash
git clone https://github.com/your-username/wikipedia-ted.git
cd wikipedia-ted
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline from the command line by providing two country names:

```bash
python main.py "Lebanon" "Switzerland"
```

**Options:**

| Flag               | Description                                         | Default |
| ------------------ | --------------------------------------------------- | ------- |
| `--format xml`     | Output patched tree as XML                          | `xml`   |
| `--format infobox` | Output patched tree as Wikipedia wikitext           | —       |
| `--overwrite`      | Re-scrape countries even if XML files already exist | `False` |

**Examples:**

```bash
# Compare Lebanon and Switzerland, output as XML
python main.py "Lebanon" "Switzerland"

# Output as Wikipedia infobox wikitext
python main.py "Lebanon" "Switzerland" --format infobox

# Force re-scrape
python main.py "Lebanon" "Switzerland" --overwrite
```

**Sample output:**

```
============================================================
  Wikipedia TED Pipeline
  Lebanon → Switzerland
============================================================

[1/5] Collecting data...
      [OK] Lebanon -> data/raw/lebanon.xml
      [OK] Switzerland -> data/raw/switzerland.xml

[2/5] Preprocessing trees...
      Lebanon: 84 nodes
      Switzerland: 91 nodes

[3/5] Computing TED and extracting edit script...
      TED score : 37
      Operations: 52 total
        INSERT  : 18
        DELETE  : 11
        RENAME  : 23
      Diff saved to: data/diffs/lebanon_switzerland.xml

[4/5] Patching Lebanon tree → Switzerland...
      Patched tree: 91 nodes

[5/5] Post-processing output (xml)...
      Output saved to: data/raw/lebanon_patched_to_switzerland.xml

============================================================
  Done. TED score: 37
============================================================
```

To collect infoboxes for all 193 UN member states at once:

```python
from src.collector import collect_all
collect_all()
```

---

## Pipeline Modules

### `models/tree.py`

Defines the `TreeNode` class used by all modules. Each node carries a `label` (tag name or token value) and an `is_content` flag distinguishing structural nodes (XML element/attribute names) from content nodes (text values).

### `src/collector.py`

Fetches Wikipedia infoboxes using the `wptools` library, sanitizes keys into valid XML tag names, strips MediaWiki template syntax from values, and writes one XML file per country to `data/raw/`.

### `src/preprocessor.py`

Parses an XML file into a `TreeNode` tree. Attributes are attached as sorted child nodes before sub-elements. Text values are tokenized into individual content leaf nodes.

### `src/ted.py`

Implements the Zhang-Shasha Tree Edit Distance algorithm adapted by Nierman and Jagadish for XML comparison. Returns a TED score and an `EditScript` containing the sequence of `INSERT`, `DELETE`, and `RENAME` operations extracted by backtracking through the DP table.

### `src/patcher.py`

Applies an `EditScript` to a source tree to produce the target tree. Operations are applied in safe order: renames first, then deletes from highest to lowest postorder index, then inserts from lowest to highest.

### `src/postprocessor.py`

Converts a `TreeNode` tree back to XML or Wikipedia infobox wikitext (`| key = value` format wrapped in `{{Infobox country}}`).

---

## Output Files

| Path                                              | Description                             |
| ------------------------------------------------- | --------------------------------------- |
| `data/raw/{country}.xml`                          | Scraped and sanitized infobox XML       |
| `data/diffs/{source}_{target}.xml`                | Edit script between two countries       |
| `data/raw/{source}_patched_to_{target}.xml`       | Patched tree output                     |
| `data/infoboxes/{source}_patched_to_{target}.txt` | Wikitext output (if `--format infobox`) |

---

## Design Decisions

**Tokenization:** Text values are split into individual token nodes rather than represented as a single atomic text node. This allows the TED algorithm to partially match textual content (e.g., "Lebanese Republic" and "Lebanese Confederation" share one token), improving semantic granularity.

**Attribute ordering:** XML attributes are sorted alphabetically and attached as child nodes before sub-elements, following the project specification for reducing comparison complexity.

**Uniform edit costs:** All operations (insert, delete, rename) use cost 1. This simplifies the cost model and ensures reproducibility across country pairs.

**Flat file storage:** Country data is stored as individual XML files rather than in a database. This aligns with the XML-centric nature of the project and avoids unnecessary infrastructure complexity.

**Operation ordering in patching:** Renames are applied first (no index shift), followed by deletes from highest to lowest postorder index, then inserts from lowest to highest. This ordering ensures postorder indices remain valid throughout the patching process.
