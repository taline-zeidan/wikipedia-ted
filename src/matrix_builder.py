"""
Pairwise Similarity Matrix Builder (Project 2)
====================================================================
Builds, validates, caches, and loads the NxN similarity matrix for all
UN member states scraped by Project 1.

Every time build_matrix() is called with overwrite=True it:
  1. Re-runs collect_all() from Project 1 to scrape fresh infoboxes.
  2. Loads and preprocesses each country's XML into a comparison tree.
  3. Computes all N*(N-1)/2 TED-based similarity pairs.
  4. Validates the result and writes it to similarity_matrix.json.

The country list (WORKING_SET) is imported directly from collector.py so
there is a single authoritative source for UN member state names.

Validation guarantees (five checks):
  1. All countries present in the matrix.
  2. Diagonal entries are exactly 1.0 (self-similarity).
  3. Matrix is symmetric: sim(A, B) == sim(B, A).
  4. All scores are in [0, 1].
  5. Exact pair count: N*(N-1)/2 computed pairs.
"""

import json
import os

from src.collector import UN_MEMBER_STATES, collect_all
from src.preprocessor import load_tree
from src.ted import compute_ted

# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------

MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "un_similarity_matrix_193.json"
)

# Single authoritative working set, sourced from Project 1 collector.
# No duplication: if the country list ever changes, only collector.py changes.
WORKING_SET: list[str] = UN_MEMBER_STATES


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_matrix(
    countries: list[str],
    overwrite: bool = False,
    rescrape: bool = True,
) -> dict:
    """
    Build and cache the pairwise similarity matrix.

    Args:
        countries: Canonical country names to include.
        overwrite: If True, recompute even when a cached matrix exists.
                   If False and a cached matrix exists, load and return it.
        rescrape:  If True, re-fetch infoboxes from Wikipedia before building.
                   If False, use whatever XML files are already on disk.
                   Only consulted when a build actually happens (i.e. when
                   overwrite=True or no cached matrix exists).

    Returns:
        Nested dict: matrix[country_a][country_b] = similarity score.

    Two common workflows:
        build_matrix(WS, overwrite=True, rescrape=True)   # full refresh
        build_matrix(WS, overwrite=True, rescrape=False)  # recompute only,
                                                          # reuse cached XML
    """
    if not overwrite and os.path.exists(MATRIX_PATH):
        print(f"[matrix_builder] Loading cached matrix from {MATRIX_PATH}")
        with open(MATRIX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["matrix"]

    # Step 1: optionally re-scrape infoboxes via Project 1
    if rescrape:
        print("[matrix_builder] Step 1: Scraping Wikipedia infoboxes via Project 1...")
        collect_all(overwrite=overwrite)
    else:
        print("[matrix_builder] Step 1: Skipped (rescrape=False), using cached XML files.")

    # Step 2: load preprocessed trees
    print(f"\n[matrix_builder] Step 2: Loading trees for {len(countries)} countries...")
    trees = {}
    skipped = []
    for country in countries:
        try:
            trees[country] = load_tree(country)
            print(f"  [OK]   {country}")
        except Exception as exc:
            skipped.append(country)
            print(f"  [SKIP] {country}: {exc}")

    loaded = list(trees.keys())
    n = len(loaded)
    total_pairs = n * (n - 1) // 2

    # Step 3: compute all pairs
    print(f"\n[matrix_builder] Step 3: Computing {total_pairs} pairs for {n} countries...")

    matrix = {c: {c: 1.0} for c in loaded}  # diagonal = 1.0

    computed = 0
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = loaded[i], loaded[j]
            result = compute_ted(trees[c1], trees[c2], c1, c2)
            score = round(result.similarity, 4)
            matrix[c1][c2] = score
            matrix[c2][c1] = score
            computed += 1
            if computed % 50 == 0 or computed == total_pairs:
                print(f"  [{computed}/{total_pairs}] {c1} vs {c2}: {score:.4f}")

    validate_matrix(matrix, loaded)

    # Step 4: save
    os.makedirs(os.path.dirname(MATRIX_PATH), exist_ok=True)
    with open(MATRIX_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"countries": loaded, "skipped": skipped, "matrix": matrix},
            f,
            indent=2,
        )

    print(f"\n[matrix_builder] Matrix saved to {MATRIX_PATH}")
    if skipped:
        print(f"[matrix_builder] Skipped {len(skipped)} countries: {skipped}")

    return matrix


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_matrix(matrix: dict, countries: list[str]) -> None:
    """
    Run five correctness checks on the computed matrix.
    Raises AssertionError if any check fails (matrix is not saved).

    Checks:
        1. All expected countries are present.
        2. Diagonal = 1.0 for every country.
        3. Symmetry: sim(A,B) == sim(B,A) within floating-point tolerance.
        4. All scores in [0, 1].
        5. Exact pair count N*(N-1)/2.
    """
    n = len(countries)
    expected_pairs = n * (n - 1) // 2

    missing = [c for c in countries if c not in matrix]
    assert not missing, f"Missing countries in matrix: {missing}"

    bad_diagonal = [c for c in countries if matrix[c][c] != 1.0]
    assert not bad_diagonal, f"Diagonal not 1.0: {bad_diagonal}"

    asymmetric = [
        (c1, c2)
        for c1 in countries
        for c2 in countries
        if abs(matrix[c1][c2] - matrix[c2][c1]) > 1e-9
    ]
    assert not asymmetric, f"Asymmetric pairs (first 5): {asymmetric[:5]}"

    out_of_bounds = [
        (c1, c2)
        for c1 in countries
        for c2 in countries
        if not (0.0 <= matrix[c1][c2] <= 1.0)
    ]
    assert not out_of_bounds, f"Out-of-bounds scores (first 5): {out_of_bounds[:5]}"

    computed_pairs = sum(
        1
        for i, c1 in enumerate(countries)
        for c2 in countries[i + 1:]
        if c1 in matrix and c2 in matrix[c1]
    )
    assert computed_pairs == expected_pairs, (
        f"Pair count mismatch: expected {expected_pairs}, found {computed_pairs}"
    )

    print(
        f"[matrix_builder] Validation passed: "
        f"{n} countries, {computed_pairs} pairs, all 5 checks OK."
    )


# ---------------------------------------------------------------------------
# Load (read-only, no recompute)
# ---------------------------------------------------------------------------

def load_matrix() -> tuple[dict, list[str], list[str]]:
    """
    Load a previously built matrix from disk.

    Returns:
        (matrix dict, list of country names in matrix order, list of skipped countries)

    Raises:
        FileNotFoundError if no cached matrix exists.
    """
    if not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError(
            f"No matrix found at {MATRIX_PATH}. Run build_matrix() first."
        )
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["matrix"], data["countries"], data.get("skipped", [])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_matrix(WORKING_SET, overwrite=True)