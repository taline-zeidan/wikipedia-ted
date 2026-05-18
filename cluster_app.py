"""
Project 2 Streamlit UI
Wikipedia Semi-structured Infobox Document Clustering Tool
COE 543/743, Lebanese American University, Spring 2026

Tabs:
  1. Similarity Matrix: heatmap + scannable sample of the 193x193 matrix
  2. Agglomerative: bottom-up hierarchical clustering + dendrogram table
  3. K-Means: partitional clustering with multiple restarts
  4. Comparison: side-by-side algorithm comparison using intra- and
     inter-cluster similarity (Ch. 10, slides 75-81).

Evaluation uses two internal measures (no external ground truth):
  - Intra-cluster similarity (PGMA): higher means more coherent clusters.
  - Inter-cluster similarity (average-link, averaged over cluster pairs):
    lower means more distinct clusters.
"""

import json
import os

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry

from src.matrix_builder import build_matrix, load_matrix
from src.collector import UN_MEMBER_STATES as WORKING_SET
from src.clustering import agglomerative, kmeans
from src.cluster_eval import evaluate, print_evaluation



st.set_page_config(
    page_title="Country Clustering - Project 2",
    page_icon="🌐",
    layout="wide",
)

st.title("Wikipedia Infobox Country Clustering")
st.caption("COE 543/743 · Project 2 · Lebanese American University · Spring 2026")

# Shared helpers

MATRIX_PATH = os.path.join("data", "un_similarity_matrix_193.json")

# Number of countries shown in the scannable matrix sample
DEFAULT_SAMPLE_SIZE = 20

# Single source of truth for cluster colors.
# Used by the choropleth map AND both pie charts so that "Cluster N" is the
# same color everywhere in the UI. Set2 + Set3 give us 20 distinct colors,
# which covers k up to the slider's max (20).
CLUSTER_PALETTE = (
    list(px.colors.qualitative.Set2) + list(px.colors.qualitative.Set3)
)


def _load_matrix_from_disk() -> tuple[dict | None, list[str] | None, list[str]]:
    """Load the cached matrix from disk; return (None, None, []) if not found.

    Returns (matrix, countries, skipped).
    """
    if not os.path.exists(MATRIX_PATH):
        return None, None, []
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["matrix"], data["countries"], data.get("skipped", [])


def _build_heatmap(
    matrix: dict,
    countries: list[str],
    title: str = "Pairwise Similarity Matrix",
) -> go.Figure:
    """Build a Plotly heatmap from the similarity matrix."""
    values = [[matrix[c1][c2] for c2 in countries] for c1 in countries]
    fig = px.imshow(
        values,
        x=countries,
        y=countries,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        height=max(400, min(900, len(countries) * 14)),
        coloraxis_colorbar_title="Similarity",
    )
    # Hide per-cell text for large matrices (unreadable anyway)
    if len(countries) <= 30:
        fig.update_traces(text=[[f"{v:.2f}" for v in row] for row in values],
                          texttemplate="%{text}", textfont_size=7)
    return fig


def _avg_similarity_to_cluster(
    country: str,
    cluster: list[str],
    matrix: dict,
) -> float:
    """
    Average similarity of a single country to the OTHER members of its cluster.
    A legitimate per-country quality indicator for any clustering algorithm —
    does not require designating any cluster member as a centroid/medoid.
    """
    others = [c for c in cluster if c != country]
    if not others:
        return 1.0  # Singleton: trivially "perfectly similar to its cluster"
    return sum(matrix[country][o] for o in others) / len(others)


def _cluster_table(
    clusters: list[list[str]],
    medoids: list[str] | None = None,
    matrix: dict | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of cluster assignments.

    Two modes (chosen by whether `medoids` is supplied):

    K-Means mode  — medoids is not None:
        Columns: Cluster, Country, Role ('medoid' or ''), Sim to Medoid, Cluster Size
        Used by K-Means, which by construction has a designated medoid per cluster
        (Lecture 10 §5.1 — the medoid is the similarity-matrix equivalent of the
        K-Means centroid).

    Agglomerative mode — medoids is None:
        Columns: Cluster, Country, Avg Sim in Cluster, Cluster Size
        Used by Agglomerative Hierarchical Clustering (Lecture 10 §5.2), which has
        NO medoid concept — it merges clusters based on inter-cluster similarity
        and never designates a representative member. The 'Avg Sim in Cluster'
        column shows each country's average similarity to the other members of
        its cluster, which is a meaningful per-country quality indicator that
        does not require inventing a medoid.
    """
    if medoids is not None:
        # K-Means mode
        rows = []
        for i, cluster in enumerate(clusters):
            medoid = medoids[i] if i < len(medoids) else ""
            for country in sorted(cluster):
                if country == medoid:
                    sim = "1.0000"
                elif matrix and medoid:
                    sim = f"{matrix[country][medoid]:.4f}"
                else:
                    sim = ""
                rows.append({
                    "Cluster": i + 1,
                    "Country": country,
                    "Role": "medoid" if country == medoid else "",
                    "Sim to Medoid": sim,
                    "Cluster Size": len(cluster),
                })
        return pd.DataFrame(rows)

    # Agglomerative mode — no medoids
    rows = []
    for i, cluster in enumerate(clusters):
        for country in sorted(cluster):
            if matrix:
                avg_sim = f"{_avg_similarity_to_cluster(country, cluster, matrix):.4f}"
            else:
                avg_sim = ""
            rows.append({
                "Cluster": i + 1,
                "Country": country,
                "Avg Sim in Cluster": avg_sim,
                "Cluster Size": len(cluster),
            })
    return pd.DataFrame(rows)


def _eval_card(eval_result, label: str = "") -> None:
    """Render an intra-/inter-cluster similarity metric card in the Streamlit UI."""
    prefix = f"{label} - " if label else ""
    col1, col2 = st.columns(2)
    col1.metric(
        f"{prefix}Intra-cluster similarity",
        f"{eval_result.intra_cluster_similarity:.4f}",
        help=(
            "PGMA (Pair Group Method Average): sum over clusters of the average "
            "pairwise similarity inside each cluster. HIGHER is better "
            "(more coherent clusters)."
        ),
    )
    col2.metric(
        f"{prefix}Inter-cluster similarity",
        f"{eval_result.inter_cluster_similarity:.4f}",
        help=(
            "Average of average-link similarities over all cluster pairs. "
            "LOWER is better (more distinct clusters)."
        ),
    )
    st.caption(
        f"{eval_result.n_clusters} clusters · {eval_result.n_objects} countries evaluated"
    )


# ISO 3166 alpha-3 mapping for choropleth
_ISO_OVERRIDES = {
    "Turkey": "TUR",
    "Democratic Republic of the Congo": "COD",
    "Ivory Coast": "CIV",
    "Republic of the Congo": "COG",
    "Palestine": "PSE",
    "North Korea": "PRK",
    "South Korea": "KOR",
    "Micronesia": "FSM",
    "Brunei": "BRN",
    "Laos": "LAO",
    "Syria": "SYR",
    "Iran": "IRN",
    "Russia": "RUS",
    "Venezuela": "VEN",
    "Bolivia": "BOL",
    "Tanzania": "TZA",
    "Vietnam": "VNM",
    "Czech Republic": "CZE",
    "Moldova": "MDA",
    "Eswatini": "SWZ",
    "Cabo Verde": "CPV",
    "Gambia": "GMB",
    "Bahamas": "BHS",
    "Comoros": "COM",
    "São Tomé and Príncipe": "STP",
}


def _get_iso_alpha3(country_name: str) -> str:
    """Resolve a country name to its ISO 3166 alpha-3 code."""
    if country_name in _ISO_OVERRIDES:
        return _ISO_OVERRIDES[country_name]
    match = pycountry.countries.get(name=country_name)
    if match:
        return match.alpha_3
    try:
        results = pycountry.countries.search_fuzzy(country_name)
        return results[0].alpha_3
    except LookupError:
        return ""


def _build_cluster_map(
    clusters: list[list[str]],
    title: str = "Cluster Map",
) -> go.Figure:
    """Build a Plotly choropleth map color-coded by cluster assignment.

    Uses CLUSTER_PALETTE indexed by cluster number so the same cluster
    gets the same color in the map AND in the pie chart on the same tab.
    """
    rows = []
    for i, cluster in enumerate(clusters):
        for country in cluster:
            iso = _get_iso_alpha3(country)
            if iso:
                rows.append({
                    "Country": country,
                    "ISO": iso,
                    "Cluster": f"Cluster {i + 1}",
                    "Cluster #": i + 1,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()

    # Explicit category-to-color map keyed by "Cluster N", matching the
    # labels used in the pie chart. This guarantees a 1:1 color alignment
    # across views, even if Plotly reorders categories internally.
    cluster_color_map = {
        f"Cluster {i + 1}": CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        for i in range(len(clusters))
    }

    fig = px.choropleth(
        df,
        locations="ISO",
        color="Cluster",
        hover_name="Country",
        color_discrete_map=cluster_color_map,
        category_orders={"Cluster": list(cluster_color_map.keys())},
        title=title,
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="lightgray",
            projection_type="natural earth",
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Cluster",
    )
    return fig


def _merges_to_linkage(merges: list, all_countries: list[str]) -> np.ndarray:
    """
    Convert a list of MergeStep objects to a SciPy linkage matrix.

    SciPy linkage format: each row is [idx_a, idx_b, distance, size].
    Indices 0..n-1 are original data points; n+ are merged clusters.
    """
    country_to_idx = {c: i for i, c in enumerate(all_countries)}

    cluster_map: dict[frozenset, int] = {}
    for c, idx in country_to_idx.items():
        cluster_map[frozenset([c])] = idx

    next_idx = len(all_countries)
    Z = []

    for merge in merges:
        key_a = frozenset(merge.cluster_a)
        key_b = frozenset(merge.cluster_b)

        idx_a = cluster_map.get(key_a)
        idx_b = cluster_map.get(key_b)
        if idx_a is None or idx_b is None:
            continue

        distance = round(1.0 - merge.similarity, 6)
        size = len(merge.merged)
        Z.append([float(idx_a), float(idx_b), distance, float(size)])

        cluster_map[frozenset(merge.merged)] = next_idx
        next_idx += 1

    return np.array(Z)


def _build_dendrogram(
    merges: list,
    n_clusters: int,
    all_countries: list[str] | None = None,
):
    """
    Build a SciPy dendrogram truncated to the top-level merges.

    Returns a matplotlib Figure rendered via st.pyplot().
    Truncated to show at most 30 leaf groups so labels stay readable.
    Branches below the cut are colored, branches above are red.
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    import matplotlib.pyplot as plt
    import matplotlib

    if all_countries is None:
        all_countries = list(merges[-1].merged) if merges else []

    Z = _merges_to_linkage(merges, all_countries)
    if len(Z) == 0:
        return None

    n = len(all_countries)
    p = min(30, max(n_clusters + 5, n_clusters * 2))

    # Distance at the cut point: the merge that would reduce k+1 to k clusters
    cut_merge_idx = len(merges) - (n_clusters - 1) - 1
    if 0 <= cut_merge_idx < len(merges):
        color_threshold = 1.0 - merges[cut_merge_idx].similarity
    else:
        color_threshold = 0

    fig, ax = plt.subplots(figsize=(14, 6))

    scipy_dendrogram(
        Z,
        labels=all_countries,
        ax=ax,
        truncate_mode="lastp",
        p=p,
        leaf_rotation=45,
        leaf_font_size=8,
        color_threshold=color_threshold,
        above_threshold_color="#EF5350",
    )

    if color_threshold > 0:
        ax.axhline(
            y=color_threshold,
            color="gray",
            linestyle="--",
            alpha=0.6,
        )
        ax.text(
            ax.get_xlim()[1] * 0.98,
            color_threshold + 0.005,
            f"k = {n_clusters}",
            ha="right",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    ax.set_ylabel("Distance (1 - similarity)")
    ax.set_title(f"Agglomerative Dendrogram (truncated to top {p} groups)")
    fig.tight_layout()

    return fig


# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Similarity Matrix",
    "🌿 Agglomerative Clustering",
    "🎯 K-Means Clustering",
    "⚖️ Comparison & Evaluation",
])


# Tab 1 - Similarity Matrix
# ----------------------------------------------------------------------

with tab1:
    st.header("Similarity Matrix")
    st.caption(
        "Build or refresh the 193×193 UN member state similarity matrix. "
        "Use the sample viewer to inspect any subset without loading the full heatmap."
    )

    st.markdown("**Matrix actions:**")
    col_btn_1, col_btn_2, col_spacer = st.columns([1.6, 1.6, 2])
    with col_btn_1:
        do_recompute = st.button(
            "Recompute Matrix",
            help=(
                "Recompute all TED similarity pairs using the XML files already "
                "on disk. Use this if you changed the preprocessor or TED code "
                "but the underlying Wikipedia data is still fine. "
                "Does NOT contact Wikipedia."
            ),
        )
    with col_btn_2:
        do_rescrape = st.button(
            "Re-scrape + Recompute",
            type="primary",
            help=(
                "Fetch fresh infoboxes from Wikipedia for every country, then "
                "recompute the full similarity matrix. Use this when you want "
                "current Wikipedia data. Takes several minutes."
            ),
        )

    if do_recompute:
        with st.spinner("Recomputing similarity matrix from cached XML files..."):
            try:
                build_matrix(WORKING_SET, overwrite=True, rescrape=False)
                st.success("Matrix recomputed and cached successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error recomputing matrix: {exc}")

    if do_rescrape:
        with st.spinner(
            "Re-scraping Wikipedia and recomputing the matrix. "
            "This takes several minutes — proactive 1 req/s pacing is enforced "
            "to stay under Wikipedia's rate limit."
        ):
            try:
                build_matrix(WORKING_SET, overwrite=True, rescrape=True)
                st.success("Matrix rebuilt from fresh Wikipedia data and cached.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error rebuilding matrix: {exc}")

    matrix, countries, skipped = _load_matrix_from_disk()

    if matrix is None:
        st.info(
            "No cached matrix found. Click one of the buttons above to compute it. "
            f"The full working set has {len(WORKING_SET)} UN member countries."
        )
    else:
        n = len(countries)
        n_pairs = n * (n - 1) // 2
        expected = len(WORKING_SET)
        if n == expected:
            st.success(
                f"Matrix loaded: **{n} countries** · **{n_pairs:,} pairs** "
                f"(full working set)"
            )
        else:
            st.warning(
                f"Matrix loaded: **{n} of {expected} countries** · "
                f"**{n_pairs:,} pairs**. "
                f"{len(skipped)} countries were skipped during build "
                f"(no infobox / failed validation). Clustering proceeds on "
                f"the {n} countries that loaded successfully."
            )

        if skipped:
            with st.expander(f"Skipped countries ({len(skipped)})"):
                st.write(", ".join(sorted(skipped)))
                st.caption(
                    "These countries are excluded from the similarity matrix "
                    "and from all clustering output (map, pie chart, dendrogram). "
                    "Causes are typically: non-standard Wikipedia infobox, "
                    "minimum-viability check failure, or repeated scrape errors."
                )

        # Scannable sample
        st.subheader("Matrix Sample")
        st.caption(
            "Select which countries to preview. "
            "The full matrix is too large to display all at once. "
            "use this to inspect specific subsets."
        )

        sample_size = st.slider(
            "Sample size (countries)",
            min_value=5,
            max_value=min(50, n),
            value=min(DEFAULT_SAMPLE_SIZE, n),
            step=5,
        )

        selected_sample = st.multiselect(
            "Countries to include in sample (leave empty to use first N alphabetically)",
            options=sorted(countries),
            default=[],
        )

        if not selected_sample:
            # Default: first N countries in alphabetical order
            display_countries = sorted(countries)[:sample_size]
        else:
            display_countries = selected_sample[:sample_size]

        st.plotly_chart(
            _build_heatmap(
                matrix,
                display_countries,
                title=f"Similarity Sample: {len(display_countries)} countries",
            ),
            use_container_width=True,
        )

        # Raw scores table
        with st.expander(f"Raw scores for sample ({len(display_countries)}×{len(display_countries)})"):
            sample_df = pd.DataFrame(
                [[matrix[c1][c2] for c2 in display_countries] for c1 in display_countries],
                index=display_countries,
                columns=display_countries,
            )
            st.dataframe(sample_df.style.format("{:.4f}").background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1
            ))

        # Full matrix download
        with st.expander("Download full matrix as CSV"):
            full_df = pd.DataFrame(
                [[matrix[c1][c2] for c2 in countries] for c1 in countries],
                index=countries,
                columns=countries,
            )
            st.download_button(
                label="Download 193×193 matrix (CSV)",
                data=full_df.to_csv(),
                file_name="similarity_matrix_193.csv",
                mime="text/csv",
            )


# Tab 2 - Agglomerative Hierarchical Clustering
# ----------------------------------------------------------------------

with tab2:
    st.header("Agglomerative Hierarchical Clustering")
    st.caption(
        "Bottom-up hierarchical clustering. Choose a linkage method and a cut method "
        "(by k clusters or by similarity threshold)."
    )

    matrix, countries, _ = _load_matrix_from_disk()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        # Linkage selector (Ch. 10, slides 79-81)
        _LINKAGE_LABELS = {
            "average": "Average link (avg) - most robust against noise, most widely used",
            "complete": "Complete link (min) - compact clusters, tends to break large ones",
            "single":   "Single link (max)   - handles non-globular shapes, can chain long/skinny clusters",
        }
        agg_linkage = st.selectbox(
            "Linkage method",
            options=list(_LINKAGE_LABELS.keys()),
            index=0,  # default: average
            format_func=lambda key: _LINKAGE_LABELS[key],
            help=(
                "Inter-cluster similarity rule used when choosing which two clusters "
                "to merge at each step (Ch. 10, slides 79-81)."
            ),
        )

        col1, col2 = st.columns(2)
        with col1:
            cut_method = st.radio(
                "Cut method",
                ["Number of clusters (k)", "Similarity threshold"],
            )
        with col2:
            if cut_method == "Number of clusters (k)":
                agg_k = st.slider(
                    "k",
                    min_value=2,
                    max_value=len(countries),
                    value=min(7, len(countries)),
                    help=(
                        "Number of output clusters. Range goes from 2 up to the "
                        "number of countries (each country in its own cluster)."
                    ),
                )
                agg_threshold = None
            else:
                agg_threshold = st.slider(
                    "Similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.01,
                )
                agg_k = None

        if st.button("Run Agglomerative Clustering", type="primary"):
            with st.spinner("Clustering (this may take a moment for 193 countries)..."):
                try:
                    kwargs = (
                        {"k": agg_k}
                        if agg_k is not None
                        else {"threshold": agg_threshold}
                    )
                    agg_result = agglomerative(
                        matrix, countries, linkage=agg_linkage, **kwargs
                    )
                    st.session_state["agg_result"] = agg_result
                    st.session_state["agg_matrix"] = matrix
                    st.session_state["agg_countries"] = countries
                except Exception as exc:
                    st.error(f"Clustering error: {exc}")

        if "agg_result" in st.session_state:
            result = st.session_state["agg_result"]
            agg_matrix = st.session_state["agg_matrix"]
            agg_countries = st.session_state["agg_countries"]
            n_clusters = len(result.flat_clusters)

            st.subheader(f"Results: {n_clusters} clusters")

            col_left, col_right = st.columns(2)
            with col_left:
                st.dataframe(
                    _cluster_table(result.flat_clusters, matrix=agg_matrix),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_right:
                sizes = [len(c) for c in result.flat_clusters]
                labels = [f"Cluster {i+1}" for i in range(n_clusters)]
                pie_color_map = {
                    labels[i]: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                    for i in range(n_clusters)
                }
                fig = px.pie(
                    values=sizes,
                    names=labels,
                    title="Cluster size distribution",
                    color=labels,
                    color_discrete_map=pie_color_map,
                    category_orders={"names": labels},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Cluster map
            st.subheader("Cluster Map")
            map_fig = _build_cluster_map(
                result.flat_clusters,
                title=f"Agglomerative Clustering: {n_clusters} clusters",
            )
            st.plotly_chart(map_fig, use_container_width=True)

            # Dendrogram
            st.subheader("Dendrogram")
            st.caption(
                "Blue bars are merges that form the final clusters. "
                "Red bars are merges that would reduce the cluster count further. "
                "The dashed line marks the cut point."
            )
            dendro_fig = _build_dendrogram(
                result.dendrogram.merges, n_clusters, agg_countries
            )
            if dendro_fig is not None:
                st.pyplot(dendro_fig)
            else:
                st.warning("Could not build dendrogram.")

            with st.expander("Full merge sequence (table)"):
                merge_data = [
                    {
                        "Step": i + 1,
                        "Group A": ", ".join(sorted(s.cluster_a)),
                        "Group B": ", ".join(sorted(s.cluster_b)),
                        "Avg-Link Similarity": s.similarity,
                        "Merged size": len(s.merged),
                    }
                    for i, s in enumerate(result.dendrogram.merges)
                ]
                st.dataframe(
                    pd.DataFrame(merge_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # Internal evaluation: intra- and inter-cluster similarity
            st.subheader("Internal Evaluation: Intra- and Inter-cluster Similarity")
            st.caption(
                "Intra-cluster similarity (PGMA): sum over clusters of the average "
                "pairwise similarity inside each cluster. Higher is better. "
                "Inter-cluster similarity: average of average-link similarities over "
                "all cluster pairs. Lower is better. (Ch. 10, slides 75-81.)"
            )
            try:
                eval_result = evaluate(result.flat_clusters, agg_matrix)
                _eval_card(eval_result)
            except ValueError as exc:
                st.warning(str(exc))


# Tab 3 - K-Means Clustering
# ----------------------------------------------------------------------

with tab3:
    st.header("K-Means Clustering")
    st.caption(
        "Partitional clustering with medoid-based centroids. "
        "Multiple restarts; best result by intra-cluster similarity is kept."
    )

    matrix, countries, _ = _load_matrix_from_disk()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            km_k = st.slider(
                "Number of clusters (k)",
                min_value=2,
                max_value=len(countries),
                value=min(7, len(countries)),
                help=(
                    "Number of output clusters. Range goes from 2 up to the "
                    "number of countries (each country in its own cluster)."
                ),
            )
        with col2:
            km_runs = st.slider("Random restarts", min_value=1, max_value=20, value=5)
        with col3:
            km_max_iter = st.slider(
                "Max iterations per run", min_value=10, max_value=500, value=100
            )

        if st.button("Run K-Means Clustering", type="primary"):
            with st.spinner(
                f"Running K-Means with k={km_k}, {km_runs} restarts..."
            ):
                try:
                    km_result = kmeans(
                        matrix,
                        countries,
                        k=km_k,
                        max_iterations=km_max_iter,
                        n_runs=km_runs,
                    )
                    st.session_state["km_result"] = km_result
                    st.session_state["km_matrix"] = matrix
                    st.session_state["km_countries"] = countries
                except Exception as exc:
                    st.error(f"Clustering error: {exc}")

        if "km_result" in st.session_state:
            result = st.session_state["km_result"]
            km_matrix = st.session_state["km_matrix"]

            st.subheader(
                f"Results: {result.k} clusters · "
                f"converged in {result.iterations_used} iterations"
            )
            st.caption(
                f"Total intra-cluster similarity: **{result.intra_cluster_similarity:.4f}** "
                "(sum of sim(country, medoid) across all clusters)"
            )

            col_left, col_right = st.columns(2)
            with col_left:
                st.dataframe(
                    _cluster_table(result.clusters, result.medoids, km_matrix),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Medoid = most representative country in each cluster")
            with col_right:
                sizes = [len(c) for c in result.clusters]
                labels = [
                    f"Cluster {i+1} [{result.medoids[i]}]"
                    for i in range(len(result.clusters))
                ]
                pie_color_map = {
                    labels[i]: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                    for i in range(len(result.clusters))
                }
                fig = px.pie(
                    values=sizes,
                    names=labels,
                    title="Cluster size distribution",
                    color=labels,
                    color_discrete_map=pie_color_map,
                    category_orders={"names": labels},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Cluster map
            st.subheader("Cluster Map")
            map_fig = _build_cluster_map(
                result.clusters,
                title=f"K-Means Clustering: {result.k} clusters",
            )
            st.plotly_chart(map_fig, use_container_width=True)

            # Internal evaluation: intra- and inter-cluster similarity
            st.subheader("Internal Evaluation: Intra- and Inter-cluster Similarity")
            st.caption(
                "Intra-cluster similarity (PGMA): sum over clusters of the average "
                "pairwise similarity inside each cluster. Higher is better. "
                "Inter-cluster similarity: average of average-link similarities over "
                "all cluster pairs. Lower is better. (Ch. 10, slides 75-81.)"
            )
            try:
                eval_result = evaluate(result.clusters, km_matrix)
                _eval_card(eval_result)
            except ValueError as exc:
                st.warning(str(exc))


# Tab 4 - Algorithm Comparison & Evaluation
# ----------------------------------------------------------------------

with tab4:
    st.header("Algorithm Comparison & Evaluation")
    st.caption(
        "Run both algorithms with the same k and compare cluster assignments "
        "and internal evaluation metrics (intra- and inter-cluster similarity) "
        "side by side."
    )

    matrix, countries, _ = _load_matrix_from_disk()

    if matrix is None:
        st.info("Build the similarity matrix first (Matrix tab).")
    else:
        # Linkage selector for the agglomerative side of the comparison
        _CMP_LINKAGE_LABELS = {
            "average": "Average link (avg) - most robust, most widely used",
            "complete": "Complete link (min) - compact clusters",
            "single":   "Single link (max)   - handles non-globular shapes",
        }
        compare_linkage = st.selectbox(
            "Agglomerative linkage method",
            options=list(_CMP_LINKAGE_LABELS.keys()),
            index=0,  # default: average
            format_func=lambda key: _CMP_LINKAGE_LABELS[key],
            key="compare_linkage",
            help="Linkage method used by the agglomerative side of this comparison.",
        )

        col1, col2 = st.columns(2)
        with col1:
            compare_k = st.slider(
                "Number of clusters (k)",
                min_value=2,
                max_value=len(countries),
                value=min(7, len(countries)),
                key="compare_k",
                help=(
                    "Number of output clusters. Range goes from 2 up to the "
                    "number of countries (each country in its own cluster)."
                ),
            )
        with col2:
            compare_runs = st.slider(
                "K-Means restarts",
                min_value=1,
                max_value=20,
                value=5,
                key="compare_runs",
            )

        if st.button("Run Both Algorithms", type="primary"):
            with st.spinner("Running Agglomerative..."):
                try:
                    agg = agglomerative(
                        matrix, countries, k=compare_k, linkage=compare_linkage
                    )
                except Exception as exc:
                    st.error(f"Agglomerative error: {exc}")
                    st.stop()

            with st.spinner(f"Running K-Means ({compare_runs} restarts)..."):
                try:
                    km = kmeans(matrix, countries, k=compare_k, n_runs=compare_runs)
                except Exception as exc:
                    st.error(f"K-Means error: {exc}")
                    st.stop()

            st.session_state["compare_agg"] = agg
            st.session_state["compare_km"] = km
            st.session_state["compare_matrix"] = matrix

        if "compare_agg" in st.session_state:
            agg = st.session_state["compare_agg"]
            km = st.session_state["compare_km"]
            cmp_matrix = st.session_state["compare_matrix"]

            # Cluster assignments side by side
            st.subheader("Cluster Assignments")
            col_agg, col_km = st.columns(2)
            with col_agg:
                st.markdown("**Agglomerative**")
                st.dataframe(
                    _cluster_table(agg.flat_clusters, matrix=cmp_matrix),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_km:
                st.markdown("**K-Means**")
                st.dataframe(
                    _cluster_table(km.clusters, km.medoids, cmp_matrix),
                    use_container_width=True,
                    hide_index=True,
                )

            # Internal evaluation comparison: intra- and inter-cluster similarity
            st.subheader("Internal Evaluation: Intra- and Inter-cluster Similarity")
            st.caption(
                "Internal measures (Ch. 10, slides 75-81). No external reference "
                "data needed. Intra-cluster similarity (PGMA) HIGHER is better; "
                "inter-cluster similarity LOWER is better."
            )

            try:
                agg_eval = evaluate(agg.flat_clusters, cmp_matrix)
                km_eval = evaluate(km.clusters, cmp_matrix)
            except ValueError as exc:
                st.warning(str(exc))
                st.stop()

            # Metric summary table
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Intra-cluster similarity (higher = better)",
                    "Inter-cluster similarity (lower = better)",
                    "Clusters",
                    "Countries",
                ],
                "Agglomerative": [
                    agg_eval.intra_cluster_similarity,
                    agg_eval.inter_cluster_similarity,
                    agg_eval.n_clusters,
                    agg_eval.n_objects,
                ],
                "K-Means": [
                    km_eval.intra_cluster_similarity,
                    km_eval.inter_cluster_similarity,
                    km_eval.n_clusters,
                    km_eval.n_objects,
                ],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Bar chart: intra and inter side by side for both algorithms
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Agglomerative",
                x=["Intra-cluster similarity", "Inter-cluster similarity"],
                y=[agg_eval.intra_cluster_similarity, agg_eval.inter_cluster_similarity],
                marker_color="#2196F3",
            ))
            fig.add_trace(go.Bar(
                name="K-Means",
                x=["Intra-cluster similarity", "Inter-cluster similarity"],
                y=[km_eval.intra_cluster_similarity, km_eval.inter_cluster_similarity],
                marker_color="#4CAF50",
            ))
            fig.update_layout(
                barmode="group",
                title=f"Intra- and Inter-cluster Similarity Comparison (k={compare_k})",
                height=400,
                yaxis_title="Similarity",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster size distribution
            st.subheader("Cluster Size Distribution")
            col_sz_agg, col_sz_km = st.columns(2)
            with col_sz_agg:
                agg_sizes = sorted([len(c) for c in agg.flat_clusters], reverse=True)
                fig_agg = px.bar(
                    x=[f"C{i+1}" for i in range(len(agg_sizes))],
                    y=agg_sizes,
                    title=f"Agglomerative: sizes",
                    labels={"x": "Cluster", "y": "# Countries"},
                    color_discrete_sequence=["#2196F3"],
                )
                st.plotly_chart(fig_agg, use_container_width=True)
            with col_sz_km:
                km_sizes = sorted([len(c) for c in km.clusters], reverse=True)
                fig_km = px.bar(
                    x=[f"C{i+1}" for i in range(len(km_sizes))],
                    y=km_sizes,
                    title=f"K-Means: sizes",
                    labels={"x": "Cluster", "y": "# Countries"},
                    color_discrete_sequence=["#4CAF50"],
                )
                st.plotly_chart(fig_km, use_container_width=True)