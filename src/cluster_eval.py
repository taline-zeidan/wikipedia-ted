"""
Internal Clustering Evaluation (Project 2)

Evaluates clustering quality using the two measures presented in Ch. 10,
slides 75-81 of the course:

  1. Intra-cluster similarity (PGMA -- Pair Group Method Average)
     For each cluster Ci, compute the average pairwise similarity of its
     members (slide 76: PGMA_i = sum Sim(xp, xq) over pairs / Ni, where
     Ni = |Ci|*(|Ci|-1)/2). We then report the AVERAGE of those per-cluster
     PGMA values across the k clusters, which keeps the metric in [0, 1].

     Note on normalisation: the slide writes PGMA as a sum across clusters,
     which lives in [0, k]. For a *comparable* evaluation metric -- same
     range as inter-cluster similarity, and directly readable across
     different values of k -- we average over clusters instead. The
     per-cluster formula is exactly the one in the slide.

     HIGHER means more coherent (tighter) clusters.

  2. Inter-cluster similarity (average-link, averaged over cluster pairs)
     For each pair of clusters, compute the average similarity of all
     cross-cluster pairs, then average across all cluster pairs.
     LOWER means more distinct (better separated) clusters.

Both measures live in [0, 1] (same range as the similarity matrix),
so they can be read and compared directly.

Why PGMA and not SSE for intra-cluster similarity:
  SSE (slide 75) requires a centroid in feature space. We only have a
  precomputed similarity matrix -- no feature vectors -- so SSE is not
  directly applicable. PGMA (slide 76) works on similarities directly
  and is presented in the slides as the alternative for this case.

Why average-link for inter-cluster similarity:
  The slides describe three options for inter-cluster similarity:
  Single Link (max), Complete Link (min), Average Link (avg). Average
  Link is described as "most robust against noise and most widely used"
  (slide 81), which makes it the natural choice for evaluation.
  Single and complete link are still available in clustering.py as
  linkage options for the agglomerative algorithm itself.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Holds the result of an internal cluster evaluation pass."""
    intra_cluster_similarity: float    # PGMA, higher is better
    inter_cluster_similarity: float    # Avg of average-link over cluster pairs; lower is better
    n_clusters: int                    # Number of clusters evaluated
    n_objects: int                     # Total number of clustered objects


# ---------------------------------------------------------------------------
# Per-cluster intra-cluster similarity: PGMA
# ---------------------------------------------------------------------------

def _pgma_cluster(cluster: list[str], matrix: dict) -> float:
    """
    PGMA contribution of a single cluster Ci:
        sum of Sim(xp, xq) over all unordered pairs (xp, xq) in Ci
        divided by Ni = |Ci| * (|Ci| - 1) / 2.

    A singleton cluster has no pairs and contributes 0 to the PGMA sum.
    """
    size = len(cluster)
    if size < 2:
        return 0.0

    total = 0.0
    for i in range(size):
        for j in range(i + 1, size):
            total += matrix[cluster[i]][cluster[j]]

    n_pairs = size * (size - 1) / 2
    return total / n_pairs


def _intra_cluster_similarity(
    clusters: list[list[str]],
    matrix: dict,
) -> float:
    """
    Mean PGMA across the k clusters (Ch. 10, slide 76):

        intra = (1 / k) * sum over clusters Ci of [ avg pairwise Sim inside Ci ]

    Each per-cluster PGMA is in [0, 1] (it's an average of similarities,
    each in [0, 1]), so the mean across clusters is also in [0, 1].
    Singleton clusters contribute 0 (no pairs).

    Higher intra-cluster similarity means more coherent clusters.

    Edge case: an empty clustering ([]) has no clusters, so intra is
    undefined; we return 0.0.
    """
    k = len(clusters)
    if k == 0:
        return 0.0
    return sum(_pgma_cluster(c, matrix) for c in clusters) / k


# ---------------------------------------------------------------------------
# Per-pair inter-cluster similarity: average-link
# ---------------------------------------------------------------------------

def _average_link(
    cluster_a: list[str],
    cluster_b: list[str],
    matrix: dict,
) -> float:
    """
    Average-link similarity between two clusters (Ch. 10, slide 81):
    average similarity of all cross-cluster pairs.
    """
    total = sum(matrix[a][b] for a in cluster_a for b in cluster_b)
    return total / (len(cluster_a) * len(cluster_b))


def _inter_cluster_similarity(
    clusters: list[list[str]],
    matrix: dict,
) -> float:
    """
    Average of the average-link similarities over all unordered cluster pairs.

    Lower inter-cluster similarity means more distinct clusters.

    Edge case: with k = 1 (a single cluster), there are no cluster pairs.
    We return 0.0 in that case (no separation to measure).
    """
    k = len(clusters)
    if k < 2:
        return 0.0

    total = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            total += _average_link(clusters[i], clusters[j], matrix)

    n_pairs = k * (k - 1) / 2
    return total / n_pairs


# ---------------------------------------------------------------------------
# Public evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    clusters: list[list[str]],
    matrix: dict,
) -> EvaluationResult:
    """
    Evaluate a clustering using intra- and inter-cluster similarity.

    Args:
        clusters: Flat list of clusters (each cluster is a list of country names).
        matrix:   Pairwise similarity matrix (nested dict).

    Returns:
        EvaluationResult with the two metrics and basic counts.
    """
    n_clusters = len(clusters)
    n_objects = sum(len(c) for c in clusters)

    intra = _intra_cluster_similarity(clusters, matrix)
    inter = _inter_cluster_similarity(clusters, matrix)

    return EvaluationResult(
        intra_cluster_similarity=round(intra, 4),
        inter_cluster_similarity=round(inter, 4),
        n_clusters=n_clusters,
        n_objects=n_objects,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper (used by the CLI / for debugging)
# ---------------------------------------------------------------------------

def print_evaluation(result: EvaluationResult, label: str = "") -> None:
    """Print an EvaluationResult to stdout in a compact, readable form."""
    header = f"Evaluation{(' - ' + label) if label else ''}"
    print(header)
    print("-" * len(header))
    print(f"  Clusters evaluated : {result.n_clusters}")
    print(f"  Objects clustered  : {result.n_objects}")
    print(f"  Intra-cluster sim. : {result.intra_cluster_similarity:.4f}   (higher = more coherent)")
    print(f"  Inter-cluster sim. : {result.inter_cluster_similarity:.4f}   (lower  = more distinct)")
