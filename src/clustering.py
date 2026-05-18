"""
Clustering Algorithms (Project 2)

Implements two clustering algorithms:

  1. Agglomerative Hierarchical Clustering
     - Bottom-up; user chooses linkage method: single, complete, or average
     - Builds a full dendrogram; user cuts at k or a similarity threshold

  2. K-Means Partitional Clustering
     - Medoid-based (closest real data object to the mean) since we operate
       on a precomputed similarity matrix rather than raw feature vectors
     - Multiple random restarts; best result kept by intra-cluster similarity
     - Convergence: no object changes cluster between iterations

Neither algorithm uses or references any geographic data.
Clustering is driven entirely by the pairwise similarity matrix from Project 1.
"""

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures: Agglomerative
# ---------------------------------------------------------------------------

@dataclass
class MergeStep:
    """Records a single merge event in the agglomerative dendrogram."""
    cluster_a: list[str]
    cluster_b: list[str]
    similarity: float          # Inter-cluster similarity at merge time (linkage-dependent)
    merged: list[str]          # Combined cluster after the merge


@dataclass
class Dendrogram:
    """
    Sequence of merge steps produced by agglomerative clustering.
    Merges are stored in order (earliest = highest similarity first).
    """
    merges: list[MergeStep] = field(default_factory=list)

    def cut_at_threshold(self, threshold: float) -> list[list[str]]:
        """
        Return flat clusters by stopping merges whose similarity falls
        below `threshold`. Clusters whose merge similarity >= threshold
        are kept merged; the rest remain as separate groups.
        """
        # Replay merges; only accept those above the threshold
        active: dict[tuple, list[str]] = {}

        # Start: every country is its own singleton cluster
        all_countries: list[str] = []
        if self.merges:
            all_countries = list(self.merges[-1].merged)
        for country in all_countries:
            active[tuple([country])] = [country]

        for merge in self.merges:
            if merge.similarity < threshold:
                # Stop merging: similarity below threshold
                break
            # Apply merge: remove the two source clusters, add the merged one
            key_a = tuple(sorted(merge.cluster_a))
            key_b = tuple(sorted(merge.cluster_b))
            active.pop(key_a, None)
            active.pop(key_b, None)
            key_merged = tuple(sorted(merge.merged))
            active[key_merged] = merge.merged

        return list(active.values())

    def cut_at_k(self, k: int) -> list[list[str]]:
        """
        Return exactly k flat clusters by replaying merges and stopping
        once the desired number of clusters is reached.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        all_countries = self.merges[-1].merged if self.merges else []
        if k >= len(all_countries):
            return [[c] for c in all_countries]

        # Start with N singleton clusters
        active: list[list[str]] = [[c] for c in all_countries]

        for merge in self.merges:
            if len(active) <= k:
                break
            key_a = tuple(sorted(merge.cluster_a))
            key_b = tuple(sorted(merge.cluster_b))
            active = [c for c in active if tuple(sorted(c)) not in (key_a, key_b)]
            active.append(merge.merged)

        return active


@dataclass
class AgglomerativeResult:
    """Returned by agglomerative(). Contains the full dendrogram and flat cut."""
    dendrogram: Dendrogram
    flat_clusters: list[list[str]]
    linkage: str = "average"


# ---------------------------------------------------------------------------
# Data structures: K-Means
# ---------------------------------------------------------------------------

@dataclass
class KMeansResult:
    """Returned by kmeans(). Contains cluster assignments and quality score."""
    clusters: list[list[str]]
    medoids: list[str]                 # Most representative object per cluster
    intra_cluster_similarity: float    # Sum of sim(object, medoid); higher is better
    k: int
    iterations_used: int


# ---------------------------------------------------------------------------
# Agglomerative clustering
# ---------------------------------------------------------------------------

def _average_link_similarity(
    cluster_a: list[str],
    cluster_b: list[str],
    matrix: dict,
) -> float:
    """
    Average-link inter-cluster similarity.
    Cluster similarity = average similarity of all cross-cluster pairs.
    Most robust against noise; most widely used (Ch. 10, slide 81).
    """
    total = sum(matrix[a][b] for a in cluster_a for b in cluster_b)
    return total / (len(cluster_a) * len(cluster_b))


def _single_link_similarity(
    cluster_a: list[str],
    cluster_b: list[str],
    matrix: dict,
) -> float:
    """
    Single-link inter-cluster similarity (Ch. 10, slide 79).
    Cluster similarity = similarity of the two MOST similar cross-cluster objects
    (i.e. the maximum). Can produce long, skinny clusters but handles non-globular
    shapes well.
    """
    return max(matrix[a][b] for a in cluster_a for b in cluster_b)


def _complete_link_similarity(
    cluster_a: list[str],
    cluster_b: list[str],
    matrix: dict,
) -> float:
    """
    Complete-link inter-cluster similarity (Ch. 10, slide 80).
    Cluster similarity = similarity of the two LEAST similar cross-cluster objects
    (i.e. the minimum). Produces compact clusters but tends to break large ones.
    """
    return min(matrix[a][b] for a in cluster_a for b in cluster_b)


# Linkage method registry: name -> similarity function
_LINKAGE_METHODS = {
    "average": _average_link_similarity,
    "single": _single_link_similarity,
    "complete": _complete_link_similarity,
}


def agglomerative(
    matrix: dict,
    countries: list[str],
    k: int | None = None,
    threshold: float | None = None,
    linkage: str = "average",
) -> AgglomerativeResult:
    """
    Agglomerative hierarchical clustering:

    1. Initialise: each object is its own cluster.
    2. Repeat:
       a. Find the two clusters with maximum inter-cluster similarity
          (using the chosen linkage method).
       b. Merge them; record the merge step.
    3. Until one cluster remains (full dendrogram built).
    4. Cut the dendrogram at k clusters or at a similarity threshold.

    Args:
        matrix:    Pairwise similarity matrix (nested dict).
        countries: List of country names to cluster.
        k:         Desired number of output clusters (mutually exclusive with threshold).
        threshold: Similarity threshold at which to stop merging.
        linkage:   Inter-cluster similarity method: "average" (default), "single",
                   or "complete". See Ch. 10, slides 79-81.

    Returns:
        AgglomerativeResult with full dendrogram and flat cluster cut.
    """
    if k is None and threshold is None:
        raise ValueError("Provide either k or threshold.")

    if linkage not in _LINKAGE_METHODS:
        raise ValueError(
            f"Unknown linkage '{linkage}'. "
            f"Choose from: {sorted(_LINKAGE_METHODS.keys())}."
        )

    link_fn = _LINKAGE_METHODS[linkage]

    clusters: list[list[str]] = [[c] for c in countries]
    dendrogram = Dendrogram()

    while len(clusters) > 1:
        best_sim = -1.0
        best_i, best_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = link_fn(clusters[i], clusters[j], matrix)
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j

        merged = clusters[best_i] + clusters[best_j]
        dendrogram.merges.append(MergeStep(
            cluster_a=list(clusters[best_i]),
            cluster_b=list(clusters[best_j]),
            similarity=round(best_sim, 4),
            merged=merged,
        ))

        clusters = [c for idx, c in enumerate(clusters) if idx not in (best_i, best_j)]
        clusters.append(merged)

    flat = dendrogram.cut_at_k(k) if k is not None else dendrogram.cut_at_threshold(threshold)
    return AgglomerativeResult(
        dendrogram=dendrogram,
        flat_clusters=flat,
        linkage=linkage,
    )


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

def _compute_medoid(cluster: list[str], matrix: dict) -> str:
    """
    Find the medoid: the cluster member with the highest average similarity
    to all other members. The medoid is the closest real object to the
    theoretical centroid when working with a precomputed similarity matrix.
    """
    best_country = cluster[0]
    best_avg = -1.0
    for candidate in cluster:
        avg = sum(matrix[candidate][other] for other in cluster) / len(cluster)
        if avg > best_avg:
            best_avg = avg
            best_country = candidate
    return best_country


def _assign_to_medoids(
    countries: list[str],
    medoids: list[str],
    matrix: dict,
) -> list[list[str]]:
    """
    Assignment step:
    Assign each object to the cluster whose medoid it is most similar to.
    Each object goes to exactly one cluster (hard partitioning).
    """
    clusters: list[list[str]] = [[] for _ in medoids]
    for country in countries:
        best_idx = max(
            range(len(medoids)),
            key=lambda i: matrix[country][medoids[i]],
        )
        clusters[best_idx].append(country)
    return clusters


def _total_intra_similarity(
    clusters: list[list[str]],
    medoids: list[str],
    matrix: dict,
) -> float:
    """
    Total intra-cluster similarity: sum of sim(object, medoid) over all
    objects in all clusters. Used to select the best run among restarts.
    Higher = better clustering quality.
    """
    return round(
        sum(
            matrix[country][medoid]
            for cluster, medoid in zip(clusters, medoids)
            for country in cluster
        ),
        4,
    )


def _kmeans_single_run(
    matrix: dict,
    countries: list[str],
    k: int,
    max_iterations: int,
) -> KMeansResult:
    """
    One full run of K-Means:

    1. Initialise: randomly select k medoids.
    2. Repeat:
       a. Assign each object to its closest medoid.
       b. Recompute medoids (object with highest avg similarity to cluster).
    3. Until no object changes cluster (convergence).
    """
    medoids = random.sample(countries, k)

    for iteration in range(1, max_iterations + 1):
        clusters = _assign_to_medoids(countries, medoids, matrix)

        # Guard: ensure no cluster is left empty (can happen with bad init)
        clusters = [
            cluster if cluster else [medoids[i]]
            for i, cluster in enumerate(clusters)
        ]

        new_medoids = [_compute_medoid(cluster, matrix) for cluster in clusters]

        # Convergence check: no medoid changed → no object changed cluster
        if new_medoids == medoids:
            return KMeansResult(
                clusters=clusters,
                medoids=new_medoids,
                intra_cluster_similarity=_total_intra_similarity(
                    clusters, new_medoids, matrix
                ),
                k=k,
                iterations_used=iteration,
            )

        medoids = new_medoids

    # Max iterations reached without convergence
    clusters = _assign_to_medoids(countries, medoids, matrix)
    clusters = [
        cluster if cluster else [medoids[i]]
        for i, cluster in enumerate(clusters)
    ]
    return KMeansResult(
        clusters=clusters,
        medoids=medoids,
        intra_cluster_similarity=_total_intra_similarity(clusters, medoids, matrix),
        k=k,
        iterations_used=max_iterations,
    )


def kmeans(
    matrix: dict,
    countries: list[str],
    k: int,
    max_iterations: int = 100,
    n_runs: int = 5,
) -> KMeansResult:
    """
    K-Means with multiple random restarts to overcome dependency on
    initial centroids.

    Runs the algorithm n_runs times with independent random initial medoids.
    Returns the run with the highest total intra-cluster similarity,
    which corresponds to the lowest SSE.

    NOTE on cluster sizes: K-Means naturally produces unequal cluster sizes
    because each object is assigned to its closest centroid. This is the
    correct assignment step. The algorithm
    is known to not be ideal for clusters with very different sizes, but this is a limitation, not a design goal to override. Enforcing equal sizes would
    violate the assignment rule and degrade intra-cluster similarity.

    Args:
        matrix:         Pairwise similarity matrix (nested dict).
        countries:      List of country names.
        k:              Number of clusters.
        max_iterations: Per-run iteration cap.
        n_runs:         Number of independent restarts.

    Returns:
        KMeansResult for the best-scoring run.
    """
    if k <= 0 or k > len(countries):
        raise ValueError(f"k must be between 1 and {len(countries)}.")

    best_result: KMeansResult | None = None
    for _ in range(n_runs):
        result = _kmeans_single_run(matrix, countries, k, max_iterations)
        if (
            best_result is None
            or result.intra_cluster_similarity > best_result.intra_cluster_similarity
        ):
            best_result = result

    return best_result