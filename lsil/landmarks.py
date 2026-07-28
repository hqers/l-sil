"""
Cluster-aware landmark selection for L-Sil.

Implements the landmark selection strategy from Pratama et al. (2026):
1. Proportional allocation per cluster
2. Central (80%) + boundary (20%) split
3. Farthest-first seeding for diversity
4. Distance caching for reuse
"""

import numpy as np
from typing import Optional, Tuple, Dict


def select_landmarks(
    X: np.ndarray,
    labels: np.ndarray,
    n_landmarks: Optional[int] = None,
    landmark_frac: float = 0.2,
    central_ratio: float = 0.8,
    random_state: Optional[int] = None,
    strategy: str = "cluster_aware",
    pool_factor: int = 5,
    pool_frac_cap: Optional[float] = 0.15,
    _dist_func=None,
    _is_cat=None,
    _ranges=None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Select representative landmarks from the dataset.

    Parameters
    ----------
    X : array of shape (n, p)
        Dataset with mixed-type features.
    labels : array of shape (n,)
        Cluster assignments for each point.
    n_landmarks : int, optional
        Total number of landmarks. Default: landmark_frac * n.
    landmark_frac : float
        Fraction of data to use as landmarks (default 0.2 = 20%).
    central_ratio : float
        Fraction of each cluster's landmarks allocated to central points
        (default 0.8 = 80% central, 20% boundary).
    random_state : int, optional
        Random seed for reproducibility.
    strategy : str
        "cluster_aware" (default), "random", or "kcenter".
        NOTE: "kcenter" currently performs uniform random selection within each
        cluster, not k-center/farthest-first. Kept for baseline compatibility.
    pool_factor : int
        Candidate pool size multiplier: pool = max(target*pool_factor, target+5).
        pool_factor <= 0 gives the degenerate pool == target, which reproduces
        the pre-patch behaviour where farthest-first was a no-op and
        random_state had no effect (retained for ablation/reproducibility).
    pool_frac_cap : float, optional
        Upper bound on the candidate pool as a fraction of the cluster size.
        Without it, pool coverage is approximately
            pool / n_cluster ~ pool_factor * central_ratio * c / sqrt(n),
        i.e. it decays as 1/sqrt(n): on small datasets the pool swallowed
        70-100% of the cluster and the central/boundary distinction collapsed.
        Default 0.15 bounds coverage from above and keeps the selection
        behaviour comparable across dataset scales, without changing the
        O(n) cost of the selection step. Set None for the uncapped rule.
    _dist_func : callable, optional
        Distance function for farthest-first. If None, uses Gower.
    _is_cat : array, optional
        Categorical feature mask (passed to Gower).
    _ranges : array, optional
        Feature ranges (passed to Gower).

    Returns
    -------
    landmark_indices : array of shape (m,)
        Indices into X of selected landmarks.
    landmark_labels : array of shape (m,)
        Cluster labels of the landmarks.
    info : dict
        Metadata: counts per cluster, central/boundary breakdown.
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    if n_landmarks is None:
        n_landmarks = max(K * 2, int(n * landmark_frac))
    n_landmarks = min(n_landmarks, n)

    if strategy == "random":
        idx = rng.choice(n, size=n_landmarks, replace=False)
        return idx, labels[idx], {"strategy": "random", "n_landmarks": len(idx)}

    # --- Cluster-aware allocation ---
    cluster_sizes = {k: np.sum(labels == k) for k in unique_labels}
    total = sum(cluster_sizes.values())

    # Proportional allocation with minimum floor of 2 per cluster
    allocation = {}
    for k in unique_labels:
        quota = max(2, int(round(n_landmarks * cluster_sizes[k] / total)))
        allocation[k] = min(quota, cluster_sizes[k])

    # Adjust to hit target
    allocated = sum(allocation.values())
    if allocated < n_landmarks:
        # Distribute remainder to largest clusters
        sorted_k = sorted(unique_labels, key=lambda k: cluster_sizes[k], reverse=True)
        for k in sorted_k:
            if allocated >= n_landmarks:
                break
            extra = min(n_landmarks - allocated, cluster_sizes[k] - allocation[k])
            allocation[k] += extra
            allocated += extra

    all_indices = []
    info = {"strategy": strategy, "clusters": {}}

    for k in unique_labels:
        mask = labels == k
        cluster_idx = np.where(mask)[0]
        quota = allocation[k]

        n_central = max(1, int(round(quota * central_ratio)))
        n_boundary = quota - n_central

        if strategy == "kcenter":
            # Simple random for kcenter baseline
            sel = rng.choice(cluster_idx, size=quota, replace=False)
            all_indices.extend(sel.tolist())
            info["clusters"][int(k)] = {"total": quota, "central": quota, "boundary": 0}
            continue

        # --- Central landmarks: closest to cluster centroid ---
        X_cluster = X[cluster_idx]

        # Compute centroid for numerical, mode for categorical
        centroid = _compute_mixed_centroid(X_cluster, _is_cat)

        # Distance from each cluster member to centroid
        dists_to_centroid = _mixed_dist_to_point(X_cluster, centroid, _is_cat, _ranges)
        sorted_by_dist = np.argsort(dists_to_centroid)

        # --- Oversized candidate pools ---
        # FIX: pools are deliberately larger than the final target count so that
        # _farthest_first_subsample below has room to actually pick a diverse
        # subset. Previously, central/boundary were sliced to exactly n_central/
        # n_boundary points and then passed straight into farthest-first with
        # target == len(candidates), which made the "n_cand <= target" guard
        # always true and skipped the traversal entirely (random_state was a
        # dead parameter for the "cluster_aware" strategy). Pattern mirrors the
        # working notebook implementation (pool = max(target*5, target+5)).
        n_cluster = len(sorted_by_dist)

        def _pool_size(target: int) -> int:
            if pool_factor <= 0:
                return int(min(target, n_cluster))
            size = max(target * pool_factor, target + 5)
            if pool_frac_cap is not None:
                size = min(size, max(target, int(np.ceil(pool_frac_cap * n_cluster))))
            return int(min(size, n_cluster))

        pool_central = sorted_by_dist[: _pool_size(n_central)]
        remaining = sorted_by_dist[len(pool_central):]
        pool_boundary = (
            remaining[-min(_pool_size(n_boundary), len(remaining)):]
            if (n_boundary > 0 and len(remaining) > 0) else np.array([], dtype=int)
        )

        # --- Farthest-first refinement: pick target count from each pool ---
        central_local = (
            _farthest_first_subsample(X_cluster, pool_central, n_central, rng, _is_cat, _ranges)
            if n_central > 0 else np.array([], dtype=int)
        )
        boundary_local = (
            _farthest_first_subsample(X_cluster, pool_boundary, n_boundary, rng, _is_cat, _ranges)
            if n_boundary > 0 else np.array([], dtype=int)
        )

        # De-duplicate in case oversized pools overlapped near the pool boundary
        selected_local = np.array(
            sorted(set(central_local.tolist()) | set(boundary_local.tolist())), dtype=int
        )

        all_indices.extend(cluster_idx[selected_local].tolist())
        info["clusters"][int(k)] = {
            "total": len(selected_local),
            "central": len(central_local),
            "boundary": len(boundary_local),
            "n_cluster": int(n_cluster),
            "pool_central": int(len(pool_central)),
            "pool_boundary": int(len(pool_boundary)),
            "coverage_central": float(len(pool_central) / max(n_cluster, 1)),
        }

    landmark_indices = np.array(all_indices, dtype=int)
    info["n_landmarks"] = len(landmark_indices)
    info["pool_factor"] = pool_factor
    info["pool_frac_cap"] = pool_frac_cap
    if info.get("clusters"):
        _cov = [v["coverage_central"] for v in info["clusters"].values()
                if "coverage_central" in v]
        if _cov:
            info["coverage_central_min"] = float(min(_cov))
            info["coverage_central_mean"] = float(sum(_cov) / len(_cov))
            info["coverage_central_max"] = float(max(_cov))

    return landmark_indices, labels[landmark_indices], info


def _compute_mixed_centroid(X: np.ndarray, is_cat: Optional[np.ndarray]) -> np.ndarray:
    """Compute centroid: mean for numerical, mode for categorical."""
    p = X.shape[1]
    centroid = np.empty(p, dtype=object)

    for j in range(p):
        col = X[:, j]
        if is_cat is not None and is_cat[j]:
            # Mode
            vals, counts = np.unique(col, return_counts=True)
            centroid[j] = vals[np.argmax(counts)]
        else:
            try:
                centroid[j] = np.nanmean(col.astype(float))
            except (ValueError, TypeError):
                vals, counts = np.unique(col, return_counts=True)
                centroid[j] = vals[np.argmax(counts)]

    return centroid


def _mixed_dist_to_point(
    X: np.ndarray,
    point: np.ndarray,
    is_cat: Optional[np.ndarray],
    ranges: Optional[np.ndarray],
) -> np.ndarray:
    """Gower distance from each row of X to a single point."""
    n, p = X.shape
    dists = np.zeros(n, dtype=np.float64)

    for j in range(p):
        if is_cat is not None and is_cat[j]:
            dists += (X[:, j] != point[j]).astype(float)
        else:
            try:
                col = X[:, j].astype(float)
                ref = float(point[j])
                r = ranges[j] if ranges is not None else 1.0
                dists += np.abs(col - ref) / max(r, 1e-12)
            except (ValueError, TypeError):
                dists += (X[:, j] != point[j]).astype(float)

    return dists / p


def _farthest_first_subsample(
    X_cluster: np.ndarray,
    candidates: np.ndarray,
    target: int,
    rng: np.random.RandomState,
    is_cat: Optional[np.ndarray],
    ranges: Optional[np.ndarray],
) -> np.ndarray:
    """
    Farthest-first traversal to maximize diversity among selected landmarks.
    """
    X_cand = X_cluster[candidates]
    n_cand = len(candidates)

    if n_cand <= target:
        return candidates

    # Start with random seed
    selected = [rng.randint(n_cand)]
    min_dists = np.full(n_cand, np.inf)

    for _ in range(target - 1):
        last = selected[-1]
        d = _mixed_dist_to_point(
            X_cand, X_cand[last], is_cat, ranges
        )
        min_dists = np.minimum(min_dists, d)
        # Mask already selected
        min_dists_copy = min_dists.copy()
        for s in selected:
            min_dists_copy[s] = -1
        next_idx = np.argmax(min_dists_copy)
        selected.append(next_idx)

    return candidates[np.array(selected)]
