"""
L-Sil and LNC* computation — paper-aligned.

L-Sil: Landmark-based Silhouette with top-r aggregation and cluster-size weighting.
LNC*: Landmark Neighbourhood Consistency with Miller-Madow entropy and IQR contrast.

Reference: Pratama, Lubis, Sembiring (2026).
  Eq. 2-3: L-Sil definition
  Eq. 6:   LNC* operational definition
  Theorem 1: O(n^{3/2}) complexity
"""

import numpy as np
from typing import Tuple


def compute_lsil(
    D_to_landmarks: np.ndarray,
    labels: np.ndarray,
    landmark_labels: np.ndarray,
    agg_mode: str = "topk",
    topk: int = 3,
    weighted: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Compute L-Sil from precomputed point-to-landmark distances (paper Eq. 2-3).

    a_L(x) = Agg{ d(x, l) | l in L, lab(l) = c, l != x }
    b_L(x) = min_{c'!=c} Agg{ d(x, l) | l in L, lab(l) = c' }
    s_L(x) = (b - a) / max(a, b)

    Parameters
    ----------
    D_to_landmarks : array (n, m)
    labels : array (n,)
    landmark_labels : array (m,)
    agg_mode : str — "topk" (paper default) | "mean" | "min"
    topk : int — paper r in {3,5}, default 3
    weighted : bool — cluster-size weighted mean (default True)

    Returns
    -------
    lsil_score : float
    per_sample : array (n,)
    """
    n, m = D_to_landmarks.shape
    unique_labels = np.unique(labels)
    lm_masks = {k: np.where(landmark_labels == k)[0] for k in unique_labels}

    if weighted:
        _, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(np.unique(labels), counts.astype(float)))
    else:
        size_map = None

    per_sample = np.zeros(n, dtype=np.float64)

    for i in range(n):
        c = labels[i]
        dists = D_to_landmarks[i]

        own_idx = lm_masks.get(c, np.array([], dtype=int))
        own_dists = dists[own_idx]
        own_nonzero = own_dists[own_dists > 1e-12]
        if len(own_nonzero) == 0:
            own_nonzero = own_dists if len(own_dists) > 0 else None
        if own_nonzero is None or len(own_nonzero) == 0:
            per_sample[i] = 0.0
            continue

        a = _aggregate(own_nonzero, agg_mode, topk)

        b = np.inf
        for k in unique_labels:
            if k == c:
                continue
            other_idx = lm_masks.get(k, np.array([], dtype=int))
            if len(other_idx) == 0:
                continue
            b = min(b, _aggregate(dists[other_idx], agg_mode, topk))

        if b == np.inf:
            per_sample[i] = 0.0
        else:
            denom = max(a, b)
            per_sample[i] = (b - a) / denom if denom > 1e-12 else 0.0

    per_sample = np.clip(per_sample, -1.0, 1.0)

    if weighted and size_map:
        w = np.array([size_map.get(labels[i], 1.0) for i in range(n)])
        score = float(np.sum(w * per_sample) / max(np.sum(w), 1e-12))
    else:
        score = float(np.mean(per_sample))

    return score, per_sample


def _aggregate(dists: np.ndarray, mode: str, topk: int) -> float:
    if len(dists) == 0:
        return np.nan
    if mode == "min":
        return float(np.min(dists))
    elif mode == "topk":
        k = min(topk, len(dists))
        return float(np.mean(np.partition(dists, k - 1)[:k]))
    return float(np.mean(dists))


def compute_lnc_star(
    D_to_landmarks: np.ndarray,
    labels: np.ndarray,
    landmark_labels: np.ndarray,
    landmark_indices: np.ndarray,
    k_neighbors: int = 10,
    alpha: float = 0.7,
    weighted: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Compute LNC* — paper Eq. 6: v(l) = alpha * NC_l + (1-alpha) * Delta_l.

    NC: Miller-Madow corrected entropy-based neighborhood consistency.
    Delta: IQR-normalized (5-95 percentile) distance contrast.

    Parameters
    ----------
    D_to_landmarks : array (n, m)
    labels : array (n,)
    landmark_labels : array (m,)
    landmark_indices : array (m,)
    k_neighbors : int — default 10
    alpha : float — default 0.7 (paper setting)
    weighted : bool — cluster-size weighted (default True)

    Returns
    -------
    lnc_star : float
    per_landmark : array (m,)
    """
    m = len(landmark_indices)
    n = D_to_landmarks.shape[0]
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
    k_eff = min(k_neighbors, n - 1)

    if weighted:
        _, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(unique_labels, counts.astype(float)))
    else:
        size_map = None

    per_landmark = np.zeros(m, dtype=np.float64)
    weights_arr = np.ones(m, dtype=np.float64)

    for j in range(m):
        lm_idx = landmark_indices[j]
        lm_label = landmark_labels[j]
        dists = D_to_landmarks[:, j]

        sorted_idx = np.argsort(dists)
        neighbors = []
        for s in sorted_idx:
            if s != lm_idx and len(neighbors) < k_eff:
                neighbors.append(s)
            if len(neighbors) >= k_eff:
                break
        if len(neighbors) == 0:
            continue

        neighbors = np.array(neighbors)
        neighbor_labels = labels[neighbors]
        neighbor_dists = dists[neighbors]

        # NC: Miller-Madow corrected entropy
        counts_k = np.zeros(K, dtype=float)
        for nl in neighbor_labels:
            counts_k[label_to_idx.get(nl, 0)] += 1.0
        p_dist = counts_k / max(1.0, counts_k.sum())
        nz = p_dist > 0
        H = float(-np.sum(p_dist[nz] * np.log(p_dist[nz] + 1e-12)))
        H += (int(nz.sum()) - 1) / (2.0 * max(1, len(neighbors)))
        NC = 1.0 if K <= 1 else float(np.clip(1.0 - H / np.log(K), 0.0, 1.0))

        # Delta: IQR-normalized distance contrast
        same_mask = neighbor_labels == lm_label
        other_mask = ~same_mask
        if same_mask.sum() > 0 and other_mask.sum() > 0:
            d_intra = float(np.mean(neighbor_dists[same_mask]))
            d_inter = float(np.mean(neighbor_dists[other_mask]))
            q5, q95 = np.percentile(neighbor_dists, [5, 95])
            iqr = max(1e-9, float(q95 - q5))
            delta = float(np.clip((d_inter - d_intra) / iqr, 0.0, 1.0))
        elif same_mask.sum() > 0:
            delta = 1.0
        else:
            delta = 0.0

        per_landmark[j] = float(np.clip(alpha * NC + (1.0 - alpha) * delta, 0.0, 1.0))
        if weighted and size_map:
            weights_arr[j] = size_map.get(lm_label, 1.0)

    if weighted:
        lnc = float(np.sum(weights_arr * per_landmark) / max(np.sum(weights_arr), 1e-12))
    else:
        lnc = float(np.mean(per_landmark))

    return lnc, per_landmark
