"""
lsil — Public API for Landmark-Based Evaluation Metrics.

Drop-in replacement for sklearn silhouette_score on mixed-type data.
Paper-aligned defaults: topk=3, alpha=0.7, |L|=c√n (c=3), weighted=True.

Usage:
    from lsil import lsil_score
    score = lsil_score(X, labels, cat_features=[0, 3, 7])
"""

import numpy as np
import time
from typing import Optional, List, Dict, Any

from .gower import _detect_feature_types, _gower_ranges, gower_distances_to_landmarks, gower_pairwise
from .landmarks import select_landmarks
from .metrics import compute_lsil, compute_lnc_star

__version__ = "0.2.0"


def adaptive_landmark_count(n: int, K: int = 3, c: float = 3.0,
                            per_cluster_min: int = 3, cap_frac: float = 0.2) -> int:
    """
    Paper-aligned: |L| = max(K*min_per, min(c*sqrt(n), cap_frac*n)).
    Theorem 1: yields O(c*p*n^{3/2}) total complexity.
    """
    m_sqrt = int(c * np.sqrt(n))
    m_cap = int(cap_frac * n)
    m_floor = K * per_cluster_min
    return max(m_floor, min(m_sqrt, m_cap, n))


def lsil_score(X, labels, cat_features=None, c=3.0, cap_frac=0.2,
               central_ratio=0.8, random_state=42, agg_mode="topk",
               topk=3, weighted=True, lnc_threshold=0.5) -> float:
    """
    Compute L-Sil score — drop-in replacement for silhouette_score.

    Defaults follow paper: c=3, topk=3, weighted=True.
    Landmark count: |L| = c*sqrt(n), capped at cap_frac*n.
    """
    result = lsil_evaluate(X, labels, cat_features=cat_features, c=c,
                           cap_frac=cap_frac, central_ratio=central_ratio,
                           random_state=random_state, agg_mode=agg_mode,
                           topk=topk, weighted=weighted, lnc_threshold=lnc_threshold)
    return result["lsil"]


def lnc_star_score(X, labels, cat_features=None, c=3.0, cap_frac=0.2,
                   central_ratio=0.8, random_state=42, k_neighbors=10,
                   alpha=0.7, weighted=True) -> float:
    """Compute LNC* score (paper Eq. 6, default alpha=0.7)."""
    result = lsil_evaluate(X, labels, cat_features=cat_features, c=c,
                           cap_frac=cap_frac, central_ratio=central_ratio,
                           random_state=random_state, k_neighbors=k_neighbors,
                           alpha=alpha, weighted=weighted)
    return result["lnc_star"]


def lsil_evaluate(X, labels, cat_features=None, c=3.0, cap_frac=0.2,
                  central_ratio=0.8, random_state=42, lnc_threshold=0.5,
                  k_neighbors=10, alpha=0.7, agg_mode="topk", topk=3,
                  weighted=True, strategy="cluster_aware",
                  return_per_point=False) -> Dict[str, Any]:
    """
    Full L-Sil + LNC* evaluation with paper-aligned defaults.

    Parameters
    ----------
    X : array (n, p) — mixed-type features (dtype=object ok)
    labels : array (n,) — cluster labels
    cat_features : list[int] — categorical column indices
    c : float — sqrt-n scaling constant (default 3, paper c in [1.5,4])
    cap_frac : float — upper landmark fraction cap (default 0.2)
    central_ratio : float — central vs boundary (default 0.8)
    agg_mode : str — "topk" (paper) | "mean" | "min"
    topk : int — paper r in {3,5}, default 3
    alpha : float — LNC* weight (default 0.7, paper)
    weighted : bool — cluster-size weighted mean (default True)

    Returns
    -------
    dict with keys: lsil, lnc_star, passed, n_landmarks, timing_s, ...
    """
    t0 = time.time()
    X = np.asarray(X, dtype=object)
    labels = np.asarray(labels)
    n, p = X.shape
    K = len(np.unique(labels))

    is_cat = _detect_feature_types(X, cat_features)
    ranges = _gower_ranges(X, is_cat)

    # Adaptive landmark count (√n law — Theorem 1)
    n_lm = adaptive_landmark_count(n, K=K, c=c, cap_frac=cap_frac)

    # Select landmarks
    t_lm = time.time()
    lm_idx, lm_labels, lm_info = select_landmarks(
        X, labels, n_landmarks=n_lm, central_ratio=central_ratio,
        random_state=random_state, strategy=strategy,
        _is_cat=is_cat, _ranges=ranges)
    t_lm = time.time() - t_lm

    # Distance cache (dominant cost: O(n * |L| * p))
    t_dist = time.time()
    D = gower_distances_to_landmarks(X, X[lm_idx], is_cat, ranges)
    t_dist = time.time() - t_dist

    # L-Sil (paper Eq. 2-3)
    t_ls = time.time()
    lsil_val, per_point = compute_lsil(D, labels, lm_labels,
                                        agg_mode=agg_mode, topk=topk, weighted=weighted)
    t_ls = time.time() - t_ls

    # LNC* (paper Eq. 6)
    t_lnc = time.time()
    lnc_val, per_lm = compute_lnc_star(D, labels, lm_labels, lm_idx,
                                         k_neighbors=k_neighbors, alpha=alpha,
                                         weighted=weighted)
    t_lnc = time.time() - t_lnc

    total = time.time() - t0

    result = {
        "lsil": float(lsil_val), "lnc_star": float(lnc_val),
        "passed": bool(lnc_val >= lnc_threshold),
        "lnc_threshold": lnc_threshold,
        "n_landmarks": len(lm_idx), "n_samples": n, "n_features": p,
        "n_cat_features": int(is_cat.sum()), "n_num_features": int((~is_cat).sum()),
        "strategy": strategy, "c": c, "cap_frac": cap_frac,
        "central_ratio": central_ratio, "agg_mode": agg_mode, "topk": topk,
        "alpha": alpha, "weighted": weighted,
        "timing_s": {"landmark_selection": round(t_lm, 4),
                     "distance_cache": round(t_dist, 4),
                     "lsil_computation": round(t_ls, 4),
                     "lnc_computation": round(t_lnc, 4),
                     "total": round(total, 4)},
        "landmark_info": lm_info,
    }
    if return_per_point:
        result["per_point_scores"] = per_point
        result["per_landmark_lnc"] = per_lm
    return result


def gower_silhouette(X, labels, cat_features=None) -> float:
    """Full Gower Silhouette — O(n²). For validation on small data only."""
    X = np.asarray(X, dtype=object)
    labels = np.asarray(labels)
    is_cat = _detect_feature_types(X, cat_features)
    ranges = _gower_ranges(X, is_cat)
    D = gower_pairwise(X, is_cat, ranges)
    n = X.shape[0]
    unique_labels = np.unique(labels)
    scores = np.zeros(n)
    for i in range(n):
        c = labels[i]
        own = labels == c; own[i] = False
        if own.sum() == 0: continue
        a = D[i, own].mean()
        b = np.inf
        for k in unique_labels:
            if k == c: continue
            other = labels == k
            if other.sum() == 0: continue
            b = min(b, D[i, other].mean())
        denom = max(a, b)
        scores[i] = (b - a) / denom if denom > 0 else 0.0
    return float(scores.mean())
