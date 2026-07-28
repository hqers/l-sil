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

__version__ = "1.0.0"


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
               topk=3, weighted=True, lnc_threshold=0.5,
               pool_factor=5, pool_frac_cap=0.15) -> float:
    """
    Compute L-Sil score — drop-in replacement for silhouette_score.

    Defaults follow paper: c=3, topk=3, weighted=True.
    Landmark count: |L| = c*sqrt(n), capped at cap_frac*n.
    """
    result = lsil_evaluate(X, labels, cat_features=cat_features, c=c,
                           cap_frac=cap_frac, central_ratio=central_ratio,
                           random_state=random_state, agg_mode=agg_mode,
                           topk=topk, weighted=weighted, lnc_threshold=lnc_threshold,
                           pool_factor=pool_factor, pool_frac_cap=pool_frac_cap)
    return result["lsil"]


def lnc_star_score(X, labels, cat_features=None, c=3.0, cap_frac=0.2,
                   central_ratio=0.8, random_state=42, k_neighbors=10,
                   alpha=0.7, weighted=True,
                   pool_factor=5, pool_frac_cap=0.15) -> float:
    """Compute LNC* score (paper Eq. 6, default alpha=0.7)."""
    result = lsil_evaluate(X, labels, cat_features=cat_features, c=c,
                           cap_frac=cap_frac, central_ratio=central_ratio,
                           random_state=random_state, k_neighbors=k_neighbors,
                           alpha=alpha, weighted=weighted,
                           pool_factor=pool_factor, pool_frac_cap=pool_frac_cap)
    return result["lnc_star"]


def lsil_evaluate(X, labels, cat_features=None, c=3.0, cap_frac=0.2,
                  central_ratio=0.8, random_state=42, lnc_threshold=0.5,
                  k_neighbors=10, alpha=0.7, agg_mode="topk", topk=3,
                  weighted=True, strategy="cluster_aware",
                  pool_factor=5, pool_frac_cap=0.15,
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
    pool_factor : int — candidate pool multiplier for farthest-first (default 5).
        pool_factor <= 0 reproduces the pre-patch degenerate pool (no
        diversification, random_state inert); retained for ablation only.
    pool_frac_cap : float — upper bound on candidate pool as a fraction of
        cluster size (default 0.15). Bounds pool coverage from above so that
        selection behaviour stays comparable across dataset scales. Not to be
        confused with cap_frac, which bounds the landmark COUNT.

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
        pool_factor=pool_factor, pool_frac_cap=pool_frac_cap,
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
        "pool_factor": pool_factor, "pool_frac_cap": pool_frac_cap,
        "random_state": random_state,
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


def gower_silhouette(X, labels, cat_features=None, block: int = 512) -> float:
    """
    Exact full Gower Silhouette, computed in row blocks.

    Numerically identical to the previous implementation (which materialised the
    full n x n matrix) but requires O(block * n) memory instead of O(n^2), so it
    is tractable as a reference on datasets of tens of thousands of rows.
    Categorical columns are mapped to integer codes first: on object dtype
    the `!=` comparison falls back to per-element Python and dominates runtime.

    Time complexity remains O(n^2 * p).
    """
    import gc

    X = np.asarray(X, dtype=object)
    labels = np.asarray(labels)
    n, p = X.shape

    is_cat = _detect_feature_types(X, cat_features)
    ranges = _gower_ranges(X, is_cat)
    num_idx = np.where(~is_cat)[0]
    cat_idx = np.where(is_cat)[0]

    Xn = X[:, num_idx].astype(np.float64) if len(num_idx) else np.zeros((n, 0))
    Rn = np.maximum(ranges[num_idx].astype(np.float64), 1e-12) if len(num_idx) else np.zeros(0)
    if len(cat_idx):
        Xc = np.empty((n, len(cat_idx)), dtype=np.int32)
        for j, k in enumerate(cat_idx):
            # np.unique(..., return_inverse=True) gives codes in sorted order,
            # equivalent to pd.factorize(sort=True) but numpy-only (keeps the
            # package dependency-free beyond numpy).
            Xc[:, j] = np.unique(X[:, k].astype(str), return_inverse=True)[1]
    else:
        Xc = np.zeros((n, 0), dtype=np.int32)

    uniq = np.unique(labels)
    K = len(uniq)
    lab_code = np.searchsorted(uniq, labels)
    counts = np.bincount(lab_code, minlength=K).astype(np.float64)

    S = np.zeros((n, K), dtype=np.float64)
    for s in range(0, n, block):
        e = min(s + block, n)
        D = np.zeros((e - s, n), dtype=np.float64)
        for k in range(Xn.shape[1]):
            D += np.abs(Xn[s:e, k][:, None] - Xn[None, :, k]) / Rn[k]
        for k in range(Xc.shape[1]):
            D += (Xc[s:e, k][:, None] != Xc[None, :, k])
        D /= p
        for cc in range(K):
            S[s:e, cc] = D[:, lab_code == cc].sum(axis=1)
        del D
    gc.collect()

    own = S[np.arange(n), lab_code]
    n_own = counts[lab_code]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(n_own > 1, own / np.maximum(n_own - 1, 1), 0.0)
        Sb = S / counts[None, :]
    Sb[np.arange(n), lab_code] = np.inf
    b = Sb.min(axis=1)

    denom = np.maximum(a, b)
    scores = np.where(denom > 0, (b - a) / denom, 0.0)
    scores[n_own <= 1] = 0.0
    return float(scores.mean())
