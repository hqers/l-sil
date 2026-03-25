"""
Gower distance for mixed-type data.

Handles numerical, categorical, and ordinal features natively
without requiring encoding or transformation.
"""

import numpy as np
from typing import Optional, List


def _detect_feature_types(
    X: np.ndarray,
    cat_features: Optional[List[int]] = None,
) -> np.ndarray:
    """Return boolean mask: True = categorical, False = numerical."""
    n_features = X.shape[1]
    is_cat = np.zeros(n_features, dtype=bool)

    if cat_features is not None:
        for idx in cat_features:
            is_cat[idx] = True
        return is_cat

    # Auto-detect: object/string dtype or low cardinality
    for j in range(n_features):
        col = X[:, j]
        try:
            col_f = col.astype(float)
            # If very few unique values relative to n, treat as categorical
            if len(np.unique(col_f[~np.isnan(col_f)])) <= 2:
                is_cat[j] = True
        except (ValueError, TypeError):
            is_cat[j] = True

    return is_cat


def _gower_ranges(
    X: np.ndarray, is_cat: np.ndarray
) -> np.ndarray:
    """Compute per-feature ranges for numerical features."""
    n_features = X.shape[1]
    ranges = np.ones(n_features, dtype=float)

    for j in range(n_features):
        if not is_cat[j]:
            col = X[:, j].astype(float)
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                r = valid.max() - valid.min()
                ranges[j] = r if r > 0 else 1.0

    return ranges


def gower_distances_to_landmarks(
    X: np.ndarray,
    landmarks: np.ndarray,
    is_cat: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    """
    Compute Gower distances from all points in X to each landmark.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Full dataset.
    landmarks : array-like of shape (m, p)
        Landmark points.
    is_cat : array of shape (p,)
        Boolean mask for categorical features.
    ranges : array of shape (p,)
        Range of each numerical feature.

    Returns
    -------
    D : array of shape (n, m)
        Distance matrix where D[i, j] = gower(X[i], landmarks[j]).
    """
    n = X.shape[0]
    m = landmarks.shape[0]
    p = X.shape[1]

    num_idx = np.where(~is_cat)[0]
    cat_idx = np.where(is_cat)[0]

    D = np.zeros((n, m), dtype=np.float64)

    # Vectorized numerical contribution
    if len(num_idx) > 0:
        X_num = X[:, num_idx].astype(np.float64)
        L_num = landmarks[:, num_idx].astype(np.float64)
        R_num = ranges[num_idx]

        # |x_j - l_j| / range_j, summed over numerical features
        for k in range(len(num_idx)):
            diff = np.abs(X_num[:, k:k+1] - L_num[:, k:k+1].T) / R_num[k]
            D += diff

    # Vectorized categorical contribution (mismatch = 1, match = 0)
    if len(cat_idx) > 0:
        for k in cat_idx:
            x_col = X[:, k].reshape(-1, 1)     # (n, 1)
            l_col = landmarks[:, k].reshape(1, -1)  # (1, m)
            D += (x_col != l_col).astype(np.float64)

    # Normalize by number of features
    D /= p

    return D


def gower_pairwise(
    X: np.ndarray,
    is_cat: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    """
    Full pairwise Gower distance matrix. O(n²p) — use only for small n.

    Returns
    -------
    D : array of shape (n, n)
    """
    return gower_distances_to_landmarks(X, X, is_cat, ranges)
