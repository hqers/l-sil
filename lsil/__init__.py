"""
lsil — Landmark-Based Evaluation Metrics for Mixed-Type Clustering
==================================================================

Paper-aligned implementation. Defaults: c=3 (√n law), topk=3, alpha=0.7,
cluster-size weighted mean.

Quick Start:
    from lsil import lsil_score
    score = lsil_score(X, labels, cat_features=[0, 3, 7])

Full Evaluation:
    from lsil import lsil_evaluate
    result = lsil_evaluate(X, labels, cat_features=[0, 3])
    print(result['lsil'], result['lnc_star'], result['passed'])

Reference:
    Pratama, Lubis, Sembiring (2026). L-Sil: Evaluating Cluster Quality
    in Mixed-Type Data via Landmark-Based Silhouette and Neighborhood
    Consistency.

GitHub: https://github.com/hqers/l-sil
"""
from .api import (lsil_score, lnc_star_score, lsil_evaluate,
                  gower_silhouette, adaptive_landmark_count)

__version__ = "0.2.0"
__all__ = ["lsil_score", "lnc_star_score", "lsil_evaluate",
           "gower_silhouette", "adaptive_landmark_count"]
