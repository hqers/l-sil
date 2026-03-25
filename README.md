# L-Sil: Landmark-Based Evaluation Metrics for Mixed-Type Clustering

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**L-Sil** is a drop-in replacement for `silhouette_score` that works natively on **mixed-type data** (numerical + categorical) with **O(n^{3/2})** complexity instead of O(n²).

## Why L-Sil?

| | Full Silhouette (Gower) | L-Sil |
|---|---|---|
| **Complexity** | O(n²·p) | O(c·p·n^{3/2}) — Theorem 1 |
| **Mixed-type** | Requires precomputed Gower matrix | Native support |
| **Scalability** | Impractical for n > 30k | Tested on n = 334k |
| **BCVD correction** | No | Yes |
| **Quality guardrail** | No | LNC* validation |
| **Landmark budget** | — | |L| = c√n (default c=3) |

## Installation

```bash
pip install lsil
```

Or from source:
```bash
git clone https://github.com/hqers/l-sil.git
cd l-sil
pip install -e .
```

## Quick Start

```python
from lsil import lsil_score

# Mixed-type data — just works
score = lsil_score(X, labels, cat_features=[0, 3, 7])
```

### Full evaluation with diagnostics

```python
from lsil import lsil_evaluate

result = lsil_evaluate(X, labels, cat_features=[0, 3, 7])
print(f"L-Sil  = {result['lsil']:.4f}")
print(f"LNC*   = {result['lnc_star']:.4f}  (passed: {result['passed']})")
print(f"|L|    = {result['n_landmarks']}  (c={result['c']}, √n law)")
print(f"Time   = {result['timing_s']['total']:.2f}s")
```

### Compare with full Gower Silhouette

```python
from lsil import lsil_score, gower_silhouette

fast = lsil_score(X, labels, cat_features=[0, 3])      # O(n^{3/2})
full = gower_silhouette(X, labels, cat_features=[0, 3]) # O(n²)
print(f"L-Sil={fast:.4f}, SS_G={full:.4f}, diff={abs(fast-full):.4f}")
```

## Paper-Aligned Defaults

| Parameter | Default | Paper reference |
|---|---|---|
| `c` (√n constant) | 3.0 | Theorem 1, c ∈ [1.5, 4] |
| `topk` (aggregation) | 3 | Practical settings, r ∈ {3, 5} |
| `alpha` (LNC* weight) | 0.7 | Eq. 6 |
| `weighted` | True | Cluster-size weighted mean |
| `central_ratio` | 0.8 | 80% central, 20% boundary |

## How It Works

1. **Adaptive Landmark Count**: |L| = c√n (Theorem 1 → O(n^{3/2}))
2. **Cluster-Aware Selection**: proportional allocation, central/boundary split, farthest-first diversity
3. **Distance Caching**: O(n·|L|·p) — computed once, reused for Auto-K and feature selection
4. **L-Sil**: top-r aggregation to same/other cluster landmarks (Eq. 2-3)
5. **LNC* Guardrail**: Miller-Madow entropy + IQR-normalized distance contrast (Eq. 6)

## Experiments

Run the complexity validation experiment:

```bash
python experiments/complexity_validation.py
```

This compares fraction-based (|L|=0.2n) vs √n-law (|L|=c√n) landmarks across dataset sizes, measuring runtime slope and approximation accuracy.

## Citation

```bibtex
@article{pratama2026lsil,
  title={L-Sil: Evaluating Cluster Quality in Mixed-Type Data via
         Landmark-Based Silhouette and Neighborhood Consistency},
  author={Pratama, Hasta and Lubis, Fetty Fitriyanti and Sembiring, Jaka},
  journal={Journal of Data Science and Analytics},
  year={2026}
}
```

## License

MIT License.
