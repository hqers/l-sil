"""
Experiment: L-Sil Complexity Validation
=======================================
Compares |L|=0.2n (fraction) vs |L|=c√n (paper Theorem 1).
Measures runtime slope, accuracy vs full SS_G, and LNC*.

Run: python experiments/complexity_validation.py
Output: table + optional CSV for paper.
"""
import numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsil import lsil_evaluate, gower_silhouette, adaptive_landmark_count


def make_mixed(n, n_num=5, n_cat=3, K=3, seed=42):
    rng = np.random.RandomState(seed)
    X = np.empty((n, n_num+n_cat), dtype=object)
    labels = np.zeros(n, dtype=int)
    cats = [['A','B','C','D'],['X','Y','Z'],['lo','mid','hi']]
    sizes = np.diff(np.concatenate([[0], np.sort(rng.choice(range(1,n), K-1, replace=False)), [n]]))
    idx = 0
    for k in range(K):
        center = rng.uniform(0,10,n_num) + k*5
        for _ in range(sizes[k]):
            for j in range(n_num): X[idx,j] = center[j]+rng.normal(0,1.5)
            for j in range(n_cat):
                cs = cats[j%len(cats)]; w = np.ones(len(cs)); w[k%len(cs)]=3; w/=w.sum()
                X[idx,n_num+j] = rng.choice(cs, p=w)
            labels[idx] = k; idx += 1
    return X, labels, list(range(n_num, n_num+n_cat))


def run():
    datasets = [(1000,8,3,3),(1500,8,3,3),(2000,7,3,5),(5000,8,4,3),(10000,8,4,3),(20000,8,4,3)]
    strategies = [
        ("frac_0.20", dict(c=999, cap_frac=0.20)),
        ("frac_0.10", dict(c=999, cap_frac=0.10)),
        ("frac_0.05", dict(c=999, cap_frac=0.05)),
        ("sqrt_c2",   dict(c=2.0, cap_frac=1.0)),
        ("sqrt_c3",   dict(c=3.0, cap_frac=1.0)),
        ("sqrt_c4",   dict(c=4.0, cap_frac=1.0)),
    ]

    rows = []
    for n, n_num, n_cat, K in datasets:
        X, labels, cat_f = make_mixed(n, n_num, n_cat, K)
        ss_g = gower_silhouette(X[:min(n,2000)], labels[:min(n,2000)], cat_f) if n<=5000 else None

        print(f"\nn={n:,}  (K={K})" + (f"  SS_G={ss_g:.4f}" if ss_g else ""))
        print(f"{'Strategy':<14} | {'|L|':>6} | {'L-Sil':>7} | {'LNC*':>6} | {'Time':>7}")
        print("-"*55)

        for name, kw in strategies:
            nlm = adaptive_landmark_count(n, K, c=kw['c'], cap_frac=kw['cap_frac'])
            if n * nlm > 50_000_000: print(f"{name:<14} | {'SKIP':>6}"); continue
            try:
                r = lsil_evaluate(X, labels, cat_features=cat_f, **kw)
                dev = abs(r['lsil']-ss_g) if ss_g else None
                print(f"{name:<14} | {r['n_landmarks']:>6} | {r['lsil']:>7.4f} | {r['lnc_star']:>6.3f} | {r['timing_s']['total']:>6.2f}s")
                rows.append(dict(n=n, strategy=name, n_lm=r['n_landmarks'], lsil=r['lsil'],
                                 lnc=r['lnc_star'], time=r['timing_s']['total'], ss_g=ss_g, dev=dev))
            except Exception as e: print(f"{name:<14} | ERROR: {e}")

    # Scaling analysis
    print(f"\n{'='*55}\nRuntime Scaling (log-log slope)\n{'='*55}")
    for sname in ["sqrt_c2","sqrt_c3","frac_0.20","frac_0.05"]:
        sr = [r for r in rows if r['strategy']==sname and r['time']>0.01]
        if len(sr)<3: continue
        ns = np.array([r['n'] for r in sr], dtype=float)
        ts = np.array([r['time'] for r in sr], dtype=float)
        A = np.vstack([np.log(ns), np.ones_like(ns)]).T
        alpha, _ = np.linalg.lstsq(A, np.log(ts), rcond=None)[0]
        print(f"  {sname:<14}: slope = {alpha:.3f}  ({'~n^1.5 OK' if 1.3<alpha<1.7 else '~n^2' if alpha>1.8 else 'sub-1.5'})")

    # Save CSV if desired
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv("complexity_results.csv", index=False)
        print("\nSaved: complexity_results.csv")
    except ImportError:
        pass

if __name__ == "__main__":
    run()
