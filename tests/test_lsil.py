"""Tests for lsil package — paper-aligned version."""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic():
    from lsil import lsil_score, gower_silhouette
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(50,3)+[0,0,0], rng.randn(50,3)+[5,5,5]]).astype(object)
    labels = np.array([0]*50+[1]*50)
    score = lsil_score(X, labels, cat_features=[])
    full = gower_silhouette(X, labels, cat_features=[])
    assert score > 0, f"Expected positive: {score}"
    print(f"  L-Sil={score:.4f}, SS_G={full:.4f}, diff={abs(score-full):.4f}")

def test_mixed():
    from lsil import lsil_score
    X = np.array([[1,'A'],[2,'A'],[1.5,'A'],[10,'B'],[11,'B'],[10.5,'B'],
                   [20,'C'],[21,'C'],[20.5,'C']], dtype=object)
    labels = np.array([0,0,0,1,1,1,2,2,2])
    score = lsil_score(X, labels, cat_features=[1])
    assert score > 0
    print(f"  Mixed: L-Sil={score:.4f}")

def test_full_evaluate():
    from lsil import lsil_evaluate
    rng = np.random.RandomState(123)
    X = np.vstack([rng.randn(30,4), rng.randn(30,4)+4]).astype(object)
    labels = np.array([0]*30+[1]*30)
    r = lsil_evaluate(X, labels, cat_features=[], return_per_point=True)
    assert r["agg_mode"] == "topk" and r["topk"] == 3, "Default should be topk=3"
    assert r["alpha"] == 0.7, "Default alpha should be 0.7"
    assert r["weighted"] == True, "Default should be weighted"
    print(f"  L-Sil={r['lsil']:.4f}, LNC*={r['lnc_star']:.4f}, |L|={r['n_landmarks']}, c={r['c']}")

def test_sqrt_n_law():
    from lsil import adaptive_landmark_count
    assert adaptive_landmark_count(10000, K=3, c=3.0) == 300  # 3*sqrt(10000)=300
    assert adaptive_landmark_count(100, K=3, c=3.0, cap_frac=1.0) == 30  # uncapped: 3*sqrt(100)=30
    assert adaptive_landmark_count(100, K=3, c=3.0, cap_frac=0.2) == 20  # capped: 0.2*100=20
    print(f"  n=10000→|L|=300, n=100(uncapped)→|L|=30, n=100(cap0.2)→|L|=20")

def test_complexity_scaling():
    from lsil import lsil_evaluate
    times = {}
    for n in [500, 1000, 2000, 4000]:
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(n//3,5), rng.randn(n//3,5)+3, rng.randn(n-2*(n//3),5)+6]).astype(object)
        labels = np.array([0]*(n//3)+[1]*(n//3)+[2]*(n-2*(n//3)))
        t0 = time.time()
        r = lsil_evaluate(X, labels, cat_features=[])
        times[n] = time.time()-t0
    # Check sub-quadratic scaling
    ratio = times[4000]/times[1000]
    expected_quad = (4000/1000)**2  # 16x for quadratic
    expected_32 = (4000/1000)**1.5  # 8x for n^1.5
    print(f"  Runtime ratio 4k/1k = {ratio:.1f}x (expect ~{expected_32:.0f}x for n^1.5, ~{expected_quad:.0f}x for n^2)")
    assert ratio < expected_quad * 0.8, f"Scaling appears quadratic: {ratio:.1f}x"

def test_modes():
    from lsil import lsil_evaluate
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(40,3), rng.randn(40,3)+5]).astype(object)
    labels = np.array([0]*40+[1]*40)
    r_topk = lsil_evaluate(X, labels, cat_features=[], agg_mode="topk", topk=3)
    r_mean = lsil_evaluate(X, labels, cat_features=[], agg_mode="mean")
    r_min = lsil_evaluate(X, labels, cat_features=[], agg_mode="min")
    print(f"  topk3={r_topk['lsil']:.4f}, mean={r_mean['lsil']:.4f}, min={r_min['lsil']:.4f}")

if __name__ == "__main__":
    tests = [("Basic numerical", test_basic), ("Mixed-type", test_mixed),
             ("Full evaluate (defaults)", test_full_evaluate),
             ("√n landmark law", test_sqrt_n_law),
             ("Complexity scaling", test_complexity_scaling),
             ("Aggregation modes", test_modes)]
    passed = 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn(); print("  PASSED"); passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
    print(f"\n{'='*50}\nResults: {passed}/{len(tests)} passed\n{'='*50}")
