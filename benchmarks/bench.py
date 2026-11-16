"""Speed: real shap 0.54.1 vs shap_rs on realistic workloads (bigger than the answer key).
Each case also checks agreement (max abs diff), so a fast-but-wrong result is visible.
Usage: python bench.py [--only SUBSTR]   (prints one JSON line per case, writes bench.json)
Single-core shap_rs: RAYON_NUM_THREADS=1 python bench.py
"""
import argparse, json, time, warnings
import numpy as np
import shap, shap_rs, xgboost, lightgbm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path

H = Path(__file__).resolve().parent


def data():
    from sklearn.datasets import fetch_california_housing
    Xa, ya = shap.datasets.adult()
    Xa = Xa.values.astype(np.float64); ya = ya.astype(int)
    c = fetch_california_housing(); Xc, yc = c.data.astype(np.float64), c.target
    return {"adult": (Xa, ya), "cal": (Xc, yc)}


warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--only"); a = ap.parse_args()
D = data(); Xa, ya = D["adult"]; Xc, yc = D["cal"]
rng = np.random.default_rng(0); bga = Xa[rng.choice(len(Xa), 100, replace=False)]
print("training...", flush=True)
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=0, n_jobs=-1).fit(Xc, yc)
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, n_jobs=-1).fit(Xa, ya)
xgb = xgboost.XGBClassifier(n_estimators=500, max_depth=6, random_state=0).fit(Xa, ya)
lgb = lightgbm.LGBMClassifier(n_estimators=500, num_leaves=63, random_state=0, verbose=-1).fit(Xa, ya)
lr = LogisticRegression(max_iter=2000).fit(Xa, ya); f = lambda Z: lr.predict_proba(Z)[:, 1]

CASES = [
    ("rf200_d12 path_dependent 500 rows", lambda L: L.TreeExplainer(rf).shap_values(Xc[:500]), True),
    ("rf100_clf path_dependent 2000 rows", lambda L: L.TreeExplainer(rfc).shap_values(Xa[:2000]), True),
    ("xgb500 interventional bg100 1000 rows", lambda L: L.TreeExplainer(xgb, bga, feature_perturbation="interventional").shap_values(Xa[:1000]), True),
    ("lgbm500 interventional bg100 500 rows", lambda L: L.TreeExplainer(lgb, bga, feature_perturbation="interventional").shap_values(Xa[:500]), True),
    ("rf100_clf interactions 200 rows", lambda L: L.TreeExplainer(rfc).shap_interaction_values(Xa[:200]), True),
    ("kernel logreg bg50 100 rows (sampled)", lambda L: L.KernelExplainer(f, shap.kmeans(Xa, 50)).shap_values(Xa[:100], silent=True), False),
]
out = []
for name, fn, cmp in CASES:
    if a.only and a.only not in name: continue
    t = time.perf_counter(); vs = fn(shap); ts = time.perf_counter() - t
    t = time.perf_counter(); vr = fn(shap_rs); tr = time.perf_counter() - t
    st = lambda v: np.stack(v) if isinstance(v, list) else np.asarray(v)
    diff = float(np.abs(st(vs) - st(vr)).max()) if cmp else None
    row = {"case": name, "shap_s": round(ts, 3), "shap_rs_s": round(tr, 3), "speedup": round(ts / tr, 1), "max_abs_diff": diff}
    out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open(H / "bench.json", "w"), indent=1)
