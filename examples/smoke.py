"""Wheel smoke test: patched shap must equal unpatched shap on small tree + kernel cases. Exit 1 on mismatch."""
import sys
import numpy as np
import shap
import shap_rs
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

Xr, yr = make_regression(300, 8, random_state=0)
rf = RandomForestRegressor(30, max_depth=6, random_state=0).fit(Xr, yr)
Xc, yc = make_classification(300, 12, random_state=0)
lr = LogisticRegression(max_iter=500).fit(Xc, yc)


def run():
    t = shap.TreeExplainer(rf).shap_values(Xr[:50])
    ti = shap.TreeExplainer(rf, Xr[:40], feature_perturbation="interventional").shap_values(Xr[:20])
    np.random.seed(0)
    k = shap.KernelExplainer(lr.predict_proba, Xc[:20]).shap_values(Xc[:5], nsamples=200, l1_reg="aic")
    np.random.seed(0)
    k2 = shap.KernelExplainer(lr.predict_proba, Xc[:20]).shap_values(Xc[:3], nsamples=300, l1_reg="num_features(4)")
    return [t, ti, np.asarray(k), np.asarray(k2)]


ref = run()
assert shap_rs.patch(), "patch() refused"
got = run()
shap_rs.unpatch()
bad = [i for i, (a, b) in enumerate(zip(ref, got)) if not np.allclose(a, b, rtol=1e-6, atol=1e-9)]
print("shap", shap.__version__, "shap_rs", shap_rs.__file__, "native_l1", shap_rs._kernel._native_l1(), "mismatch", bad)
sys.exit(1 if bad else 0)
