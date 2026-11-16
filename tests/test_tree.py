"""TreeExplainer: patched shap vs real shap, same process, same inputs (see conftest.assert_same)."""
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)

import shap
from conftest import assert_same

rs = np.random.RandomState(1)
X = rs.randn(300, 7)
X[:, 6] = np.round(X[:, 6] * 2)  # an integer-valued column
y = X[:, 0] + X[:, 1] * X[:, 2] + 0.2 * rs.randn(300)
yc = (y > 0).astype(int)
y3 = np.digitize(y, [-0.7, 0.7])
BG, XE = X[:50], X[200:240]
RF = RandomForestRegressor(30, max_depth=7, random_state=0).fit(X, y)
RFC3 = RandomForestClassifier(20, max_depth=6, random_state=0).fit(X, y3)
GBC = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=0).fit(X, yc)
EXACT = dict(rtol=1e-12, atol=1e-12)
# interventional fast paths sum shap's float32 leaf terms in another order (README / docs/USAGE.md): ~1e-8
INTERV = dict(rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("model", [RF, RFC3, GBC], ids=["rf", "rf3", "gbc"])
def test_path_dependent(model):
    assert_same(lambda: (shap.TreeExplainer(model).shap_values(XE), shap.TreeExplainer(model)(XE),
                         shap.TreeExplainer(model).expected_value), **EXACT)


@pytest.mark.parametrize("model", [RF, RFC3], ids=["rf", "rf3"])
def test_interactions_and_saabas(model):
    assert_same(lambda: (shap.TreeExplainer(model).shap_interaction_values(XE[:10]),
                         shap.TreeExplainer(model).shap_values(XE, approximate=True)), **EXACT)


@pytest.mark.parametrize("output", ["raw", "probability", "log_loss"])
def test_interventional(output):
    e = lambda: shap.TreeExplainer(GBC, BG, feature_perturbation="interventional", model_output=output)
    assert_same(lambda: (e().shap_values(XE, y=yc[200:240] if output == "log_loss" else None),
                         e().expected_value if output != "log_loss" else None), **INTERV)


def test_interventional_exact_mode():
    os.environ["SHAP_RS_NO_FAST"] = "1"
    try:
        for output in ("raw", "probability"):
            assert_same(lambda: shap.TreeExplainer(GBC, BG, feature_perturbation="interventional",
                                                   model_output=output).shap_values(XE), **EXACT)
    finally:
        del os.environ["SHAP_RS_NO_FAST"]


def test_dataframe_explainer_and_masker_sampling():
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(7)])
    # > 100 background rows: shap subsamples with np.random (same draws patched)
    assert_same(lambda: shap.Explainer(RF, df.iloc[:150])(df.iloc[200:230]), **INTERV)
    assert_same(lambda: shap.TreeExplainer(RF)(df.iloc[200:230]), **EXACT)


def test_nan_hist_gb_and_isolation_forest():
    Xn = X.copy()
    Xn[rs.rand(*X.shape) < 0.1] = np.nan
    hgb = HistGradientBoostingRegressor(max_iter=30, random_state=0).fit(Xn, y)
    iso = IsolationForest(n_estimators=20, random_state=0).fit(X)
    assert_same(lambda: shap.TreeExplainer(hgb).shap_values(Xn[200:240]), **EXACT)
    assert_same(lambda: shap.TreeExplainer(iso).shap_values(XE), **EXACT)
    # shap's own additivity failure (HistGB + NaN + interventional) is reproduced
    assert_same(lambda: shap.TreeExplainer(hgb, Xn[:40], feature_perturbation="interventional")
                .shap_values(Xn[200:240]), **INTERV)


def test_errors():
    assert_same(lambda: shap.TreeExplainer(GBC, model_output="probability").shap_values(XE))
    assert_same(lambda: shap.TreeExplainer(GBC, BG, model_output="log_loss").shap_values(XE))


def test_xgboost_lightgbm():
    xgboost = pytest.importorskip("xgboost")
    lightgbm = pytest.importorskip("lightgbm")
    xgb = xgboost.XGBClassifier(n_estimators=40, max_depth=4, random_state=0).fit(X, yc)
    lgb = lightgbm.LGBMRegressor(n_estimators=40, num_leaves=15, random_state=0, verbose=-1).fit(X, y)
    for m in (xgb, lgb):
        assert_same(lambda: (shap.TreeExplainer(m).shap_values(XE), shap.TreeExplainer(m).shap_interaction_values(XE[:5])),
                    **EXACT)
        assert_same(lambda: shap.TreeExplainer(m, BG, feature_perturbation="interventional").shap_values(XE), **INTERV)
    assert_same(lambda: shap.TreeExplainer(xgb, BG, feature_perturbation="interventional",
                                           model_output="probability").shap_values(XE), **INTERV)


def test_catboost():
    catboost = pytest.importorskip("catboost")
    kw = dict(iterations=40, depth=4, verbose=0, random_seed=0, allow_writing_files=False)
    reg = catboost.CatBoostRegressor(**kw).fit(X, y)
    clf = catboost.CatBoostClassifier(**kw).fit(X, yc)
    for m in (reg, clf):
        assert_same(lambda: shap.TreeExplainer(m).shap_values(XE), **EXACT)  # catboost's own ShapValues, as in shap
        assert_same(lambda: shap.TreeExplainer(m, BG, feature_perturbation="interventional").shap_values(XE), **INTERV)
    assert_same(lambda: shap.TreeExplainer(clf, BG, feature_perturbation="interventional",
                                           model_output="probability").shap_values(XE), **INTERV)
    # categorical features: shap cannot parse the trees for interventional; same failure patched
    df = pd.DataFrame({"a": X[:, 0], "c": np.where(X[:, 1] > 0, "u", "v")})
    cat = catboost.CatBoostRegressor(**kw, cat_features=["c"]).fit(df, y)
    assert_same(lambda: shap.TreeExplainer(cat).shap_values(df.iloc[:10]), **EXACT)
    assert_same(lambda: shap.TreeExplainer(cat, df.iloc[:20], feature_perturbation="interventional")
                .shap_values(df.iloc[:10]), **INTERV)


def test_too_few_columns_raises():
    # Documented difference: shap 0.54.1 does not check the width of X and its C++ reads past each row (undefined
    # output); patched shap raises a ValueError instead of reading out of bounds.
    import shap_rs
    assert shap_rs.patch()
    try:
        with pytest.raises(ValueError, match="uses feature index"):
            shap.TreeExplainer(RF).shap_values(XE[:, :3])
        with pytest.raises(ValueError, match="uses feature index"):
            shap.TreeExplainer(RF, BG[:, :3], feature_perturbation="interventional").shap_values(XE[:, :3])
    finally:
        shap_rs.unpatch()
    with pytest.raises(ValueError, match="uses feature index"):
        shap_rs.TreeExplainer(RF).shap_values(XE[:, :3])
