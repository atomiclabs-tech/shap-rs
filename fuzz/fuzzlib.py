"""Fuzz answer key for the shap_rs patch layer. Shared by gen.py and check.py.

Every case is fully determined by its seed: random data, random model, random explainer call.
gen.py runs the call with REAL shap 0.54.1 and pickles the model, inputs and shap's output.
check.py replays the identical call with shap_rs.patch() active and compares.
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

H = Path(__file__).resolve().parent
CASES = Path(os.environ["SHAP_FUZZ_CASES"]).resolve() if os.environ.get("SHAP_FUZZ_CASES") else H / "cases"  # per-env answer keys (version matrix)

TREE_FAMILIES = ["dt_reg", "dt_clf", "rf_reg", "rf_clf", "et_reg", "et_clf", "gbr", "gbc", "hgb_reg", "hgb_clf",
                 "iso", "xgb_reg", "xgb_clf", "lgbm_reg", "lgbm_clf"]
NAN_OK = {"hgb_reg", "hgb_clf", "xgb_reg", "xgb_clf", "lgbm_reg", "lgbm_clf"}
KERNEL_FAMILIES = ["ridge", "logreg_p1", "logreg_proba", "mlp_reg", "knn_clf"]


def _data(r, n_feat, n_rows, n_classes, nan_frac, int_frac, reg):
    from sklearn.datasets import make_classification, make_regression
    if reg:
        X, y = make_regression(n_samples=n_rows, n_features=n_feat, n_informative=max(1, n_feat // 2),
                               noise=r.uniform(0, 20), random_state=int(r.integers(1 << 30)))
        y = y * (10.0 ** r.uniform(-2, 3))  # vary target scale
    else:
        X, y = make_classification(n_samples=n_rows, n_features=n_feat, n_informative=max(2, min(n_feat, n_feat // 2 + 1)),
                                   n_redundant=0, n_repeated=0, n_classes=n_classes,
                                   n_clusters_per_class=1, random_state=int(r.integers(1 << 30)))
    for j in range(n_feat):  # some integer-valued (categorical-like) columns
        if r.random() < int_frac:
            X[:, j] = np.floor(np.abs(X[:, j]) * 3)
    if nan_frac > 0:
        X[r.random(X.shape) < nan_frac] = np.nan
    return X, y


def make_case(seed):
    """-> spec dict, model, X_bg (or None), X_explain. Deterministic in seed."""
    import sklearn.ensemble as E, sklearn.tree as T
    r = np.random.default_rng(seed)
    kind = "kernel" if r.random() < 0.2 else "tree"
    spec = {"seed": seed, "kind": kind}
    if kind == "tree":
        fam = TREE_FAMILIES[r.integers(len(TREE_FAMILIES))]
        reg = fam.endswith("_reg") or fam in ("gbr", "iso")
        n_classes = 2 if reg or r.random() < 0.6 else int(r.integers(3, 6))
        if fam == "gbc" and n_classes > 2 and r.random() < 0.5:
            n_classes = 2
        n_feat = int(r.choice([2, 3, 5, 8, 12, 20, 40, 64]))
        n_rows = int(r.integers(200, 2000))
        nan_frac = float(r.choice([0, 0, 0.05, 0.3])) if fam in NAN_OK else 0.0
        X, y = _data(r, n_feat, n_rows, n_classes, nan_frac, r.choice([0, 0.3]), reg)
        depth = int(r.choice([1, 2, 3, 6, 10, 16])); ntree = int(r.choice([1, 3, 10, 40]))
        rs = int(r.integers(1000))
        mk = {
            "dt_reg": lambda: T.DecisionTreeRegressor(max_depth=depth, random_state=rs),
            "dt_clf": lambda: T.DecisionTreeClassifier(max_depth=depth, random_state=rs),
            "rf_reg": lambda: E.RandomForestRegressor(n_estimators=ntree, max_depth=depth, random_state=rs, n_jobs=-1),
            "rf_clf": lambda: E.RandomForestClassifier(n_estimators=ntree, max_depth=depth, random_state=rs, n_jobs=-1),
            "et_reg": lambda: E.ExtraTreesRegressor(n_estimators=ntree, max_depth=depth, random_state=rs, n_jobs=-1),
            "et_clf": lambda: E.ExtraTreesClassifier(n_estimators=ntree, max_depth=depth, random_state=rs, n_jobs=-1),
            "gbr": lambda: E.GradientBoostingRegressor(n_estimators=ntree, max_depth=min(depth, 8), random_state=rs),
            "gbc": lambda: E.GradientBoostingClassifier(n_estimators=ntree, max_depth=min(depth, 6), random_state=rs),
            "hgb_reg": lambda: E.HistGradientBoostingRegressor(max_iter=ntree, max_depth=depth, random_state=rs),
            "hgb_clf": lambda: E.HistGradientBoostingClassifier(max_iter=ntree, max_depth=depth, random_state=rs),
            "iso": lambda: E.IsolationForest(n_estimators=ntree, random_state=rs),
        }
        if fam.startswith("xgb"):
            import xgboost
            kw = dict(n_estimators=ntree, max_depth=min(depth, 10), random_state=rs, n_jobs=4)
            model = (xgboost.XGBRegressor(**kw) if reg else xgboost.XGBClassifier(**kw))
        elif fam.startswith("lgbm"):
            import lightgbm
            kw = dict(n_estimators=ntree, num_leaves=int(r.choice([2, 4, 15, 63])), random_state=rs, verbose=-1, n_jobs=4)
            model = (lightgbm.LGBMRegressor(**kw) if reg else lightgbm.LGBMClassifier(**kw))
        else:
            model = mk[fam]()
        model.fit(X) if fam == "iso" else model.fit(X, y)
        pandas_in = bool(r.random() < 0.3)
        cols = [f"f{j}" for j in range(n_feat)]
        Xe = X[r.choice(len(X), int(r.integers(1, 40)), replace=False)]
        mode = r.choice(["path_dependent", "path_dependent", "interventional", "interventional", "interactions", "approximate"])
        if mode == "interactions" and n_feat > 20:
            mode = "path_dependent"
        Xbg = X[r.choice(len(X), int(r.integers(1, 60)), replace=False)] if mode == "interventional" else None
        model_output = "raw"
        if mode == "interventional" and not reg and r.random() < 0.4:
            model_output = str(r.choice(["probability", "log_loss"]))
        api = str(r.choice(["shap_values", "call", "Explainer"])) if mode != "interactions" else str(r.choice(["interaction_values", "call_interactions"]))
        spec.update(family=fam, mode=str(mode), model_output=model_output, api=api, pandas=pandas_in, cols=cols,
                    n_feat=n_feat, n_classes=n_classes, depth=depth, ntree=ntree, nan_frac=nan_frac,
                    check_additivity=bool(r.random() < 0.8))
        if model_output == "log_loss":
            spec["y_explain"] = y[r.choice(len(y), len(Xe))].tolist() if not reg else None
        if pandas_in:
            Xe = pd.DataFrame(Xe, columns=cols)
            Xbg = pd.DataFrame(Xbg, columns=cols) if Xbg is not None else None
        return spec, model, Xbg, Xe

    # ---- kernel ----
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.neighbors import KNeighborsClassifier
    fam = KERNEL_FAMILIES[r.integers(len(KERNEL_FAMILIES))]
    reg = fam in ("ridge", "mlp_reg")
    n_feat = int(r.integers(1, 13))
    n_classes = 2 if (reg or fam == "logreg_p1" or r.random() < 0.5) else int(r.integers(3, 5))
    X, y = _data(r, n_feat, 400, n_classes, 0.0, r.choice([0, 0.3]), reg)
    model = {"ridge": lambda: Ridge(alpha=float(r.uniform(0.1, 10))),
             "logreg_p1": lambda: LogisticRegression(max_iter=2000),
             "logreg_proba": lambda: LogisticRegression(max_iter=2000),
             "mlp_reg": lambda: MLPRegressor(hidden_layer_sizes=(8,), max_iter=300, random_state=0),
             "knn_clf": lambda: KNeighborsClassifier(n_neighbors=5)}[fam]().fit(X, y)
    full = 2 ** n_feat - 2
    ns = r.choice(["auto", "full", "sampled"])
    nsamples = "auto" if ns == "auto" else int(max(full, 1) + 10) if ns == "full" else int(r.integers(max(2 * n_feat + 2, 10), 600))
    link = "logit" if (fam in ("logreg_p1", "logreg_proba", "knn_clf") and r.random() < 0.4) else "identity"
    L1 = ["auto", "auto", False, "num_features(3)", "aic", "bic", 0.01]
    l1 = L1[int(r.integers(len(L1)))]
    pandas_in = bool(r.random() < 0.3)
    cols = [f"f{j}" for j in range(n_feat)]
    Xbg = X[r.choice(len(X), int(r.integers(1, 30)), replace=False)]
    Xe = X[r.choice(len(X), int(r.integers(1, 12)), replace=False)]
    if pandas_in:
        Xbg = pd.DataFrame(Xbg, columns=cols); Xe = pd.DataFrame(Xe, columns=cols)
    spec.update(family=fam, n_feat=n_feat, n_classes=n_classes, nsamples=nsamples, link=link, l1_reg=l1,
                api=str(r.choice(["shap_values", "call"])), pandas=pandas_in, cols=cols, np_seed=int(r.integers(1 << 30)))
    return spec, model, Xbg, Xe


def kernel_f(spec, model):
    fam = spec["family"]
    if fam in ("ridge", "mlp_reg"):
        return model.predict
    if fam == "logreg_p1":
        return lambda Z: model.predict_proba(Z)[:, 1]
    return model.predict_proba


def run_case(shap, spec, model, Xbg, Xe):
    """Run the public shap API exactly as a user would. Returns a dict of numpy outputs."""
    out = {}
    if spec["kind"] == "kernel":
        np.random.seed(spec["np_seed"])  # shap's KernelExplainer samples with the global numpy RNG
        e = shap.KernelExplainer(kernel_f(spec, model), Xbg, link=spec["link"])
        kw = dict(nsamples=spec["nsamples"], l1_reg=spec["l1_reg"], silent=True)
        if spec["api"] == "call":
            ex = e(Xe, l1_reg=spec["l1_reg"], silent=True)  # shap's Kernel __call__ takes no nsamples
            out.update(values=ex.values, base_values=ex.base_values, data=ex.data, feature_names=ex.feature_names)
        else:
            out.update(values=e.shap_values(Xe, **kw), expected_value=e.expected_value)  # Kernel ev is static
        return _clean(out)
    mode, api = spec["mode"], spec["api"]
    if mode == "interventional":
        kw = dict(data=Xbg, feature_perturbation="interventional", model_output=spec["model_output"])
    else:
        kw = dict(feature_perturbation="tree_path_dependent")
    if api == "Explainer":
        e = shap.Explainer(model, **({"masker": Xbg} if mode == "interventional" else {}),
                           **({"model_output": spec["model_output"]} if mode == "interventional" else {}))
    else:
        e = shap.TreeExplainer(model, **kw)
    y = np.asarray(spec["y_explain"]) if spec.get("y_explain") is not None else None
    ca = spec["check_additivity"]
    if api == "interaction_values":
        out.update(values=e.shap_interaction_values(Xe, y=y), expected_value=_ev(e, y))
    elif api == "call_interactions":
        ex = e(Xe, y=y, interactions=True, check_additivity=ca)
        out.update(values=ex.values, base_values=ex.base_values, data=ex.data, feature_names=ex.feature_names)
    elif api in ("call", "Explainer"):
        ex = e(Xe, y=y, check_additivity=ca, approximate=(mode == "approximate"))
        out.update(values=ex.values, base_values=ex.base_values, data=ex.data, feature_names=ex.feature_names)
    else:
        out.update(values=e.shap_values(Xe, y=y, check_additivity=ca, approximate=(mode == "approximate")),
                   expected_value=_ev(e, y))
    return _clean(out)


def _clean(out):
    """shap sometimes leaks a bound method (log_loss expected value) into outputs; store a marker instead."""
    def c(v):
        if callable(v):
            return "<callable>"
        if isinstance(v, np.ndarray) and v.dtype == object and any(callable(x) for x in v.ravel()):
            return "<callable>"
        return v
    return {k: c(v) for k, v in out.items()}


def _ev(e, y):
    ev = e.expected_value
    return ev(y) if callable(ev) else ev  # log_loss output: expected value depends on y


def norm(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and v and not isinstance(v[0], str):
        return np.stack([np.asarray(a, dtype=np.float64) for a in v])
    if isinstance(v, (list, tuple)):
        return list(v)
    a = np.asarray(v)
    return a.astype(np.float64) if a.dtype.kind in "fiub" else a


def compare(exp, got):
    """-> (ok, message). Numeric keys: same shape and max|diff| <= 1e-6*max(1, max|exp|); NaN must match NaN."""
    if "error" in exp:
        if "error" not in got:
            return False, f"shap raised {exp['error']} but shap_rs did not"
        return (exp["error"].split(":")[0] == got["error"].split(":")[0],
                f"both raised (shap: {exp['error'][:80]} | rs: {got['error'][:80]})")
    if "error" in got:
        return False, f"shap_rs raised {got['error'][:200]}"
    worst = 0.0
    for k, ev in exp.items():
        if k not in got:
            return False, f"missing {k}"
        if isinstance(ev, str) or isinstance(got[k], str):
            if ev != got[k]:
                return False, f"{k}: {got[k]!r} != {ev!r}"
            continue
        e, g = norm(ev), norm(got[k])
        if isinstance(e, list) or (isinstance(e, np.ndarray) and e.dtype.kind not in "fiub"):
            if (list(e) if e is not None else None) != (list(g) if g is not None else None):
                return False, f"{k} differs"
            continue
        if e is None or g is None:
            if (e is None) != (g is None):
                return False, f"{k}: None mismatch"
            continue
        if e.shape != g.shape:
            return False, f"{k} shape {g.shape} != {e.shape}"
        if not np.array_equal(np.isnan(e), np.isnan(g)):
            return False, f"{k} NaN pattern differs"
        fe, fg = np.nan_to_num(e), np.nan_to_num(g)
        tol = 1e-6 * max(1.0, float(np.abs(fe).max()) if fe.size else 1.0)
        d = float(np.abs(fe - fg).max()) if fe.size else 0.0
        if d > tol:
            return False, f"{k} maxdiff {d:.2e} > tol {tol:.1e}"
        worst = max(worst, d / tol)
    return True, f"worst diff/tol {worst:.3f}"
