"""TreeExplainer with shap 0.54.1's public API; all SHAP computation runs in Rust (shap_rs._core).

Model parsing reuses shap's own `TreeEnsemble` (model -> arrays). Its two internal _cext calls
(`compute_expectations`, `dense_tree_update_weights`) are redirected to the Rust ports for the
duration of the parse, so shap._cext is never called.
"""

from __future__ import annotations

import contextlib
import types
import warnings

import numpy as np
import pandas as pd

from . import _core

output_transform_codes = {"identity": 0, "logistic": 1, "logistic_nlogloss": 2, "squared_loss": 3}
feature_perturbation_codes = {"interventional": 0, "tree_path_dependent": 1}



def _check_width(cl, features, *mats):
    """shap 0.54.1 does not check that X (or the background) has as many columns as the trees use; its C++ then
    reads past the end of each row (into the next row, and past the buffer for the last one), so its output is
    undefined. Raise instead of reading out of bounds."""
    feats = np.asarray(features)[np.asarray(cl) >= 0]
    if feats.size:
        need = int(feats.max()) + 1
        for a in mats:
            if a is not None and a.ndim == 2 and a.shape[1] < need:
                raise ValueError(f"The model uses feature index {need - 1} but the data passed has only {a.shape[1]} "
                                 "columns (shap 0.54.1 would read out of bounds here; its result is undefined).")

def _i32(a):
    return np.ascontiguousarray(a, dtype=np.int32)


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _shim_compute_expectations(children_left, children_right, node_sample_weight, values):
    v = np.ascontiguousarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if not v.flags.writeable:
        v = v.copy()
    d = _core.compute_expectations(_i32(children_left), _i32(children_right), _f64(node_sample_weight), v)
    if v is not values:
        values[...] = v.reshape(values.shape)
    return d


def _shim_dense_tree_update_weights(cl, cr, cd, features, thresholds, threshold_types, values, tree_limit,
                                    node_sample_weight, data, data_missing):
    w = node_sample_weight if (node_sample_weight.dtype == np.float64 and node_sample_weight.flags.c_contiguous) \
        else np.ascontiguousarray(node_sample_weight, dtype=np.float64)
    _core.dense_tree_update_weights(_i32(cl), _i32(cr), _i32(cd), _i32(features), _f64(thresholds),
                                    _i32(threshold_types), w, _f64(data),
                                    np.ascontiguousarray(data_missing, dtype=bool))
    if w is not node_sample_weight:
        node_sample_weight[...] = w


_shim = types.SimpleNamespace(compute_expectations=_shim_compute_expectations,
                              dense_tree_update_weights=_shim_dense_tree_update_weights)


@contextlib.contextmanager
def _rust_cext():
    import shap.explainers._tree as st
    old = st._cext
    st._cext = _shim
    try:
        yield
    finally:
        st._cext = old


def _lgbm_fast_tree_info(booster):
    """Arrays for every tree of a LightGBM Booster from `model_to_string()` (~25x faster than the JSON
    `dump_model()` shap parses). Same layout as shap's SingleTree for LightGBM dumps: internal node i at
    index i (= split_index), leaf j at num_parents + j. Returns None (caller falls back to shap's path)
    for categorical splits or stump trees."""
    text = booster.model_to_string()
    out = []
    for block in text.split("\nTree=")[1:]:
        block = block.split("\nend of trees")[0]
        kv = {}
        for line in block.split("\n")[1:]:
            k, sep, v = line.partition("=")
            if sep:
                kv[k] = v
        nl = int(kv["num_leaves"])
        if nl < 2 or int(kv.get("num_cat", "0")) != 0:
            return None
        npar = nl - 1
        nn = 2 * npar + 1
        arr = lambda key, dt: np.array(kv[key].split(" "), dtype=dt)
        lc, rc = arr("left_child", np.int64), arr("right_child", np.int64)
        lc = np.where(lc >= 0, lc, ~lc + npar)
        rc = np.where(rc >= 0, rc, ~rc + npar)
        dtype = arr("decision_type", np.int64)
        if np.any(dtype & 1):
            return None
        cl = -np.ones(nn, dtype=np.int32); cr = -np.ones(nn, dtype=np.int32); cd = -np.ones(nn, dtype=np.int32)
        cl[:npar] = lc; cr[:npar] = rc
        cd[:npar] = np.where(dtype & 2, lc, rc)
        feats = -np.ones(nn, dtype=np.int32); feats[:npar] = arr("split_feature", np.int64)
        thr = -np.ones(nn, dtype=np.float64); thr[:npar] = arr("threshold", np.float64)
        vals = np.empty((nn, 1), dtype=np.float64)
        vals[:npar, 0] = arr("internal_value", np.float64); vals[npar:, 0] = arr("leaf_value", np.float64)
        nsw = np.empty(nn, dtype=np.float64)
        nsw[:npar] = arr("internal_count", np.float64); nsw[npar:] = arr("leaf_count", np.float64)
        out.append({"children_left": cl, "children_right": cr, "children_default": cd, "features": feats,
                    "thresholds": thr, "values": vals, "node_sample_weight": nsw})
    return out


@contextlib.contextmanager
def _fast_lgbm_dump(model):
    """Temporarily shadow `dump_model` on the model's Booster instance (only while shap parses it)."""
    booster = None
    try:
        import lightgbm
        if isinstance(model, lightgbm.Booster):
            booster = model
        elif isinstance(model, lightgbm.LGBMModel):
            booster = model.booster_
    except Exception:
        booster = None
    info = _lgbm_fast_tree_info(booster) if booster is not None else None
    if info is None:
        yield
        return
    booster.dump_model = lambda *a, **k: {"tree_info": info}
    try:
        yield
    finally:
        del booster.dump_model


class TreeExplainer:
    def __init__(self, model, data=None, model_output="raw", feature_perturbation="auto", feature_names=None,
                 approximate=None, link=None, linearize_link=None):
        import shap
        from shap.explainers._tree import TreeEnsemble, _check_xgboost_version
        from shap.utils._legacy import DenseData
        from shap.utils._exceptions import ExplainerError, InvalidFeaturePerturbationError

        if feature_names is not None:
            self.data_feature_names = feature_names
        elif isinstance(data, pd.DataFrame):
            self.data_feature_names = list(data.columns)

        # same masker handling as shap.Explainer.__init__ (Independent masker, max_samples=100)
        if data is None:
            pass
        elif isinstance(data, DenseData):
            data = data.data
        elif isinstance(data, shap.maskers.Independent):
            data = data.data
        elif isinstance(data, (pd.DataFrame, np.ndarray)) or hasattr(data, "shape"):
            masker = shap.maskers.Independent(data)
            if getattr(masker, "clustering", None) is not None:
                raise ExplainerError("TreeExplainer does not support clustered data inputs!")
            data = masker.data
        else:
            raise ExplainerError(f"Unsupported masker type: {type(data)}!")

        self.data = data.values if isinstance(data, pd.DataFrame) else data

        if feature_perturbation == "auto":
            feature_perturbation = "interventional" if self.data is not None else "tree_path_dependent"
        elif feature_perturbation == "interventional":
            if self.data is None:
                warnings.warn("In the future, passing feature_perturbation='interventional' without providing a "
                              "background dataset will raise an error.", FutureWarning)
                feature_perturbation = "tree_path_dependent"
            elif self.data.shape[0] > 1_000:
                warnings.warn(f"Passing {self.data.shape[0]} background samples may lead to slow runtimes.")
        elif feature_perturbation != "tree_path_dependent":
            raise InvalidFeaturePerturbationError(
                "feature_perturbation must be 'auto', 'interventional', or 'tree_path_dependent'. "
                f"Got {feature_perturbation} instead.")

        self.data_missing = None if self.data is None else pd.isna(self.data)
        self.feature_perturbation = feature_perturbation
        self.expected_value = None
        if isinstance(model, TreeEnsemble):
            self.model = model
        else:
            with _rust_cext(), _fast_lgbm_dump(model):
                self.model = TreeEnsemble(model, self.data, self.data_missing, model_output)
        self.model_output = model_output

        if feature_perturbation == "tree_path_dependent":
            if self.model.model_output != "raw":
                raise ValueError('Only model_output="raw" is supported for feature_perturbation="tree_path_dependent"')
        elif data is None:
            raise ValueError('A background dataset must be provided unless you are using '
                             'feature_perturbation="tree_path_dependent"!')

        if self.model.model_output != "raw":
            if self.model.objective is None and self.model.tree_output is None:
                raise Exception("Model does not have a known objective or output type! When model_output is "
                                'not "raw" then we need to know the model\'s objective or link function.')

        if self.model.model_type == "xgboost":
            import xgboost
            _check_xgboost_version(xgboost.__version__)

        m = self.model
        self._arrays = (_i32(m.children_left), _i32(m.children_right), _i32(m.children_default), _i32(m.features),
                        _f64(m.thresholds), _i32(m.threshold_types), _f64(m.values))
        self._nsw = _f64(m.node_sample_weight)
        self._base_offset = _f64(m.base_offset)
        if self.data is not None:
            self._R = _f64(self.data)
            self._R_missing = np.ascontiguousarray(self.data_missing, dtype=bool)

        if self.model.model_output == "log_loss":
            self.expected_value = self.__dynamic_expected_value
        elif data is not None:
            try:
                self.expected_value = self._predict(self.data).mean(0)
            except ValueError:
                raise ExplainerError("Currently TreeExplainer can only handle models with categorical splits when "
                                     'feature_perturbation="tree_path_dependent" and no background data is passed.')
            if hasattr(self.expected_value, "__len__") and len(self.expected_value) == 1:
                self.expected_value = self.expected_value[0]
        elif hasattr(self.model, "node_sample_weight"):
            self.expected_value = self.model.values[:, 0].sum(0)
            if self.expected_value.size == 1:
                self.expected_value = self.expected_value[0]
            self.expected_value += self.model.base_offset
            if self.model.model_output != "raw":
                self.expected_value = None

        if self.model.model_output == "probability_doubled" and self.expected_value is not None:
            self.expected_value = [1 - self.expected_value, self.expected_value]

    def __dynamic_expected_value(self, y):
        return self._predict(self.data, np.ones(self.data.shape[0]) * y).mean(0)

    # -- TreeEnsemble.predict, with the _cext call replaced by Rust ------------------------------
    def _predict(self, X, y=None, output=None, tree_limit=None):
        m = self.model
        if output is None:
            output = m.model_output
        if m.model_type == "pyspark":
            raise NotImplementedError("Predict with pyspark isn't implemented.")
        if m.model_type == "xgboost" and m.num_stacked_models != m.num_outputs:
            raise NotImplementedError("XGBoost with boosted random forest is not yet supported.")
        if tree_limit is None:
            tree_limit = -1 if m.tree_limit is None else m.tree_limit
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy(dtype=m.input_dtype, na_value=np.nan)
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype.type != m.input_dtype:
            X = X.astype(m.input_dtype)
        X_missing = np.isnan(X, dtype=bool)
        if tree_limit < 0 or tree_limit > m.values.shape[0]:
            tree_limit = m.values.shape[0]
        if output == "logloss":
            if y is None:
                raise ValueError("Both samples and labels must be provided when explaining the loss")
        transform = m.get_transform()
        cl, cr, cd, f, t, tt, v = self._arrays
        _check_width(cl, f, X)
        out = _core.dense_tree_predict(cl, cr, cd, f, t, tt, v, int(m.max_depth), int(tree_limit), self._base_offset,
                                       output_transform_codes[transform], _f64(X),
                                       np.ascontiguousarray(X_missing), None if y is None else _f64(y))
        if flat_output:
            return out.flatten()[0] if m.num_outputs == 1 else out.reshape(-1, m.num_outputs)
        return out.flatten() if m.num_outputs == 1 else out

    def _validate_inputs(self, X, y, tree_limit, check_additivity):
        from shap.utils._exceptions import DimensionError, ExplainerError
        m = self.model
        if tree_limit is None:
            tree_limit = -1 if m.tree_limit is None else m.tree_limit
        if tree_limit < 0 or tree_limit > m.values.shape[0]:
            tree_limit = m.values.shape[0]
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy(dtype=m.input_dtype, na_value=np.nan)
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != m.input_dtype:
            X = X.astype(m.input_dtype)
        X_missing = np.isnan(X, dtype=bool)
        assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"
        if m.model_output == "log_loss":
            if y is None:
                raise ExplainerError('Both samples and labels must be provided when model_output = "log_loss" '
                                     "(i.e. `explainer.shap_values(X, y)`)!")
            if X.shape[0] != len(y):
                raise DimensionError(f"The number of labels ({len(y)}) does not match the number of samples to "
                                     f"explain ({X.shape[0]})!")
        if self.feature_perturbation == "tree_path_dependent" and not m.fully_defined_weighting:
            raise ExplainerError("The background dataset you provided does not cover all the leaves in the model, "
                                 'so TreeExplainer cannot run with the feature_perturbation="tree_path_dependent" '
                                 "option!")
        if check_additivity and m.model_type == "pyspark":
            check_additivity = False
        return X, y, X_missing, flat_output, tree_limit, check_additivity

    def _short_circuit(self, X, tree_limit, approximate, check_additivity):
        """Same delegation as shap: XGBoost/LightGBM/CatBoost compute path-dependent SHAP natively."""
        from shap.explainers._tree import _xgboost_n_iterations
        m = self.model
        phi = None
        model_output_vals = None
        if m.model_type == "xgboost":
            import xgboost
            n_iterations = _xgboost_n_iterations(tree_limit, m.num_stacked_models)
            if not isinstance(X, xgboost.core.DMatrix):
                X = xgboost.DMatrix(X, **getattr(m, "_xgb_dmatrix_props", {}))
            phi = m.original_model.predict(X, iteration_range=(0, n_iterations), pred_contribs=True,
                                           approx_contribs=approximate, validate_features=False)
            if check_additivity and m.model_output == "raw":
                model_output_vals = m.original_model.predict(X, iteration_range=(0, n_iterations),
                                                             output_margin=True, validate_features=False)
        elif m.model_type == "lightgbm":
            phi = m.original_model.predict(X, num_iteration=tree_limit, pred_contrib=True)
            if phi.shape[1] != X.shape[1] + 1:
                phi = phi.reshape(X.shape[0], phi.shape[1] // (X.shape[1] + 1), X.shape[1] + 1)
        elif m.model_type == "catboost":
            import catboost
            if not isinstance(X, catboost.Pool):
                X = catboost.Pool(X, cat_features=m.cat_feature_indices)
            phi = m.original_model.get_feature_importance(data=X, fstr_type="ShapValues")
        if phi is None:
            return None
        if len(phi.shape) == 3:
            self.expected_value = [phi[0, i, -1] for i in range(phi.shape[1])]
            out = [phi[:, i, :-1] for i in range(phi.shape[1])]
        else:
            self.expected_value = phi[0, -1]
            out = phi[:, :-1]
        if check_additivity and model_output_vals is not None:
            self.assert_additivity(out, model_output_vals)
        if isinstance(out, list):
            out = np.stack(out, axis=-1)
        return out

    def _dense_tree_shap(self, X, X_missing, y, tree_limit, transform, interactions):
        cl, cr, cd, f, t, tt, v = self._arrays
        R = self._R if self.data is not None else None
        Rm = self._R_missing if self.data is not None else None
        _check_width(cl, f, X, R)
        return _core.dense_tree_shap(cl, cr, cd, f, t, tt, v, self._nsw, int(self.model.max_depth), _f64(X),
                                     np.ascontiguousarray(X_missing), None if y is None else _f64(y), R, Rm,
                                     int(tree_limit), self._base_offset,
                                     feature_perturbation_codes[self.feature_perturbation],
                                     output_transform_codes[transform], interactions)

    def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True, from_call=False):
        from shap.explainers._tree import _xgboost_cat_unsupported
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit
        if self.feature_perturbation == "tree_path_dependent" and self.model.model_type != "internal" \
                and self.data is None:
            phi = self._short_circuit(X, tree_limit, approximate, check_additivity)
            if phi is not None:
                return phi
        X, y, X_missing, flat_output, tree_limit, check_additivity = self._validate_inputs(
            X, y, tree_limit, check_additivity)
        transform = self.model.get_transform()
        _xgboost_cat_unsupported(self.model)
        if approximate:
            cl, cr, cd, f, t, tt, v = self._arrays
            _check_width(cl, f, X)
            phi = _core.dense_tree_saabas(cl, cr, cd, f, t, tt, v, int(tree_limit), self._base_offset, _f64(X),
                                          np.ascontiguousarray(X_missing))
        else:
            phi = self._dense_tree_shap(X, X_missing, y, tree_limit, transform, False)
        out = self._get_shap_output(phi, flat_output)
        if check_additivity and self.model.model_output == "raw":
            self.assert_additivity(out, self._predict(X))
        if isinstance(out, list):
            out = np.stack(out, axis=-1)
        return out

    def _get_shap_output(self, phi, flat_output):
        if self.model.num_outputs == 1:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = phi[0, -1, 0]
            out = phi[0, :-1, 0] if flat_output else phi[:, :-1, 0]
        else:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
            if flat_output:
                out = [phi[0, :-1, i] for i in range(self.model.num_outputs)]
            else:
                out = [phi[:, :-1, i] for i in range(self.model.num_outputs)]
        if self.model.model_output == "probability_doubled":
            out = [-out, out]
        return out

    def shap_interaction_values(self, X, y=None, tree_limit=None):
        assert self.model.model_output == "raw", \
            'Only model_output = "raw" is supported for SHAP interaction values right now!'
        transform = "identity"
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit
        if self.model.model_type == "xgboost" and self.feature_perturbation == "tree_path_dependent":
            import xgboost
            from shap.explainers._tree import _xgboost_n_iterations
            if not isinstance(X, xgboost.core.DMatrix):
                X = xgboost.DMatrix(X)
            n_iterations = _xgboost_n_iterations(tree_limit, self.model.num_stacked_models)
            phi = self.model.original_model.predict(X, iteration_range=(0, n_iterations), pred_interactions=True,
                                                    validate_features=False)
            if len(phi.shape) == 4:
                self.expected_value = [phi[0, i, -1, -1] for i in range(phi.shape[1])]
                return np.swapaxes(phi[:, :, :-1, :-1], axis1=1, axis2=3)
            self.expected_value = phi[0, -1, -1]
            return phi[:, :-1, :-1]
        elif self.model.model_type == "catboost" and self.feature_perturbation == "tree_path_dependent":
            import catboost
            if not isinstance(X, catboost.Pool):
                X = catboost.Pool(X, cat_features=self.model.cat_feature_indices)
            phi = self.model.original_model.get_feature_importance(data=X, fstr_type="ShapInteractionValues")
            if len(phi.shape) == 4:
                self.expected_value = getattr(self, "expected_value", [phi[0, i, -1, -1] for i in range(phi.shape[1])])
                return [phi[:, i, :-1, :-1] for i in range(phi.shape[1])]
            self.expected_value = getattr(self, "expected_value", phi[0, -1, -1])
            return phi[:, :-1, :-1]

        X, y, X_missing, flat_output, tree_limit, _ = self._validate_inputs(X, y, tree_limit, False)
        if self.feature_perturbation == "interventional":
            raise ValueError("FEATURE_DEPENDENCE::independent does not support interactions!")
        phi = self._dense_tree_shap(X, X_missing, y, tree_limit, transform, True)
        return self._get_shap_interactions_output(phi, flat_output)

    def _get_shap_interactions_output(self, phi, flat_output):
        if self.model.num_outputs == 1:
            self.expected_value = getattr(self, "expected_value", phi[0, -1, -1, 0])
            out = phi[0, :-1, :-1, 0] if flat_output else phi[:, :-1, :-1, 0]
        else:
            self.expected_value = [phi[0, -1, -1, i] for i in range(phi.shape[3])]
            if flat_output:
                out = np.stack([phi[0, :-1, :-1, i] for i in range(self.model.num_outputs)], axis=-1)
            else:
                out = np.stack([phi[:, :-1, :-1, i] for i in range(self.model.num_outputs)], axis=-1)
        return out

    def assert_additivity(self, phi, model_output):
        from shap.utils._exceptions import ExplainerError

        def check_sum(sum_val, model_output):
            if not np.allclose(sum_val, model_output, atol=1e-2, rtol=1e-2):
                ind = np.argmax(np.abs(sum_val - model_output))
                raise ExplainerError(
                    "Additivity check failed in TreeExplainer! This check failed because for one of the samples the "
                    f"sum of the SHAP values was {sum_val[ind]:f}, while the model output was {model_output[ind]:f}.")

        if isinstance(phi, list):
            for i in range(len(phi)):
                check_sum(self.expected_value[i] + phi[i].sum(-1), model_output[:, i])
        else:
            check_sum(self.expected_value + phi.sum(-1), model_output)
