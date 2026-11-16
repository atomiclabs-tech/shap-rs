"""shap_rs.patch() / shap_rs.unpatch(): run the real `shap` package's public API on shap_rs's Rust compute.

Tree explainers: shap's own Python layer (argument handling, model parsing, output shaping, Explanation
objects, exceptions) is kept as is; the C extension it calls (`shap.explainers._tree._cext`) is swapped for a
namespace whose functions run the Rust ports (same signatures, results written into the caller's buffers).

KernelExplainer: only `explain` (one row) is replaced (see _kexplain.py): coalition sampling (numpy's global
RNG replayed), synthetic data, expected values, l1 feature selection and the WLS solve run in Rust, in shap's
order. Construction, `shap_values` (input handling, tqdm, row loop, output) and `__call__` stay shap's own.

unpatch() restores every replaced attribute to the exact original object.
"""

from __future__ import annotations

import types

import numpy as np

from . import _core
from ._kexplain import kernel_explain
from ._tree import _check_width, _f64, _fast_lgbm_dump, _i32

_saved: list[tuple[object, str, object]] = []


# -- tree: _cext replacement ------------------------------------------------------------------------
def _cext_compute_expectations(children_left, children_right, node_sample_weight, values):
    v = values if (values.dtype == np.float64 and values.flags.c_contiguous and values.flags.writeable) \
        else np.ascontiguousarray(values, dtype=np.float64).copy()
    d = _core.compute_expectations(_i32(children_left), _i32(children_right), _f64(node_sample_weight),
                                   v if v.ndim > 1 else v.reshape(-1, 1))
    if v is not values:
        values[...] = v.reshape(values.shape)
    return d


def _cext_dense_tree_update_weights(cl, cr, cd, features, thresholds, threshold_types, values, tree_limit,
                                    node_sample_weight, data, data_missing):
    _check_width(cl, features, data)
    w = node_sample_weight if (node_sample_weight.dtype == np.float64 and node_sample_weight.flags.c_contiguous) \
        else np.ascontiguousarray(node_sample_weight, dtype=np.float64)
    _core.dense_tree_update_weights(_i32(cl), _i32(cr), _i32(cd), _i32(features), _f64(thresholds),
                                    _i32(threshold_types), w, _f64(data),
                                    np.ascontiguousarray(data_missing, dtype=bool))
    if w is not node_sample_weight:
        node_sample_weight[...] = w


def _opt_f64(a):
    return None if a is None else _f64(a)


def _cext_dense_tree_predict(cl, cr, cd, features, thresholds, threshold_types, values, max_depth, tree_limit,
                             base_offset, transform, X, X_missing, y, out):
    _check_width(cl, features, X)
    out[...] = _core.dense_tree_predict(_i32(cl), _i32(cr), _i32(cd), _i32(features), _f64(thresholds),
                                        _i32(threshold_types), _f64(values), int(max_depth), int(tree_limit),
                                        _f64(base_offset), int(transform), _f64(X),
                                        np.ascontiguousarray(X_missing, dtype=bool),
                                        None if y is None else _f64(y).reshape(-1)).reshape(out.shape)


def _cext_dense_tree_shap(cl, cr, cd, features, thresholds, threshold_types, values, node_sample_weight, max_depth,
                          X, X_missing, y, R, R_missing, tree_limit, base_offset, phi, feature_dependence,
                          model_output, interactions):
    _check_width(cl, features, X, R)
    phi[...] = _core.dense_tree_shap(
        _i32(cl), _i32(cr), _i32(cd), _i32(features), _f64(thresholds), _i32(threshold_types), _f64(values),
        _f64(node_sample_weight), int(max_depth), _f64(X), np.ascontiguousarray(X_missing, dtype=bool),
        None if y is None else _f64(y).reshape(-1), _opt_f64(R),
        None if R_missing is None else np.ascontiguousarray(R_missing, dtype=bool), int(tree_limit),
        _f64(base_offset), int(feature_dependence), int(model_output), bool(interactions)).reshape(phi.shape)


def _cext_dense_tree_saabas(cl, cr, cd, features, thresholds, threshold_types, values, max_depth, tree_limit,
                            base_offset, transform, X, X_missing, y, out):
    _check_width(cl, features, X)
    out[...] = _core.dense_tree_saabas(_i32(cl), _i32(cr), _i32(cd), _i32(features), _f64(thresholds),
                                       _i32(threshold_types), _f64(values), int(tree_limit), _f64(base_offset),
                                       _f64(X), np.ascontiguousarray(X_missing, dtype=bool)).reshape(out.shape)


def _make_cext_shim(orig):
    ns = types.SimpleNamespace(**{k: getattr(orig, k) for k in dir(orig) if not k.startswith("__")})
    ns.compute_expectations = _cext_compute_expectations
    ns.dense_tree_update_weights = _cext_dense_tree_update_weights
    ns.dense_tree_predict = _cext_dense_tree_predict
    ns.dense_tree_shap = _cext_dense_tree_shap
    ns.dense_tree_saabas = _cext_dense_tree_saabas
    return ns


def _wrap_tree_ensemble_init(orig_init):
    def __init__(self, model, data=None, data_missing=None, model_output=None):
        with _fast_lgbm_dump(model):
            orig_init(self, model, data, data_missing, model_output)
    __init__.__wrapped__ = orig_init
    return __init__


# -- patch / unpatch --------------------------------------------------------------------------------
def _set(obj, name, value):
    _saved.append((obj, name, obj.__dict__[name] if isinstance(obj, type) else getattr(obj, name)))
    setattr(obj, name, value)


SHAP_VERSION = "0.54.1"  # the only shap release shap_rs is verified to match exactly


def patch(force: bool = False) -> bool:
    """Route shap's TreeExplainer / KernelExplainer compute (and shap.Explainer dispatching to them) to Rust.
    Returns True when patched. On any shap other than SHAP_VERSION it warns and leaves shap untouched,
    because exact agreement is only verified for that release; pass force=True to patch anyway."""
    if _saved:
        return True
    import shap
    if shap.__version__ != SHAP_VERSION and not force:
        import warnings
        warnings.warn(f"shap_rs {_core_version()} is verified against shap {SHAP_VERSION} only, found shap "
                      f"{shap.__version__}: not patching (shap_rs.patch(force=True) to override).", RuntimeWarning,
                      stacklevel=2)
        return False
    import shap.explainers._kernel as sk
    import shap.explainers._tree as st

    _set(st, "_cext", _make_cext_shim(st._cext))
    _set(st.TreeEnsemble, "__init__", _wrap_tree_ensemble_init(st.TreeEnsemble.__init__))
    _set(sk.KernelExplainer, "explain", kernel_explain)
    return True


def _core_version():
    try:
        from importlib.metadata import version
        return version("shap_rs")
    except Exception:
        return "?"


def unpatch():
    """Restore every attribute replaced by patch() to the original object."""
    while _saved:
        obj, name, value = _saved.pop()
        setattr(obj, name, value)


def is_patched():
    return bool(_saved)
