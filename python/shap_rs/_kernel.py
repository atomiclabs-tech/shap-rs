"""KernelExplainer with shap 0.54.1's API. Coalition enumeration/sampling, kernel weighting, expected values,
LARS feature selection and the weighted least squares solve run in Rust; only the user's model is called
from Python (batched across rows to cut call overhead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _core

_CHUNK_ROWS = 1 << 21  # synthetic rows per model call


def _native_l1() -> bool:
    """Run l1_reg feature selection (LARS / LassoLarsIC / Lasso) in Rust, or delegate it to sklearn.
    The Rust LARS reproduces numpy's BLAS summation order on arm64 (macOS Accelerate, Linux OpenBLAS), which
    matters when the LARS path hits exact ties. Elsewhere (x86-64, ...) sklearn runs on the same inputs, as in shap.
    Override with SHAP_RS_L1=rust or SHAP_RS_L1=sklearn."""
    import os
    import platform
    mode = os.environ.get("SHAP_RS_L1", "").lower()
    if mode in ("rust", "sklearn"):
        return mode == "rust"
    return platform.machine().lower() in ("arm64", "aarch64")


def _l1_select(X, y, code, param):
    """shap 0.54.1 KernelExplainer.solve's feature selection, on the mask_aug / eyAdj_aug built in Rust."""
    if code == 1:
        from sklearn.linear_model import lars_path
        return [int(i) for i in lars_path(X, y, max_iter=int(param))[1]]
    if code in (2, 3):
        from sklearn.linear_model import LassoLarsIC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion="aic" if code == 2 else "bic"))
        return [int(i) for i in np.nonzero(model.fit(X, y)[1].coef_)[0]]
    from sklearn.linear_model import Lasso
    return [int(i) for i in np.nonzero(Lasso(alpha=param).fit(X, y).coef_)[0]]


def _l1_code(l1_reg):
    """shap's KernelExplainer.solve l1_reg dispatch -> (code, param) for _core.kernel_solve_batch."""
    if isinstance(l1_reg, str):
        if l1_reg == "auto":
            import warnings
            warnings.warn("l1_reg='auto' is deprecated and will be removed in a future version.", DeprecationWarning)
            return 5, 0.0
        if l1_reg.startswith("num_features("):
            return 1, float(int(l1_reg[len("num_features("):-1]))
        if l1_reg in ("aic", "bic"):
            return (2 if l1_reg == "aic" else 3), 0.0
    elif l1_reg is False or (np.ndim(l1_reg) == 0 and not isinstance(l1_reg, bool) and l1_reg == 0):
        return 0, 0.0
    # anything else goes to sklearn's Lasso(alpha=l1_reg); let sklearn validate the parameter
    from sklearn.linear_model import Lasso
    Lasso(alpha=l1_reg)._validate_params()
    return 4, float(l1_reg)


def explain_rows(ke, X, kwargs, index=None):
    """KernelSHAP for every row of 2-D X with the state of a (shap or shap_rs) KernelExplainer `ke`
    (data, model, link, fnull, vector_out, D, P). Returns phi of shape (n, P, D). Sampled coalitions
    replay numpy's global RNG exactly as shap's per-row `explain` would, and leave it in the same state.
    index = (index_values, index_name, column_names) for keep_index=True: the model then receives DataFrames
    indexed like shap builds them (per explained row, sorted when keep_index_ordered)."""
    nsamples = kwargs.get("nsamples", "auto")
    nsamples = None if nsamples == "auto" else int(nsamples)
    l1_reg = kwargs.get("l1_reg", "num_features(10)")
    import scipy.sparse
    D = ke.D
    sparse = scipy.sparse.issparse(X)
    groups = ke.data.groups
    P = ke.data.groups_size
    g_offs = g_idx = None
    if groups is not None and not (len(groups) == ke.data.data.shape[1]
                                   and all(len(g) == 1 and g[0] == i for i, g in enumerate(groups))):
        g_offs = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
        g_idx = np.concatenate([np.asarray(g, dtype=np.int64).ravel() for g in groups]) if len(groups) \
            else np.zeros(0, dtype=np.int64)
    if sparse:
        bgc = scipy.sparse.csr_matrix(ke.data.data, dtype=np.float64)
        bgc.sort_indices()
        bg = None
        Xc = scipy.sparse.csr_matrix(X, dtype=np.float64)
        Xc.sort_indices()
        n_bg = bgc.shape[0]
    else:
        bg = np.ascontiguousarray(ke.data.data, dtype=np.float64)
        n_bg = bg.shape[0]
    w = np.ascontiguousarray(ke.data.weights, dtype=np.float64)
    name = getattr(ke.link, "__name__", type(ke.link).__name__).lower()
    link_code = 1 if "logit" in name else 0
    Xf = None if sparse else np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.zeros((n, P, D))
    if n == 0:
        return out

    def f(Z):
        o = ke.model.f(Z)
        if isinstance(o, (pd.DataFrame, pd.Series)):
            o = o.values
        return np.asarray(o)

    if index is not None:
        from shap.utils._legacy import convert_to_instance_with_index
        index_values, index_name, column_name = index
        fx_all = f(convert_to_instance_with_index(X, column_name, index_values, index_name).convert_to_df())
        fx_all = fx_all.reshape(n, -1) if ke.vector_out else fx_all.reshape(n, 1)
    else:
        fx_all = f(X).reshape(n, -1) if ke.vector_out else f(X).reshape(n, 1)
    fnull = np.ascontiguousarray(ke.fnull, dtype=np.float64).reshape(-1)
    code, param = _l1_code(l1_reg)
    st = np.random.get_state()
    key, pos = np.ascontiguousarray(st[1], dtype=np.uint32), int(st[2])
    per_row = (2 * P + 2048 if nsamples is None else nsamples) * n_bg
    step = max(1, _CHUNK_ROWS // max(per_row, 1))
    try:
        for s0 in range(0, n, step):
            s1 = min(n, s0 + step)
            if sparse:
                xs = Xc[s0:s1]
                i64 = lambda a: np.ascontiguousarray(a, dtype=np.int64)
                batch, ip, ix, dv, key, pos = _core.kernel_prepare_batch_sparse(
                    i64(xs.indptr), i64(xs.indices), np.ascontiguousarray(xs.data, dtype=np.float64),
                    i64(bgc.indptr), i64(bgc.indices), np.ascontiguousarray(bgc.data, dtype=np.float64),
                    X.shape[1], nsamples, key, pos, D if code == 4 else 0)
                synth = scipy.sparse.csr_matrix((dv, ix, ip), shape=(len(ip) - 1, X.shape[1]))
            else:
                batch, synth, key, pos = _core.kernel_prepare_batch(Xf[s0:s1], bg, nsamples, key, pos,
                                                                    D if code == 4 else 0, g_offs, g_idx)
            if index is None:
                ys = np.reshape(f(synth), (-1, D)) if synth.shape[0] else np.zeros((0, D))
            else:
                ys = _keep_index_eval(ke, f, synth, batch.row_sizes, D)
            out[s0:s1] = _core.kernel_solve_batch(batch, np.ascontiguousarray(ys, dtype=np.float64), w,
                                                  np.ascontiguousarray(fx_all[s0:s1], dtype=np.float64), fnull,
                                                  P, link_code, code, param,
                                                  None if _native_l1() else _l1_select)
    finally:
        np.random.set_state((st[0], key, pos, st[3], st[4]))
    return out


def _keep_index_eval(ke, f, synth, row_sizes, D):
    """shap's KernelExplainer.run with keep_index: one model call per explained row on a DataFrame indexed by
    the tiled background index (sorted by index when keep_index_ordered)."""
    ys = np.zeros((synth.shape[0], D))
    N = ke.data.data.shape[0]
    o = 0
    for size in row_sizes:
        if size:
            idx = pd.DataFrame(np.tile(ke.data.index_value, size // N), columns=[ke.data.index_name])
            data = pd.DataFrame(synth[o:o + size], columns=ke.data.group_names)
            data = pd.concat([idx, data], axis=1).set_index(ke.data.index_name)
            if getattr(ke, "keep_index_ordered", False):
                data = data.sort_index()
            ys[o:o + size] = np.reshape(f(data), (size, D))
        o += size
    return ys


class KernelExplainer:
    def __init__(self, model, data, feature_names=None, link="identity", **kwargs):
        from shap.utils._legacy import (DenseData, convert_to_data, convert_to_link, convert_to_model,
                                        match_model_to_data)
        if feature_names is not None:
            self.data_feature_names = feature_names
        elif isinstance(data, pd.DataFrame):
            self.data_feature_names = list(data.columns)
        self._background_columns = None
        if isinstance(data, pd.DataFrame):
            if not data.columns.is_unique:
                raise ValueError("Background data must have unique column names.")
            self._background_columns = data.columns.copy()
        self.link = convert_to_link(link)
        self.keep_index = kwargs.get("keep_index", False)
        if self.keep_index:
            raise NotImplementedError("keep_index is not supported by shap_rs")
        self.model = convert_to_model(model, keep_index=False)
        self.data = convert_to_data(data, keep_index=False)
        model_null = match_model_to_data(self.model, self.data)
        if not isinstance(self.data, DenseData):
            raise TypeError("shap_rs KernelExplainer only supports dense background data.")
        if self.data.transposed:
            raise ValueError("Shap explainer does not support transposed DenseData currently.")
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]
        self._bg = np.ascontiguousarray(self.data.data, dtype=np.float64)
        self._w = np.ascontiguousarray(self.data.weights, dtype=np.float64)
        name = getattr(self.link, "__name__", type(self.link).__name__).lower()
        self._link_code = 1 if "logit" in name else 0

    def _f(self, Z):
        out = self.model.f(Z)
        if isinstance(out, (pd.DataFrame, pd.Series)):
            out = out.values
        return np.asarray(out)

    def shap_values(self, X, **kwargs):
        input_order = None
        columns = self._background_columns
        if isinstance(X, pd.DataFrame) and columns is not None:
            if set(X.columns) != set(columns):
                raise ValueError("X must have the same column names as the background data")
            input_order = columns.get_indexer(X.columns)
            X = X.loc[:, columns]
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        X = np.asarray(X)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("Instance must have 1 or 2 dimensions!")

        out = explain_rows(self, X, kwargs)
        if not self.vector_out:
            out = out[:, :, 0]
        if input_order is not None:
            out = np.take(out, input_order, axis=1)
        return out[0] if single else out
