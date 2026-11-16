"""Replacement for shap 0.54.1's `KernelExplainer.explain` (one explained row), used by shap_rs.patch().

shap's own `shap_values` stays in charge: it converts the input, shows the tqdm bar, loops over rows, calls
`explain` once per row and assembles the output, so all of that matches shap by construction. This `explain`
follows shap's step order exactly, so the user's model sees the same calls (same objects, container types,
dtypes, row order and batch boundaries), np.random is drawn from at the same points, and exceptions and
warnings happen where shap's happen:

  varying groups -> model(x) -> [M >= 2] coalition sampling (np.random) -> model(synthetic rows)
  -> per output: l1 selection (sklearn's Lasso draws np.random once) -> weighted least squares.

Coalition sampling, synthetic data, expected values, l1 selection and the WLS solve are Rust. The l1 ports flag
every point where sklearn would warn or raise (ConvergenceWarning, numpy RuntimeWarning, ValueError); those
selections are rerun by sklearn itself (on the same inputs, as shap calls it), so sklearn's own warnings and
exceptions surface unchanged. shap's own warnings are re-emitted with shap's file and line.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse

from . import _core

_RAND_R_MAX = 0x7FFFFFFF  # sklearn.utils._random.RAND_R_MAX: Lasso.fit draws randint(0, RAND_R_MAX)
_SINGULAR = (
    "Linear regression equation is singular, a least squares solutions is used instead.\n"
    "To avoid this situation and get a regular matrix do one of the following:\n"
    "1) turn up the number of samples,\n"
    "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
    "3) group features together to reduce the number of inputs that need to be explained."
)
_AUTO = "l1_reg='auto' is deprecated and will be removed in a future version."
_lines: dict = {}


def _shap_line(key):
    """Line in shap/explainers/_kernel.py where shap emits the warning `key`."""
    if not _lines:
        import shap.explainers._kernel as sk
        with open(sk.__file__, encoding="utf-8") as fh:
            src = fh.read().splitlines()
        marks = {"auto": "l1_reg='auto' is deprecated", "singular": "Linear regression equation is singular",
                 "zero_tail": "self.kernelWeights[nfixed_samples:] *= weight_left"}
        for k, m in marks.items():
            i = next((j for j, s in enumerate(src) if m in s), None)
            if i is not None and k == "singular":  # multi-line call: the warning reports the `warnings.warn(` line
                while i > 0 and "warnings.warn(" not in src[i]:
                    i -= 1
            _lines[k] = (i + 1) if i is not None else 1
    return _lines[key]


def _warn_as_shap(message, category, key):
    """warnings.warn as if issued by shap's KernelExplainer code (same file, line, module and registry)."""
    import shap.explainers._kernel as sk
    warnings.warn_explicit(message, category, sk.__file__, _shap_line(key), module=sk.__name__,
                           registry=sk.__dict__.setdefault("__warningregistry__", {}), module_globals=sk.__dict__)


def _numeric(dt):
    return np.issubdtype(dt, np.number) and not np.issubdtype(dt, np.complexfloating)


def _sorted_csr(a):
    c = scipy.sparse.csr_matrix(a, dtype=np.float64, copy=True)
    c.sum_duplicates()  # also sorts the indices (lil rows are sorted and duplicate-free)
    i64 = lambda v: np.ascontiguousarray(v, dtype=np.int64)
    return i64(c.indptr), i64(c.indices), np.ascontiguousarray(c.data, dtype=np.float64)


def _groups_csr(groups):
    offs = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    idx = np.concatenate([np.asarray(g, dtype=np.int64).ravel() for g in groups]) if len(groups) \
        else np.zeros(0, dtype=np.int64)
    return offs, idx


def _varying(self, x):
    """self.varying_groups(x), in Rust for numeric dense/dense and sparse/sparse data (shap's own otherwise)."""
    bg = self.data.data
    if scipy.sparse.issparse(x):
        if scipy.sparse.issparse(bg) and _numeric(x.dtype) and _numeric(bg.dtype) and x.shape[0] == 1:
            _, xi, xv = _sorted_csr(x)
            v = _core.kernel_row_varying_sparse(xi, xv, *_sorted_csr(bg), bg.shape[1])
            return np.asarray(v, dtype=np.result_type(bg.indices.dtype, np.int32))
        return self.varying_groups(x)
    if (isinstance(x, np.ndarray) and isinstance(bg, np.ndarray) and x.ndim == 2 and x.shape[0] == 1
            and bg.ndim == 2 and _numeric(x.dtype) and _numeric(bg.dtype) and x.shape[1] == bg.shape[1]
            and self.data.groups is not None):
        groups = self.data.groups
        trivial = len(groups) == bg.shape[1] and all(len(g) == 1 and g[0] == i for i, g in enumerate(groups))
        offs, idx = (None, None) if trivial else _groups_csr(groups)
        if idx is None or (len(idx) and idx.min() >= 0 and idx.max() < bg.shape[1]):
            v = _core.kernel_row_varying(np.ascontiguousarray(x[0], dtype=np.float64),
                                         np.ascontiguousarray(bg, dtype=np.float64), offs, idx)
            return np.asarray(v, dtype=np.intp)
    return self.varying_groups(x)


def _check_nsamples(self):
    """shap's allocate() is where a non-integer or negative nsamples fails; reproduce that failure."""
    ns = self.nsamples
    if isinstance(ns, (int, np.integer)) and ns >= 0:
        return int(ns)
    type(self).allocate(self)  # raises exactly as in shap (shap's own buffer setup, no SHAP compute)
    import operator
    return operator.index(ns)


def _synth(self, x, masks, n_added):
    """shap's synth_data (allocate + addsample for every mask) in the same container and dtype."""
    bg = self.data.data
    ns = masks.shape[0]
    vfg = self.varyingFeatureGroups
    vg = [np.atleast_1d(np.asarray(g)) for g in vfg] if isinstance(vfg, list) else \
        [np.atleast_1d(g) for g in np.asarray(vfg)]
    vg_offs, vg_idx = _groups_csr(vg) if vg else (np.zeros(1, np.int64), np.zeros(0, np.int64))
    x_ok = _numeric(x.dtype) and (scipy.sparse.issparse(x) or isinstance(x, np.ndarray)) and x.shape[0] == 1
    if x_ok and bg.dtype in (np.float64, np.float32) and (len(vg_idx) == 0 or vg_idx.min() >= 0):
        xr = np.ascontiguousarray((x.toarray() if scipy.sparse.issparse(x) else x)[0], dtype=np.float64)
        if xr.shape[0] == bg.shape[1]:
            m = np.ascontiguousarray(masks, dtype=np.float64)
            if not scipy.sparse.issparse(bg):
                synth = _core.kernel_row_synth(xr, np.ascontiguousarray(bg, dtype=np.float64), m, n_added,
                                               vg_offs, vg_idx)
                return synth if bg.dtype == np.float64 else synth.astype(bg.dtype)
            ip, ix, dv = _core.kernel_row_synth_sparse(xr, *_sorted_csr(bg), m, n_added, vg_offs, vg_idx)
            return scipy.sparse.csr_matrix((dv.astype(bg.dtype, copy=False), ix, ip),
                                           shape=(ns * bg.shape[0], bg.shape[1])).tolil()
    # anything else (object / integer data, ...): shap's own buffer code, sample by sample
    type(self).allocate(self)
    for s in range(n_added):
        type(self).addsample(self, x, masks[s], 1.0)
    return self.synth_data


def _model_out(self, out):
    from shap.utils import safe_isinstance
    if isinstance(out, (pd.DataFrame, pd.Series)):
        out = out.values
    elif safe_isinstance(out, "tensorflow.python.framework.ops.SymbolicTensor"):
        out = self._convert_symbolic_tensor(out)
    return out


def _select_sklearn(X, y, l1):
    """What shap's solve runs for feature selection (sklearn itself: its own warnings, exceptions, RNG use)."""
    if isinstance(l1, str) and l1.startswith("num_features("):
        from sklearn.linear_model import lars_path
        return lars_path(X, y, max_iter=int(l1[len("num_features("):-1]))[1]
    if l1 in ("auto", "bic", "aic"):
        from sklearn.linear_model import LassoLarsIC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion="aic" if l1 == "auto" else l1))
        return np.nonzero(model.fit(X, y)[1].coef_)[0]
    from sklearn.linear_model import Lasso
    return np.nonzero(Lasso(alpha=l1).fit(X, y).coef_)[0]


def _select(self, eyAdj, total, l1):
    """Feature selection of shap's solve: Rust when nothing observable differs, else sklearn on the same input."""
    from ._kernel import _native_l1
    masks, kw = self.maskMatrix, self.kernelWeights
    code = param = None
    if isinstance(l1, str) and l1.startswith("num_features("):
        r = int(l1[len("num_features("):-1])  # same ValueError as shap for a malformed string
        if r >= 0:
            code, param = 1, float(r)
    elif l1 in ("auto", "bic", "aic"):
        code, param = (3, 0.0) if l1 == "bic" else (2, 0.0)
    else:
        from sklearn.linear_model import Lasso
        Lasso(alpha=l1)._validate_params()  # Lasso(alpha=l1).fit validates first: same exception
        code, param = 4, float(l1)
    if code is not None and _native_l1():
        st = np.random.get_state() if code == 4 else None
        if code == 4:
            np.random.mtrand._rand.randint(0, _RAND_R_MAX)  # sklearn's Lasso.fit seeds its CD from the global RNG
        sel = _core.kernel_row_select(eyAdj, total, masks, kw, code, param)
        if sel is not None:
            return sel
        if st is not None:
            np.random.set_state(st)
    X, y = _core.kernel_row_aug(eyAdj, total, masks, kw)
    return [int(i) for i in _select_sklearn(X, y, l1)]


def kernel_explain(self, incoming_instance, **kwargs):
    from shap.utils._legacy import convert_to_instance, match_instance_to_data

    instance = convert_to_instance(incoming_instance)
    match_instance_to_data(instance, self.data)

    self.varyingInds = _varying(self, instance.x)
    if self.data.groups is None:
        self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
        self.M = self.varyingFeatureGroups.shape[0]
    else:
        self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
        self.M = len(self.varyingFeatureGroups)
        groups = self.data.groups
        if self.varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in self.varyingInds):
            self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
            if self.varyingFeatureGroups.shape[1] == 1:
                self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

    if self.keep_index:
        model_out = self.model.f(instance.convert_to_df())
    else:
        model_out = self.model.f(instance.x)
    model_out = _model_out(self, model_out)
    self.fx = model_out[0]
    if not self.vector_out:
        self.fx = np.array([self.fx])

    if self.M == 0:
        phi = np.zeros((self.data.groups_size, self.D))
    elif self.M == 1:
        phi = np.zeros((self.data.groups_size, self.D))
        diff = self.link.f(self.fx) - self.link.f(self.fnull)
        for d in range(self.D):
            phi[self.varyingInds[0], d] = diff[d]
    else:
        self.l1_reg = kwargs.get("l1_reg", "num_features(10)")
        self.nsamples = kwargs.get("nsamples", "auto")
        if self.nsamples == "auto":
            self.nsamples = 2 * self.M + 2**11
        self.max_samples = 2**30
        if self.M <= 30:
            self.max_samples = 2**self.M - 2
            if self.nsamples > self.max_samples:
                self.nsamples = self.max_samples
        ns = _check_nsamples(self)

        # coalitions (np.random.choice + one np.random.permutation per draw, replayed in Rust)
        st = np.random.get_state()
        masks, kw, n_added, key, pos, zero_tail = _core.kernel_row_plan(
            self.M, ns, np.ascontiguousarray(st[1], dtype=np.uint32), int(st[2]))
        np.random.set_state((st[0], key, pos, st[3], st[4]))
        if zero_tail:  # shap: weight_left / kernelWeights[nfixed:].sum() with an empty tail
            if np.geterr()["divide"] == "warn":
                _warn_as_shap("divide by zero encountered in scalar divide", RuntimeWarning, "zero_tail")
            else:
                np.float64(1.0) / np.float64(0.0)
        self.synth_data = _synth(self, instance.x, masks, n_added)
        self.maskMatrix, self.kernelWeights = masks, kw
        self.nsamplesAdded, self.nsamplesRun = n_added, 0
        self.lastMask = np.zeros(ns)
        self.y = np.zeros((ns * self.N, self.D))
        self.ey = np.zeros((ns, self.D))
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, ns)

        # run(): one model call on the synthetic rows, as shap builds them
        num_to_run = n_added * self.N
        data = self.synth_data[0:num_to_run, :]
        if self.keep_index:
            index = self.synth_data_index[0:num_to_run]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        modelOut = _model_out(self, self.model.f(data))
        self.y[0:num_to_run, :] = np.reshape(modelOut, (num_to_run, self.D))
        self.ey = _core.kernel_row_ey(self.y, np.ascontiguousarray(self.data.weights, dtype=np.float64), ns, n_added)
        self.nsamplesRun = n_added

        # solve() per output
        phi = np.zeros((self.data.groups_size, self.D))
        fraction_evaluated = self.nsamples / self.max_samples
        for d in range(self.D):
            eyAdj = np.ascontiguousarray(self.linkfv(self.ey[:, d]) - self.link.f(self.fnull[d]), dtype=np.float64)
            total_raw = self.link.f(self.fx[d]) - self.link.f(self.fnull[d])
            # a (n, 1)-shaped model output with a scalar null output makes this a length-1 array in shap, which
            # broadcasts through the selection and the solve and fails when stored into phi (below)
            total = float(np.asarray(total_raw).reshape(-1)[0]) if np.size(total_raw) == 1 else float(total_raw)
            nonzero_inds = None
            if self.l1_reg == "auto":
                _warn_as_shap(_AUTO, DeprecationWarning, "auto")
            if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
                nonzero_inds = _select(self, eyAdj, total, self.l1_reg)
            if nonzero_inds is not None and len(nonzero_inds) == 0:
                vphi = np.zeros(self.M)
            else:
                sel = list(range(self.M)) if nonzero_inds is None else [int(i) for i in nonzero_inds]
                vphi, singular = _core.kernel_row_wls(eyAdj, total, masks, kw, sel)
                if singular:
                    _warn_as_shap(_SINGULAR, UserWarning, "singular")
                if np.ndim(total_raw) != 0:
                    np.zeros(1)[0] = total_raw  # shap: phi[nonzero_inds[-1]] = <array> - sum(w) -> same numpy error
            phi[self.varyingInds, d] = vphi

    if not self.vector_out:
        phi = np.squeeze(phi, axis=1)
    return phi
