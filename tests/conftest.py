"""Differential test harness: every scenario runs twice in this process, first on REAL shap 0.54.1 (unpatched),
then with shap_rs.patch() active, on the same inputs and seeds. The two runs must agree on the result (type,
shape, dtype, values), the exception (type and message), the warnings (category, message, file, line), the
calls the user's model received (container type, dtype, shape, columns/index, exact content, order), the tqdm
progress bars shap opened, and the global numpy RNG state afterwards. Nothing is compared against stored output.
"""
from __future__ import annotations

import hashlib
import io
import warnings
from contextlib import redirect_stderr

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import shap
import shap.explainers._kernel as sk
import shap_rs


def _digest(a) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype == object:
        return hashlib.sha1(repr(a.tolist()).encode()).hexdigest()
    return hashlib.sha1(a.tobytes()).hexdigest() + str(a.dtype)


def describe(z):
    """Exact, comparable description of whatever the model was called with."""
    if isinstance(z, pd.DataFrame):
        return ("DataFrame", z.shape, tuple(map(str, z.columns)), tuple(map(str, z.dtypes)),
                z.index.name, _digest(z.index.values), _digest(z.to_numpy()))
    if isinstance(z, pd.Series):
        return ("Series", z.shape, str(z.dtype), _digest(z.index.values), _digest(z.to_numpy()))
    if scipy.sparse.issparse(z):
        d = (type(z).__name__, z.shape, str(z.dtype))
        if z.format == "lil":
            return d + (repr(z.rows.tolist()), repr(z.data.tolist()))
        c = z.tocsr()
        return d + (_digest(c.indptr), _digest(c.indices), _digest(c.data))
    if isinstance(z, np.ndarray):
        return ("ndarray", z.shape, str(z.dtype), _digest(z))
    return (type(z).__name__, repr(z))


class Logged:
    """Wraps a model function; records every call (and lets the scenario draw np.random inside it)."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, z):
        self.calls.append(describe(z))
        return self.fn(z)


def _norm_out(v):
    if isinstance(v, shap.Explanation):
        return ("Explanation", _norm_out(v.values), _norm_out(v.base_values), _norm_out(v.data),
                None if v.feature_names is None else list(v.feature_names))
    if isinstance(v, list):
        return [_norm_out(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_norm_out(x) for x in v)
    if scipy.sparse.issparse(v):
        return describe(v)
    if isinstance(v, (np.ndarray, np.generic, float, int)):
        a = np.asarray(v)
        return ("array", type(v).__name__, a.shape, str(a.dtype), a)
    return v


class _TqdmLog:
    def __init__(self, orig, log):
        self.orig, self.log = orig, log

    def __call__(self, it=None, *a, **kw):
        it = list(it) if it is not None else it
        self.log.append(("tqdm", len(it) if it is not None else None, bool(kw.get("disable", False))))
        return self.orig(it, *a, **kw)


def run(scenario, patched: bool, seed: int = 0):
    """Run scenario() on real shap (patched=False) or patched shap; capture everything observable."""
    shap_rs.unpatch()
    if patched:
        assert shap_rs.patch()
    bars: list = []
    orig_tqdm = sk.tqdm
    sk.tqdm = _TqdmLog(orig_tqdm, bars)
    np.random.seed(seed)
    err = out = None
    stderr = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as w, redirect_stderr(stderr):
            warnings.simplefilter("always")
            try:
                out = scenario()
            except Exception as e:  # noqa: BLE001 - exceptions are part of the compared behaviour
                err = (type(e), str(e))
    finally:
        sk.tqdm = orig_tqdm
        shap_rs.unpatch()
    st = np.random.get_state()
    return {
        "out": _norm_out(out), "err": err,
        "warnings": [(x.category, str(x.message), x.filename, x.lineno) for x in w],
        "rng": (st[1].tobytes(), st[2], st[3], st[4]), "bars": bars, "progress_shown": "%|" in stderr.getvalue(),
    }


def _close(a, b, rtol, atol, path="out"):
    if isinstance(a, tuple) and a and a[0] == "array":
        assert isinstance(b, tuple) and b[0] == "array", f"{path}: {b!r} is not an array"
        assert a[1:4] == b[1:4], f"{path}: type/shape/dtype {a[1:4]} != {b[1:4]}"
        x, y = a[4], b[4]
        if x.dtype.kind in "fc":
            np.testing.assert_allclose(y, x, rtol=rtol, atol=atol, equal_nan=True, err_msg=path)
        else:
            np.testing.assert_array_equal(y, x, err_msg=path)
        return
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), f"{path}: {a!r} != {b!r}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, rtol, atol, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


def assert_same(scenario, seed: int = 0, rtol: float = 1e-9, atol: float = 1e-12, models=(), values_only=False):
    """scenario: zero-arg callable building explainers and returning results. `models`: Logged wrappers used
    inside it (their call logs are compared exactly). Returns the reference run."""
    def once(patched):
        for m in models:
            m.calls = []
        r = run(scenario, patched, seed)
        r["calls"] = [list(m.calls) for m in models]
        return r

    ref = once(False)
    got = once(True)
    assert got["err"] == ref["err"], f"exception differs:\n shap:    {ref['err']}\n patched: {got['err']}"
    _close(ref["out"], got["out"], rtol, atol)
    if values_only:
        return ref
    assert got["warnings"] == ref["warnings"], f"warnings differ:\n shap:    {ref['warnings']}\n patched: {got['warnings']}"
    for i, (a, b) in enumerate(zip(ref["calls"], got["calls"])):
        assert len(a) == len(b), f"model {i}: {len(a)} calls in shap, {len(b)} patched"
        for j, (x, y) in enumerate(zip(a, b)):
            assert x == y, f"model {i} call {j} differs:\n shap:    {x}\n patched: {y}"
    assert got["bars"] == ref["bars"], f"tqdm differs: {ref['bars']} vs {got['bars']}"
    assert got["progress_shown"] == ref["progress_shown"]
    assert got["rng"] == ref["rng"], "global numpy RNG state differs after the call"
    return ref


@pytest.fixture(autouse=True)
def _always_unpatched():
    shap_rs.unpatch()
    yield
    shap_rs.unpatch()
