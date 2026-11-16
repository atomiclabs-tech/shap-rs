"""KernelExplainer: patched shap vs real shap, same process, same inputs and seeds (see conftest.assert_same)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import shap
from conftest import Logged, assert_same

rs = np.random.RandomState(0)
X = rs.randn(120, 6)
y_reg = X @ rs.randn(6) + 0.3 * X[:, 0] * X[:, 1]
y_clf = (y_reg > 0).astype(int)
LIN = LinearRegression().fit(X, y_reg)
LOG = LogisticRegression().fit(X, y_clf)
KNN = KNeighborsClassifier(5).fit(X, y_clf)
BG, XE = X[:15], X[100:104]


def kernel(f, bg=BG, link="identity", **ctor):
    return shap.KernelExplainer(f, bg, link=link, **ctor)


# ---------------------------------------------------------------- values, l1_reg, nsamples, links, outputs
L1 = ["num_features(10)", "num_features(3)", "aic", "bic", 0.01, False, 0, "auto"]


@pytest.mark.parametrize("l1", L1)
@pytest.mark.parametrize("nsamples", [None, 30, 200])
def test_l1_nsamples(l1, nsamples):
    f = Logged(LIN.predict)
    kw = {} if nsamples is None else {"nsamples": nsamples}
    assert_same(lambda: kernel(f).shap_values(XE, l1_reg=l1, silent=True, **kw), models=[f])


@pytest.mark.parametrize("l1", ["num_features(4)", "aic", "bic", 0.001, False])
def test_multioutput_logit(l1):
    f = Logged(LOG.predict_proba)
    assert_same(lambda: kernel(f, link="logit").shap_values(XE, l1_reg=l1, nsamples=60, silent=True), models=[f])


@pytest.mark.parametrize("l1", ["num_features(2)", "aic", "bic", 0.05, False])
def test_discrete_model_ties(l1):
    # KNN probabilities are discrete: exact ties and degenerate LARS paths (sklearn warns there)
    f = Logged(KNN.predict_proba)
    assert_same(lambda: kernel(f).shap_values(X[60:66], l1_reg=l1, nsamples=40, silent=True), models=[f])


def test_call_api_explanation():
    f = Logged(LOG.predict_proba)
    df_bg = pd.DataFrame(BG, columns=list("abcdef"))
    df = pd.DataFrame(XE, columns=list("abcdef"))
    assert_same(lambda: kernel(lambda d: f(d), bg=df_bg)(df, silent=True), models=[f])


def test_one_row_1d_and_full_enumeration():
    f = Logged(LIN.predict)
    assert_same(lambda: (kernel(f).shap_values(XE[0], silent=True), kernel(f).shap_values(XE[:2], nsamples=10000)),
                models=[f])


def test_constant_features_m0_m1():
    bg = np.tile(XE[0], (5, 1))
    bg[:, 2] += np.arange(5)
    f = Logged(LIN.predict)
    assert_same(lambda: (kernel(f, bg=bg).shap_values(np.vstack([XE[0], bg[1]]), silent=True),
                         kernel(f, bg=bg[:1]).shap_values(XE[0:1], silent=True)), models=[f])


def test_kmeans_background_and_float32_int_data():
    km = shap.kmeans(X[:60], 6)
    f = Logged(LIN.predict)
    bg32 = BG.astype(np.float32)
    bgi = np.round(BG * 3).astype(np.int64)
    assert_same(lambda: (kernel(f, bg=km).shap_values(XE, nsamples=50, silent=True),
                         kernel(f, bg=bg32).shap_values(XE, nsamples=50, silent=True),
                         kernel(f, bg=bgi).shap_values(np.round(XE * 3), nsamples=50, silent=True)), models=[f])


def test_groups():
    from shap.utils._legacy import DenseData
    groups = [np.array([0, 1]), np.array([2]), np.array([3, 4, 5])]
    dd = DenseData(BG.copy(), ["g01", "g2", "g345"], groups)
    f = Logged(LIN.predict)
    assert_same(lambda: kernel(f, bg=dd).shap_values(XE, nsamples=20, silent=True), models=[f])


def test_object_dtype_data():
    # non-numeric containers run shap's own buffer code; the model sees the same object arrays
    bg = BG.astype(object)
    f = Logged(lambda z: LIN.predict(np.asarray(z, dtype=float)))
    assert_same(lambda: kernel(f, bg=bg).shap_values(XE.astype(object), nsamples=30, silent=True), models=[f])


# ---------------------------------------------------------------- the model receives what shap passes
def test_dataframe_model_input_dtypes():
    df_bg = pd.DataFrame({"a": BG[:, 0], "b": (BG[:, 1] * 10).astype(np.int64), "c": BG[:, 2].astype(np.float32)})
    df = pd.DataFrame({"a": XE[:, 0], "b": (XE[:, 1] * 10).astype(np.int64), "c": XE[:, 2].astype(np.float32)},
                      index=[10, 11, 12, 13])
    f = Logged(lambda z: np.asarray(z, dtype=float) @ np.array([1.0, 0.1, -2.0]))
    assert_same(lambda: kernel(f, bg=df_bg).shap_values(df, nsamples=12, silent=True), models=[f])


def test_model_returning_dataframe_and_wrong_shape():
    good = Logged(lambda z: pd.DataFrame({"p": LIN.predict(z)}))
    bad = Logged(lambda z: np.zeros((len(z) + 1,)) if len(z) > len(BG) else LIN.predict(z))  # only synthetic calls
    assert_same(lambda: kernel(good).shap_values(XE, nsamples=20, silent=True), models=[good])
    assert_same(lambda: kernel(bad).shap_values(XE, nsamples=20, silent=True), models=[bad])


# ---------------------------------------------------------------- RNG interleaving (MC dropout style)
def test_model_using_np_random():
    def noisy(z):
        return LIN.predict(z) + 0.01 * np.random.randn(len(z))
    f = Logged(noisy)
    assert_same(lambda: kernel(f).shap_values(XE, nsamples=40, l1_reg="aic", silent=True), models=[f])
    assert_same(lambda: kernel(f).shap_values(XE, nsamples=40, l1_reg=0.01, silent=True), models=[f])


def test_rng_state_and_later_draws():
    f = Logged(LIN.predict)
    assert_same(lambda: (kernel(f).shap_values(XE, nsamples=33, l1_reg=0.02, silent=True), np.random.rand(3)),
                models=[f])


# ---------------------------------------------------------------- bad arguments: same exception, same point
@pytest.mark.parametrize("l1", ["foo", -1.0, "num_features(x)", "num_features(-1)", True, np.array([1.0]), None,
                                "num_features(0)"])
def test_invalid_l1(l1):
    def noisy(z):
        return LIN.predict(z) + 0.0 * np.random.rand(len(z))
    f = Logged(noisy)
    assert_same(lambda: kernel(f).shap_values(XE, l1_reg=l1, nsamples=20, silent=True), models=[f])


@pytest.mark.parametrize("ns", [0, 1, 2, 8, -3, 2.5, "foo", np.int64(17)])
def test_odd_nsamples(ns):
    bg4 = BG[:, :4]
    lin4 = LinearRegression().fit(X[:, :4], y_reg)
    f = Logged(lin4.predict)
    assert_same(lambda: kernel(f, bg=bg4).shap_values(XE[:, :4], nsamples=ns, l1_reg=False, silent=True),
                models=[f])


def test_input_errors():
    f = Logged(LIN.predict)
    df_bg = pd.DataFrame(BG, columns=list("abcdef"))
    assert_same(lambda: kernel(f).shap_values(XE[None], silent=True), models=[f])
    assert_same(lambda: kernel(f).shap_values(XE[:0], silent=True), models=[f])
    assert_same(lambda: kernel(f, bg=df_bg).shap_values(pd.DataFrame(XE, columns=list("abcdeg")), silent=True),
                models=[f])
    assert_same(lambda: kernel(f).shap_values([list(XE[0])], silent=True), models=[f])


# ---------------------------------------------------------------- warnings, progress bar
def test_singular_warning():
    f = Logged(LIN.predict)
    assert_same(lambda: kernel(f).shap_values(XE, nsamples=5, l1_reg=False, silent=True), models=[f])


def test_auto_deprecation_warning():
    f = Logged(LOG.predict_proba)
    assert_same(lambda: kernel(f).shap_values(XE, l1_reg="auto", nsamples=10, silent=True), models=[f])


@pytest.mark.parametrize("silent", [False, True])
def test_progress_bar(silent):
    f = Logged(LIN.predict)
    ref = assert_same(lambda: kernel(f).shap_values(XE, nsamples=20, silent=silent), models=[f])
    assert ref["progress_shown"] is (not silent)


# ---------------------------------------------------------------- sparse data (all formats), mixed sparse/dense
FORMATS = ["csr_matrix", "csc_matrix", "coo_matrix", "lil_matrix", "dok_matrix", "bsr_matrix", "dia_matrix",
           "csr_array", "csc_array", "coo_array", "lil_array"]
Xs = X.copy()
Xs[np.abs(Xs) < 0.7] = 0.0


def _sp(a, fmt):
    return getattr(scipy.sparse, fmt)(a)


def _sparse_model(z):
    z = z.toarray() if scipy.sparse.issparse(z) else np.asarray(z)
    return LIN.predict(z)


@pytest.mark.parametrize("fmt", FORMATS)
def test_sparse_both(fmt):
    f = Logged(_sparse_model)
    assert_same(lambda: kernel(f, bg=_sp(Xs[:8], fmt)).shap_values(_sp(Xs[100:103], fmt), nsamples=30, silent=True),
                models=[f])


@pytest.mark.parametrize("fmt", FORMATS)
def test_sparse_x_dense_background(fmt):
    f = Logged(_sparse_model)
    assert_same(lambda: kernel(f, bg=Xs[:8]).shap_values(_sp(Xs[100:103], fmt), nsamples=30, silent=True),
                models=[f])


@pytest.mark.parametrize("fmt", FORMATS)
def test_dense_x_sparse_background(fmt):
    f = Logged(_sparse_model)
    assert_same(lambda: kernel(f, bg=_sp(Xs[:8], fmt)).shap_values(Xs[100:103], nsamples=30, silent=True),
                models=[f])


def test_sparse_explicit_zeros_and_float32():
    bg = scipy.sparse.csr_matrix(Xs[:8].astype(np.float32))
    bg.data[::3] = 0.0  # explicit stored zeros
    f = Logged(_sparse_model)
    assert_same(lambda: kernel(f, bg=bg).shap_values(scipy.sparse.csr_matrix(Xs[100:103]), nsamples=30, silent=True),
                models=[f])


# ---------------------------------------------------------------- keep_index
KI_BG = pd.DataFrame(BG[:6, :4], columns=list("abcd"), index=pd.Index([5, 3, 9, 1, 7, 2], name="id"))
KI_X = pd.DataFrame(XE[:, :4], columns=list("abcd"), index=pd.Index([40, 30, 20, 10], name="id"))
LIN4 = LinearRegression().fit(X[:, :4], y_reg)


def _df_model(z):
    return LIN4.predict(np.asarray(z, dtype=float))


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("kind", ["df", "series", "ndarray", "1d", "df_row"])
def test_keep_index(kind, ordered):
    f = Logged(_df_model)
    inp = {"df": KI_X, "series": KI_X.iloc[0], "ndarray": KI_X.values, "1d": KI_X.values[0],
           "df_row": KI_X.iloc[[1]]}[kind]
    assert_same(lambda: kernel(f, bg=KI_BG, keep_index=True, keep_index_ordered=ordered)
                .shap_values(inp, nsamples=12, silent=True), models=[f])


def test_explain_called_directly():
    f = Logged(LIN.predict)
    assert_same(lambda: (kernel(f).explain(XE[1], nsamples=25), kernel(f).explain(XE[1:2], l1_reg="bic")), models=[f])


# ---------------------------------------------------------------- more shapes of use
def test_custom_link():
    from shap.utils._legacy import Link

    class Cube(Link):
        def __init__(self):
            self.__name__ = "cube"

        @staticmethod
        def f(x):
            return np.asarray(x) ** 3

        @staticmethod
        def finv(x):
            return np.cbrt(x)

    f = Logged(LIN.predict)
    assert_same(lambda: kernel(f, link=Cube()).shap_values(XE, nsamples=40, l1_reg="num_features(4)", silent=True),
                models=[f])


def test_many_features_sampled():
    rs40 = np.random.RandomState(3)
    X40 = rs40.randn(60, 40)
    w40 = rs40.randn(40)
    f = Logged(lambda z: np.tanh(z @ w40))
    assert_same(lambda: kernel(f, bg=X40[:10]).shap_values(X40[50:53], nsamples=300, silent=True), models=[f])
    assert_same(lambda: kernel(f, bg=X40[:10]).shap_values(X40[50:52], silent=True), models=[f])


def test_multioutput_lasso_with_noisy_model_reused_explainer():
    def noisy(z):
        return LOG.predict_proba(z) + 1e-3 * np.random.rand(len(z), 2)
    f = Logged(noisy)

    def scenario():
        e = kernel(f)
        a = e.shap_values(XE[:2], nsamples=50, l1_reg=0.002, silent=True, gc_collect=True)
        b = e.shap_values(XE[2:], nsamples=50, l1_reg="bic", silent=True)
        return a, b, e.fx, e.nsamples, e.M, e.maskMatrix, e.kernelWeights, e.ey, e.y, e.varyingInds
    assert_same(scenario, models=[f])
