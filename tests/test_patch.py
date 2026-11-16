"""patch() / unpatch() mechanics, the version guard, and the wheel smoke script."""
import subprocess
import sys
import warnings
from pathlib import Path

import shap
import shap.explainers._kernel as sk
import shap.explainers._tree as st
import shap_rs


def test_patch_unpatch_restores_originals():
    before = (st._cext, st.TreeEnsemble.__dict__["__init__"], sk.KernelExplainer.__dict__["explain"],
              sk.KernelExplainer.__dict__["shap_values"])
    assert shap_rs.patch() and shap_rs.is_patched()
    assert st._cext is not before[0] and sk.KernelExplainer.__dict__["explain"] is not before[2]
    assert sk.KernelExplainer.__dict__["shap_values"] is before[3]  # shap's own row loop stays
    assert shap_rs.patch()  # idempotent
    shap_rs.unpatch()
    after = (st._cext, st.TreeEnsemble.__dict__["__init__"], sk.KernelExplainer.__dict__["explain"],
             sk.KernelExplainer.__dict__["shap_values"])
    assert all(a is b for a, b in zip(before, after)) and not shap_rs.is_patched()


def test_version_guard(monkeypatch):
    monkeypatch.setattr(shap, "__version__", "0.99.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert shap_rs.patch() is False
    assert not shap_rs.is_patched()
    assert any(issubclass(x.category, RuntimeWarning) and "verified against shap 0.54.1" in str(x.message) for x in w)
    assert shap_rs.patch(force=True) is True
    shap_rs.unpatch()


def test_smoke_script():
    smoke = Path(__file__).resolve().parents[1] / "extra" / "smoke.py"
    r = subprocess.run([sys.executable, str(smoke)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
