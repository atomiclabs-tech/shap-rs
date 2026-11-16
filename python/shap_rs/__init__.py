"""shap_rs: Rust port of SHAP's compute engine (TreeSHAP, KernelSHAP) with shap 0.54.1's API.

    import shap, shap_rs
    shap_rs.patch()    # shap.TreeExplainer / shap.Explainer / shap.KernelExplainer now compute in Rust
    shap_rs.unpatch()  # restores shap exactly
"""
from ._kernel import KernelExplainer
from ._patch import SHAP_VERSION, is_patched, patch, unpatch
from ._tree import TreeExplainer

__all__ = ["TreeExplainer", "KernelExplainer", "patch", "unpatch", "is_patched", "SHAP_VERSION"]
