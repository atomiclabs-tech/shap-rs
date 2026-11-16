# Fuzz harness

The release gate for shap-rs. `gen.py` builds random cases (random data, random model, random explainer settings) and records what **real shap 0.54.1** returns, including the exceptions it raises. `check.py` replays every case with `shap_rs.patch()` active and compares.

While patched, `check.py` makes shap's own compute raise if it is ever called (`shap._cext` tree functions and `KernelExplainer.explain/solve/run`), so a pass means the numbers came from shap-rs.

```bash
pip install shap==0.54.1 shap-rs xgboost lightgbm scikit-learn pandas
cd fuzz
python gen.py 0 2100      # answer key from real shap, written to fuzz/cases/
python check.py           # replay with shap_rs patched
```

Set `SHAP_FUZZ_CASES=/some/dir` to keep answer keys for several environments side by side. The `tests` workflow runs this on Linux, Windows and macOS when started manually with `fuzz: true`.
