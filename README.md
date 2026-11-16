# shap-rs: fast SHAP values, written in Rust

[![PyPI](https://img.shields.io/pypi/v/shap-rs.svg)](https://pypi.org/project/shap-rs/)
[![Python](https://img.shields.io/pypi/pyversions/shap-rs.svg)](https://pypi.org/project/shap-rs/)
[![tests](https://github.com/atomiclabs-tech/shap-rs/actions/workflows/tests.yml/badge.svg)](https://github.com/atomiclabs-tech/shap-rs/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/atomiclabs-tech/shap-rs/blob/main/LICENSE)

**SHAP values taking too long?** Add one line and your existing [shap](https://github.com/shap/shap) code runs **10-80x faster** on XGBoost, LightGBM, CatBoost, random forest and scikit-learn models. Same API, same `Explanation` objects, same plots, same SHAP values.

```python
import shap
import shap_rs

shap_rs.patch()   # shap's TreeExplainer and KernelExplainer now compute in Rust

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)          # unchanged shap code
shap.plots.beeswarm(shap_values)    # every shap plot still works
```

shap-rs is a drop-in Rust engine for the `shap` Python library. It is not a fork and not a rewrite of the API. It replaces only the slow inner loops (TreeSHAP and KernelSHAP), and leaves everything else to shap itself.

## Install

```bash
pip install shap-rs
```

Prebuilt wheels for Linux (x86_64, aarch64), macOS (Apple Silicon, Intel) and Windows (x64), for Python 3.12 and 3.13. No Rust toolchain needed.

shap-rs is verified against **shap 0.54.1** and installs it for you. On any other shap version, `patch()` warns and leaves shap untouched.

## Benchmarks

Measured on clean GitHub Actions runners. Each row is the same model and data, run through real shap 0.54.1 and through shap-rs. The script is [`benchmarks/bench.py`](https://github.com/atomiclabs-tech/shap-rs/blob/main/benchmarks/bench.py), and the `bench` workflow re-runs it on GitHub Actions.

| Workload | shap 0.54.1 | shap-rs | Speedup (5-core Mac) | Speedup (2-core Linux) |
|---|---:|---:|---:|---:|
| XGBoost, 500 trees, interventional, 1,000 rows | 7.6 s | 0.09 s | **82x** | 24x |
| Random forest interaction values, 200 rows | 15.3 s | 0.35 s | **44x** | 13x |
| Random forest regressor, 200 trees, depth 12, 500 rows | 17.7 s | 0.46 s | **38x** | 13x |
| Random forest classifier, 2,000 rows | 7.9 s | 0.27 s | **30x** | 9x |
| LightGBM, 500 trees, interventional, 500 rows | 5.7 s | 0.20 s | **29x** | 10x |
| KernelExplainer, logistic regression, 100 rows | 2.0 s | 1.0 s | 2x | 3x |

Times are from the Mac runner (Apple M2 Pro, 5 cores). shap-rs uses every CPU core, so the gain grows with core count. On a single core it is still 7-28x faster for tree models.

## Are the SHAP values the same?

Yes. Every result is checked against real shap:

- **2,055 randomized test cases** ([fuzz harness](https://github.com/atomiclabs-tech/shap-rs/tree/main/fuzz)) (random data, random model, random explainer settings) match shap on macOS, Linux and Windows.
- Path-dependent TreeSHAP and interaction values match to about 1e-13 (floating point noise).
- The fast interventional path matches to about 6e-8. If you need bit-for-bit output, set `SHAP_RS_NO_FAST=1` for the exact mode.
- Where shap raises an error, shap-rs raises the same error.

## Why is SHAP slow?

TreeSHAP walks every tree for every row, and with interventional (background data) explanations it does that again for every background row. A 500-tree XGBoost model with 100 background rows and 1,000 rows to explain means about 50 million tree walks. shap runs much of this on one CPU core.

shap-rs runs the same algorithms in Rust, uses all your cores (via [rayon](https://github.com/rayon-rs/rayon)), and uses faster algorithms for the interventional case that give the same values.

## What gets faster

**TreeExplainer**, for scikit-learn decision trees, random forests, extra trees, gradient boosting, HistGradientBoosting and IsolationForest, plus XGBoost, LightGBM and CatBoost:

- path-dependent SHAP values
- interventional SHAP values (`raw`, `probability`, `log_loss`)
- SHAP interaction values
- `approximate=True`

**KernelExplainer**: coalition sampling, expected values, l1 feature selection and the weighted least squares all run in Rust. Your model is called exactly as shap calls it, so for slow models most of the time is still your model.

**Not changed** (shap's own code runs): Linear, Permutation, Exact, Partition, Deep and GPUTree explainers.

## Usage

```python
import shap_rs

shap_rs.patch()        # returns True when patched
shap_rs.is_patched()   # True
shap_rs.unpatch()      # puts back shap's original code exactly
```

You can also use the standalone classes without patching:

```python
from shap_rs import TreeExplainer, KernelExplainer

shap_values = TreeExplainer(model).shap_values(X)
```

Full guide: [docs/USAGE.md](https://github.com/atomiclabs-tech/shap-rs/blob/main/docs/USAGE.md).

## FAQ

**How do I speed up SHAP for XGBoost or LightGBM?**
`pip install shap-rs`, then call `shap_rs.patch()` before you build your `TreeExplainer`. Nothing else in your code changes.

**Do I need a GPU?**
No. shap-rs runs on your CPU and uses all of its cores.

**Does it work in Jupyter notebooks?**
Yes. Patch once at the top of the notebook.

**Which shap version do I need?**
shap 0.54.1. The patch depends on shap's internals, so new shap releases need a matching shap-rs release.

## Limits

- Python 3.12 and 3.13 are tested. Python 3.14, PyPy and Windows on ARM are not tested yet.
- Tied to shap 0.54.1 (see above).
- shap's progress line for long interventional runs is not printed.

## About

shap-rs is built by [Atomic Labs](https://atomiclabs.tech). It is an independent project and is not affiliated with the shap maintainers. All credit for SHAP and the shap library goes to Scott Lundberg and the shap contributors.

MIT licensed. Issues and pull requests welcome.
