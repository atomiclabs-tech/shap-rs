# shap_rs usage

shap_rs runs the compute behind the real `shap` package (TreeSHAP, KernelSHAP) in Rust. You keep using shap's API,
`Explanation` objects and plots. Only the slow inner loops change.

## Install

```bash
pip install shap_rs          # pulls in shap==0.54.1; wheels: Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64
# from source (needs a Rust toolchain): pip install maturin && maturin develop --release
```

shap_rs is verified against **shap 0.54.1 only** (numpy >= 2, CPython 3.12+).

## patch() / unpatch()

```python
import shap, shap_rs

shap_rs.patch()                       # returns True when patched
explainer = shap.TreeExplainer(model, background, feature_perturbation="interventional")
sv = explainer(X)                     # same shap API and Explanation, Rust compute
shap.plots.beeswarm(sv)

shap_rs.unpatch()                     # puts back shap's original objects (identity-equal)
shap_rs.is_patched()                  # -> False
```

- `patch()` is idempotent. With any shap version other than 0.54.1 it warns (`RuntimeWarning`), leaves shap
  untouched and returns False. `shap_rs.patch(force=True)` patches anyway, but exact agreement is then unverified.
- The standalone `shap_rs.TreeExplainer` / `shap_rs.KernelExplainer` classes also exist with shap's signatures.
  Prefer `patch()`: it keeps every shap code path, and the standalone `KernelExplainer` batches model calls across
  rows (fewer, larger calls than shap makes).

## What is accelerated, and by how much

All numbers are from an M4 Mac mini with all cores; the "1 core" column shows how much comes from parallelism.

| Workload | shap 0.54.1 | shap_rs | 1 core |
|---|---|---|---|
| RandomForest (200 trees, depth 12) path-dependent, 500 rows | 62 s | 1.3 s | 7 s |
| RandomForest classifier interactions, 200 rows | 51–80 s | 1.0 s | 5.4 s |
| XGBoost 500 trees interventional, bg 100, 1000 rows | 29–39 s | 0.28 s | 1.1 s |
| LightGBM 500 trees interventional, bg 100, 500 rows | 21–68 s | 0.6–0.8 s | 2.6 s |
| CatBoost 200 trees interventional, bg 100, 200 rows (raw / probability) | 3.8 / 6.6 s | 0.24 / 0.62 s | |
| KernelExplainer, logistic regression, kmeans bg 50, 100 rows, sampled, seeded (diff 1e-15) | 5.2 s | 1.1 s | |

- **TreeExplainer** (sklearn trees/forests/GBMs/HistGB/IsolationForest, XGBoost, LightGBM, CatBoost): path-dependent,
  interventional (`raw`, `probability`, `log_loss`), interaction values and `approximate=True` all run in Rust.
  As in shap, XGBoost/LightGBM/CatBoost path-dependent values come from the libraries' own `pred_contribs` /
  `ShapValues`.
- **KernelExplainer**: coalition sampling, synthetic data, expected values, l1 feature selection and the weighted
  least squares run in Rust. Your model is called exactly as shap calls it: once on x and once on all synthetic
  rows per explained row, with the same container type (ndarray in the background's dtype, `lil_matrix` for sparse
  backgrounds, an indexed DataFrame with `keep_index=True`), the same rows in the same order. If your model uses
  `np.random` (MC dropout, for example), it sees the same random numbers as under shap. Kernel time is usually
  dominated by your model, so the gain depends on how fast your model is (4.6x above, with a cheap model).
- **Not accelerated** (shap's own code runs): Linear, Permutation, Exact, Partition, Deep and GPUTree explainers,
  and pyspark models. For the model-agnostic explainers, the time is spent in your model anyway.

## Environment variables

| Variable | Effect |
|---|---|
| `RAYON_NUM_THREADS=n` | Number of threads for the Rust compute (default: all cores). Must be set before the first shap_rs call in the process. |
| `SHAP_RS_L1=rust\|sklearn` | Kernel `l1_reg` feature selection (LARS, LassoLarsIC, Lasso). Default: Rust on arm64 (macOS, Linux), where it reproduces numpy's BLAS summation order; sklearn elsewhere (x86-64), which is what shap itself calls. Either way, any selection where sklearn would warn or raise is run by sklearn, so its warnings and errors are unchanged. |
| `SHAP_RS_NO_FAST=1` | Use the faithful ports of shap's algorithms instead of the fast ones. Interventional values then match shap to ≤ 1e-16 instead of ≤ ~4e-8, at 3–7x the time (still much faster than shap). |

## Known differences from shap 0.54.1

- **Interventional TreeSHAP values differ by up to ~4e-8 (absolute, typical values ~1).** shap sums the per-reference
  terms of each tree in float32 at every split where x and the reference diverge. The fast Rust paths sum the same
  float32 terms per leaf, in a different order. Being exact needs work for each distinct reference (27 of 100 at
  depth 6, 88 of 100 at depth 12), which made it 3–7x slower. Set `SHAP_RS_NO_FAST=1` if you need shap's exact
  float32 rounding. Path-dependent values and interactions agree to ~1e-13.
- **Long interventional runs:** after 10 s, shap's C++ prints a time-based text progress bar to stderr. shap_rs
  prints nothing.
- **X narrower than the model:** if X (or the background) has fewer columns than the trees use, shap reads out of
  bounds and returns undefined values. shap_rs raises `ValueError`.
- **Platforms:** exact Kernel `l1_reg` agreement depends on the BLAS summation order at exact LARS ties. It has been
  verified on macOS arm64, Linux arm64/x86-64, Windows x64 and Intel macOS, but not on Windows on ARM or musl Linux.
- Explainer objects carry shap's per-row attributes after `explain` (`maskMatrix`, `kernelWeights`, `synth_data`,
  `y`, `ey`, ...). `lastMask` is always zeros, as in shap.

## Testing

```bash
cd shap-rs && python -m pytest -q tests
```

Every test runs a scenario on real (unpatched) shap and then patched, in the same process, and compares values,
types, exceptions, warnings, the model's calls, the progress bar and the numpy RNG state.
