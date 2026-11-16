"""Replay every fuzz case through real shap's public API with shap_rs.patch() active.

Guards: shap's own compute paths are replaced with functions that raise BEFORE shap_rs is imported,
so a patch that silently falls back to shap's original compute fails loudly:
  - shap._cext.dense_tree_shap / dense_tree_saabas  (all of shap's TreeSHAP compute)
  - shap.KernelExplainer.explain / solve / run      (KernelSHAP compute)
Delegation to XGBoost/LightGBM's own native SHAP (pred_contribs / pred_contrib), as shap does, is allowed.

PASS = same outputs as real shap (see fuzzlib.compare), or the same exception type where shap raised.
Usage: python check.py [--only SUBSTR] [--first N] [--show] [--seed S]
"""
import argparse, json, pickle, sys, time, traceback, warnings
from collections import Counter
warnings.filterwarnings("ignore")
import numpy as np
import shap
import shap._cext as C
from shap.explainers._kernel import KernelExplainer as KE
from fuzzlib import H, CASES, run_case, compare

ap = argparse.ArgumentParser(); ap.add_argument("--only"); ap.add_argument("--first", type=int)
ap.add_argument("--show", action="store_true"); ap.add_argument("--seed", type=int); a = ap.parse_args()


def guard(name):
    def f(*args, **kw):
        raise RuntimeError(f"GUARD: shap's original {name} was called while patched")
    return f


for n in ("dense_tree_shap", "dense_tree_saabas"):
    setattr(C, n, guard(f"_cext.{n}"))
for n in ("explain", "solve", "run"):
    setattr(KE, n, guard(f"KernelExplainer.{n}"))
before = {n: getattr(shap.TreeExplainer, n) for n in ("shap_values", "shap_interaction_values", "__call__")}

import shap_rs  # noqa: E402
shap_rs.patch()

files = sorted(CASES.glob("*.pkl"))
if a.seed is not None:
    files = [CASES / f"{a.seed:05d}.pkl"]
res, by_group, fails = [], Counter(), Counter()
t_rs = t_shap = 0.0
for p in files:
    c = pickle.load(open(p, "rb"))
    s = c["spec"]
    label = f"{s['seed']:05d} {s['kind']}/{s['family']}/{s.get('mode', s.get('l1_reg'))}/{s['api']}{'/pd' if s['pandas'] else ''}"
    if a.only and a.only not in label:
        continue
    t = time.perf_counter()
    try:
        got = run_case(shap, s, c["model"], c["Xbg"], c["Xe"])
    except Exception as e:
        got = {"error": f"{type(e).__name__}: {e}"}
        if a.show:
            traceback.print_exc()
    dt = time.perf_counter() - t
    ok, msg = compare(c["expected"], got)
    if "GUARD" in str(got.get("error", "")):
        ok, msg = False, got["error"][:160]
    group = f"{s['kind']}/{s['family']}/{s.get('mode', 'kernel')}"
    by_group[(group, ok)] += 1
    if not ok:
        fails[msg.split(" ")[0] + " " + msg.split(" ")[1] if " " in msg else msg] += 1
    t_rs += dt; t_shap += c["shap_s"]
    res.append({"case": label, "ok": ok, "detail": msg, "rs_s": round(dt, 4), "shap_s": round(c["shap_s"], 4)})
    if a.show or not ok:
        print(f"{'PASS' if ok else 'FAIL'} {label}  {msg}  rs {dt:.3f}s shap {c['shap_s']:.3f}s", flush=True)
    if a.first and len(res) >= a.first:
        break

shap_rs.unpatch()
restored = all(getattr(shap.TreeExplainer, n) is f for n, f in before.items())
npass = sum(r["ok"] for r in res)
print("\nper group (pass/total):")
for g in sorted({g for g, _ in by_group}):
    print(f"  {g}: {by_group[(g, True)]}/{by_group[(g, True)] + by_group[(g, False)]}")
print("top failure kinds:", fails.most_common(8))
print(f"unpatch restores shap.TreeExplainer: {restored}")
print(f"total time: shap {t_shap:.1f}s  shap_rs-patched {t_rs:.1f}s")
print(f"PASS {npass}/{len(res)}")
if not (a.only or a.first or a.seed is not None):
    json.dump({"pass": npass, "total": len(res), "unpatch_ok": restored, "cases": res}, open(H / ("results.json" if CASES == H / "cases" else f"results-{CASES.name}.json"), "w"), indent=0)
