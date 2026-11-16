"""Generate fuzz cases with REAL shap 0.54.1 (the answer key that check.py replays against shap_rs).
Usage: python gen.py START END   (seeds START..END-1, one pickle per case in cases/)"""
import pickle, sys, time, traceback, warnings
warnings.filterwarnings("ignore")
import shap
from fuzzlib import CASES, make_case, run_case

CASES.mkdir(exist_ok=True)
a, b = int(sys.argv[1]), int(sys.argv[2])
for seed in range(a, b):
    p = CASES / f"{seed:05d}.pkl"
    if p.exists():
        continue
    try:
        spec, model, Xbg, Xe = make_case(seed)
    except Exception as e:  # invalid random config (e.g. model can't fit): skip the seed
        print(seed, "SKIP", type(e).__name__, str(e)[:80], flush=True); continue
    t = time.perf_counter()
    try:
        out = run_case(shap, spec, model, Xbg, Xe)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"}
    dt = time.perf_counter() - t
    pickle.dump({"spec": spec, "model": model, "Xbg": Xbg, "Xe": Xe, "expected": out, "shap_s": dt}, open(p, "wb"))
    print(seed, spec["kind"], spec["family"], spec.get("mode", ""), spec["api"], "ERR" if "error" in out else "ok", f"{dt:.2f}s", flush=True)
