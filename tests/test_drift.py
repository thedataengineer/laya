"""Label-free drift monitoring over a fitted gate.

A monitor that fires spuriously gets muted, which is worse than not having one, so the
false-positive rate is tested here as a first-class property rather than assumed. Pure
NumPy: no weights, no torch.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.conformal import ConformalGate  # noqa: E402
from laya.drift import (  # noqa: E402
    GateMonitor,
    binomial_two_sided_p,
    ks_p_value,
    ks_two_sample,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def ok(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, (": " + detail) if detail else ""))


def raises(name, fn, fragment=None):
    try:
        fn()
    except Exception as exc:                                  # noqa: BLE001
        if fragment is not None and fragment not in str(exc):
            FAIL.append("%s: raised %r, wanted %r" % (name, str(exc), fragment))
        else:
            PASS.append(name)
        return
    FAIL.append("%s: did not raise" % name)


# ------------------------------------------------------------------ exact binomial test
# Against the defining sum: every outcome no more likely than the one observed.
def reference_binom_p(k, n, p):
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return sum(v for v in pmf if v <= pmf[k] * (1 + 1e-9))


worst = 0.0
for n in (10, 40, 100):
    for p in (0.05, 0.3, 0.5, 0.85):
        for k in range(0, n + 1, max(1, n // 12)):
            worst = max(worst, abs(binomial_two_sided_p(k, n, p) - reference_binom_p(k, n, p)))
ok("binom/matches the defining sum", worst < 1e-9, "max deviation %.2e" % worst)

ok("binom/the mode is not significant", binomial_two_sided_p(50, 100, 0.5) > 0.9)
ok("binom/a large departure is", binomial_two_sided_p(90, 100, 0.5) < 1e-10)
ok("binom/is two sided", binomial_two_sided_p(10, 100, 0.5) < 1e-10)
check("binom/degenerate n", binomial_two_sided_p(0, 0, 0.5), 1.0)
check("binom/p=0 with no successes", binomial_two_sided_p(0, 10, 0.0), 1.0)
check("binom/p=0 with a success is impossible", binomial_two_sided_p(1, 10, 0.0), 0.0)
check("binom/p=1 with all successes", binomial_two_sided_p(10, 10, 1.0), 1.0)
ok("binom/never exceeds one",
   all(binomial_two_sided_p(k, 30, 0.4) <= 1.0 + 1e-12 for k in range(31)))
# The extremes an exact test exists for: the normal approximation is worst here.
ok("binom/handles a rate near zero", 0.0 < binomial_two_sided_p(0, 500, 0.01) <= 1.0)
ok("binom/handles a rate near one", 0.0 < binomial_two_sided_p(500, 500, 0.99) <= 1.0)

# ------------------------------------------------------------------ KS
ok("ks/p falls as the statistic grows", ks_p_value(0.3, 500) < ks_p_value(0.05, 500))
ok("ks/p falls as n grows", ks_p_value(0.1, 5000) < ks_p_value(0.1, 50))
check("ks/zero statistic is not evidence", ks_p_value(0.0, 500), 1.0)
check("ks/no data is not evidence", ks_p_value(0.3, 0), 1.0)
ok("ks/stays a probability", 0.0 <= ks_p_value(2.0, 500) <= 1.0)

rng = np.random.default_rng(0)
ref = rng.random(4000)
sketch = {"n": 4000, "levels": list(np.linspace(0, 1, 101)),
          "quantiles": [float(q) for q in np.quantile(ref, np.linspace(0, 1, 101))],
          "mean": float(ref.mean())}
same = ks_two_sample(rng.random(2000), sketch)
ok("ks/same distribution is quiet", same["p_value"] > 0.01,
   "p=%.3g statistic=%.3f" % (same["p_value"], same["statistic"]))
shifted = ks_two_sample(rng.beta(2, 5, 2000), sketch)
ok("ks/a real shift is loud", shifted["p_value"] < 1e-6, "p=%.3g" % shifted["p_value"])
ok("ks/statistic is a proportion", 0.0 <= shifted["statistic"] <= 1.0)
check("ks/no observations is not evidence", ks_two_sample([], sketch)["p_value"], 1.0)
check("ks/no reference is not evidence",
      ks_two_sample([0.5], {"n": 0, "levels": [0.0], "quantiles": [0.0]})["p_value"], 1.0)


# ------------------------------------------------------------------ a fitted gate
KEYS = ["a", "b", "c", "d"]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def gen(n, seed, margin=2.2):
    r = np.random.default_rng(seed)
    results, labels = [], []
    for _ in range(n):
        g = int(r.integers(0, 4))
        lg = r.normal(0, 1.0, 4)
        lg[g] += margin
        p = softmax(lg)
        results.append({"answers": {"q": {
            "type": "choice", "choice": KEYS[int(p.argmax())],
            "probabilities": {k: float(v) for k, v in zip(KEYS, p)},
            "confidence": float(p.max())}}})
        labels.append({"q": KEYS[g]})
    return results, labels


cal_r, cal_l = gen(3000, 1)
GATE = ConformalGate.calibrate(cal_r, cal_l, alpha=0.05, delta=0.05)

ok("sketch/calibration stored a score sketch", "score_sketch" in GATE["q"].diagnostics)
sk = GATE["q"].diagnostics["score_sketch"]
check("sketch/records the calibration size", sk["n"], 3000)
check("sketch/is 101 quantiles", len(sk["quantiles"]), 101)
ok("sketch/quantiles are non-decreasing",
   all(a <= b + 1e-12 for a, b in zip(sk["quantiles"], sk["quantiles"][1:])))
ok("sketch/survives a JSON round trip",
   ConformalGate.from_dict(GATE.to_dict())["q"].diagnostics["score_sketch"] == sk)


def watch(margin, seed, n=1000, gate=GATE):
    results, _ = gen(n, seed, margin)
    m = GateMonitor(gate, window=n)
    m.observe_batch([gate.apply(r) for r in results])
    return m


# ------------------------------------------------------------------ quiet when nothing moved
res = watch(2.2, 500).check()
check("quiet/status", res["status"], "ok")
check("quiet/question status", res["questions"]["q"]["status"], "ok")
ok("quiet/acceptance test does not fire", res["questions"]["q"]["acceptance"]["drifted"] is False)
ok("quiet/score test does not fire", res["questions"]["q"]["scores"]["drifted"] is False)
ok("quiet/observed rate tracks the calibrated one",
   abs(res["questions"]["q"]["acceptance"]["observed_rate"]
       - res["questions"]["q"]["acceptance"]["calibrated_rate"]) < 0.06,
   str(res["questions"]["q"]["acceptance"]))

# The property that decides whether anyone leaves this monitor switched on.
FP_TRIALS = 60
fp = sum(watch(2.2, 3_000 + t).check()["status"] == "expired" for t in range(FP_TRIALS))
ok("quiet/false-positive rate stays inside the test level", fp <= 3,
   "%d/%d clean windows were called expired" % (fp, FP_TRIALS))

# ------------------------------------------------------------------ loud when it did
loud = watch(1.0, 900).check()
check("drift/status", loud["status"], "expired")
q = loud["questions"]["q"]
ok("drift/acceptance test fires", q["acceptance"]["drifted"] is True)
ok("drift/score test fires", q["scores"]["drifted"] is True)
ok("drift/says what moved", "acceptance rate moved" in q["reason"], q.get("reason", ""))
ok("drift/p-values are decisive", q["acceptance"]["p_value"] < 1e-6)

# A shift small enough to be interesting: 2.2 -> 1.8 logit separation.
ok("drift/catches a modest shift", watch(1.8, 901).check()["status"] == "expired")

# ------------------------------------------------------------------ too little traffic
for n, want in ((0, "watching"), (20, "watching"), (99, "watching"), (200, "ok")):
    m = GateMonitor(GATE, window=2000)
    if n:
        results, _ = gen(n, 61)
        m.observe_batch([GATE.apply(r) for r in results])
    check("power/n=%d is %s" % (n, want), m.check()["status"], want)
thin = GateMonitor(GATE)
check("power/an empty monitor is watching", thin.check()["status"], "watching")
ok("power/it says why",
   "before a verdict carries any power" in thin.check()["questions"]["q"]["reason"])

# ------------------------------------------------------------------ ingestion
m = GateMonitor(GATE, window=50)
raw, _ = gen(200, 71)
m.observe_batch(raw)                       # ungated results: the gate is applied for us
check("ingest/window bounds what is retained", len(m._scores["q"]), 50)
check("ingest/counts everything seen", m.n_seen, 200)
ok("ingest/ungated input still yields a verdict", m.check()["questions"]["q"]["n_observed"] == 50)

pre = GateMonitor(GATE, window=200)
pre.observe_batch([GATE.apply(r) for r in raw[:150]])
post = GateMonitor(GATE, window=200)
post.observe_batch(raw[:150])
check("ingest/pre-gated and raw results agree",
      pre.check()["questions"]["q"]["acceptance"]["observed_rate"],
      post.check()["questions"]["q"]["acceptance"]["observed_rate"])

partial = GateMonitor(GATE, window=200)
partial.observe({"answers": {"unrelated": {"type": "noul", "noul": 0.5}}})
check("ingest/an absent question is skipped, not guessed", len(partial._scores["q"]), 0)
check("ingest/it still counts as traffic", partial.n_seen, 1)
raises("ingest/rejects a result with no answers", lambda: m.observe({"answers": 5}),
       "answers")

raises("ctor/rejects a zero window", lambda: GateMonitor(GATE, window=0), "at least 1")
raises("ctor/rejects a degenerate level", lambda: GateMonitor(GATE, alpha_test=0.0),
       "must lie in")
ok("ctor/repr names the size", "1 questions" in repr(GateMonitor(GATE)))

# ------------------------------------------------------------------ every mode
multi_r, multi_l = [], []
r = np.random.default_rng(9)
for _ in range(2000):
    unsafe = bool(r.random() < 0.3)
    pt = float(r.beta(6, 2) if unsafe else r.beta(2, 6))
    lvl = int(r.integers(0, 5))
    slg = r.normal(0, 1.0, 5)
    slg[lvl] += 2.5
    sp = softmax(slg)
    multi_r.append({"answers": {
        "unsafe": {"type": "noul", "noul": pt, "confidence": max(pt, 1 - pt)},
        "sev": {"type": "score", "score": float((np.arange(5) * sp).sum()),
                "legend": {str(i): "l%d" % i for i in range(5)},
                "probabilities": {str(i): float(v) for i, v in enumerate(sp)}},
    }})
    multi_l.append({"unsafe": unsafe, "sev": lvl})

multi_gate = ConformalGate.calibrate(multi_r, multi_l, alpha=0.05, delta=0.05,
                                     mode={"unsafe": "miss", "sev": "interval"})
mm = GateMonitor(multi_gate, window=1000)
mm.observe_batch([multi_gate.apply(x) for x in multi_r[:1000]])
mres = mm.check()
check("modes/miss and interval both report", sorted(mres["questions"]), ["sev", "unsafe"])
check("modes/miss is quiet on its own calibration data",
      mres["questions"]["unsafe"]["status"], "ok")
ok("modes/miss tests its acceptance rate",
   "acceptance" in mres["questions"]["unsafe"])
ok("modes/interval has no acceptance rate to test",
   "acceptance" not in mres["questions"]["sev"])
ok("modes/interval still watches its scores", "statistic" in mres["questions"]["sev"]["scores"])
check("modes/interval is quiet on its own data", mres["questions"]["sev"]["status"], "ok")

# A miss gate that starts seeing far more unsafe traffic must notice.
shift_r = []
r2 = np.random.default_rng(13)
for _ in range(1000):
    pt = float(r2.beta(6, 2) if r2.random() < 0.85 else r2.beta(2, 6))
    shift_r.append({"answers": {"unsafe": {"type": "noul", "noul": pt,
                                           "confidence": max(pt, 1 - pt)}}})
ms = GateMonitor(multi_gate, window=1000)
ms.observe_batch(shift_r)
check("modes/miss notices a prevalence shift",
      ms.check()["questions"]["unsafe"]["status"], "expired")

# ------------------------------------------------------------------ Bonferroni
one = GateMonitor(GATE, alpha_test=0.01).check()["per_test_level"]
two = GateMonitor(multi_gate, alpha_test=0.01).check()["per_test_level"]
ok("bonferroni/level splits across tests", abs(one - 0.005) < 1e-12, "got %r" % one)
ok("bonferroni/and across questions", abs(two - 0.0025) < 1e-12, "got %r" % two)

# ------------------------------------------------------------------ legacy gates
legacy = ConformalGate.from_dict(GATE.to_dict())
del legacy["q"].diagnostics["score_sketch"]
lm = GateMonitor(legacy, window=500)
lres = watch(2.2, 77, gate=legacy).check()
ok("legacy/a gate with no sketch still monitors acceptance",
   "acceptance" in lres["questions"]["q"])
ok("legacy/and says why the score test is missing",
   "refit to enable" in lres["questions"]["q"]["scores"]["unavailable"])
check("legacy/it does not fail closed", lres["status"], "ok")

# ------------------------------------------------------------------ the report
rep = watch(1.0, 902).report()
ok("report/leads with the status", rep.startswith("Laya gate drift  status=expired"))
ok("report/names the question", "q " in rep)
ok("report/shows both tests", "KS=" in rep and "%" in rep)
ok("report/says what to do", "Refit on freshly labelled traffic" in rep)
ok("report/never claims the guarantee still holds",
   "no evidence of expiry, not proof" in rep)
clean = watch(2.2, 903).report()
ok("report/a clean run still carries the caveat", "not proof" in clean)
ok("report/a clean run does not tell anyone to refit",
   "Refit on freshly labelled traffic" not in clean)

# ------------------------------------------------------------------ package surface
import laya  # noqa: E402

ok("export/GateMonitor is in __all__", "GateMonitor" in laya.__all__)
check("export/GateMonitor resolves", laya.GateMonitor, GateMonitor)

import subprocess  # noqa: E402

PROBE = r'''
import sys
sys.path.insert(0, %r)
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked")
        return None
sys.meta_path.insert(0, Blocker())
import laya
laya.GateMonitor
assert "torch" not in sys.modules, "drift monitoring pulled in torch"
print("ok")
''' % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
check("torch-free/monitoring resolves with torch blocked", proc.stdout.strip(), "ok")

# ------------------------------------------------------------------ report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all drift-monitoring tests passed")
sys.exit(1 if FAIL else 0)
