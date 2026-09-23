"""Distribution-free risk control: the bound, the walk, the gate, and its guarantees.

No model weights and no torch. `taut.conformal` is pure NumPy by design -- a gate is
fitted, serialised and applied without loading anything -- so these tests exercise the
real code paths against synthetic Taut answers.

The statistical checks here are smoke-sized so CI stays fast. The full validation, 1000
trials per configuration scored against population risk rather than a sampled estimate,
lives in `research/conformal/validate_risk.py`.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taut.conformal import (  # noqa: E402
    ConformalGate,
    QuestionGate,
    binomial_upper_bound,
    conformal_quantile,
    min_calibration_size,
    miss_threshold,
    selective_threshold,
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
    except Exception as exc:                                  # noqa: BLE001 - that is the point
        if fragment is not None and fragment not in str(exc):
            FAIL.append("%s: raised %r, expected a message containing %r" % (name, str(exc), fragment))
        else:
            PASS.append(name)
        return
    FAIL.append("%s: did not raise" % name)


# ------------------------------------------------------------------ the exact bound
# Clopper-Pearson: the upper limit is the p solving P(Bin(n, p) <= k) = delta. Check the
# returned value against that defining equation rather than against a table, so the test
# fails if the implementation drifts towards a normal or Hoeffding approximation.
def binom_cdf(k, n, p):
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    return sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k + 1))


worst = 0.0
for n in (10, 25, 59, 100):
    # k == n is excluded on purpose: every p satisfies P(Bin(n, p) <= n) = 1, so the
    # equation has no interior root and the only honest bound is the vacuous 1.0,
    # asserted separately below.
    for k in range(0, min(n - 1, 12) + 1):
        for delta in (0.01, 0.05, 0.10):
            u = binomial_upper_bound(k, n, delta)
            worst = max(worst, abs(binom_cdf(k, n, u) - delta))
ok("bound/solves P(Bin(n,p) <= k) = delta", worst < 1e-6, "max residual %.2e" % worst)

check("bound/k == n is vacuous", binomial_upper_bound(7, 7, 0.05), 1.0)
ok("bound/zero errors still positive", 0.0 < binomial_upper_bound(0, 59, 0.05) <= 0.05)
ok("bound/monotone in k",
   all(binomial_upper_bound(k, 100, 0.05) <= binomial_upper_bound(k + 1, 100, 0.05)
       for k in range(0, 20)))
ok("bound/tightens as n grows",
   binomial_upper_bound(0, 500, 0.05) < binomial_upper_bound(0, 59, 0.05))

# ------------------------------------------------------------------ certifiability floor
# n >= log(delta)/log(1-alpha): the point below which a flawless record still cannot
# certify alpha, so the gate must say "not enough data" rather than return a threshold.
check("floor/alpha=0.05 delta=0.05", min_calibration_size(0.05, 0.05), 59)
check("floor/alpha=0.02 delta=0.05", min_calibration_size(0.02, 0.05), 149)
check("floor/alpha=0.01 delta=0.05", min_calibration_size(0.01, 0.05), 299)
for alpha in (0.01, 0.02, 0.05, 0.10):
    n = min_calibration_size(alpha, 0.05)
    ok("floor/%g is exactly the crossing point" % alpha,
       binomial_upper_bound(0, n, 0.05) <= alpha < binomial_upper_bound(0, n - 1, 0.05))
raises("floor/rejects alpha out of range", lambda: min_calibration_size(0.0, 0.05))
raises("floor/rejects delta out of range", lambda: min_calibration_size(0.05, 1.0))

# ------------------------------------------------------------------ the fixed-sequence walk
rng = np.random.default_rng(7)


def draw_selective(n, seed, link=lambda p: p):
    r = np.random.default_rng(seed)
    s = 0.25 + 0.75 * r.beta(5.0, 1.5, size=n)
    return s, r.random(n) < link(s)


scores, correct = draw_selective(2000, 11)
fit = selective_threshold(scores, correct, alpha=0.05, delta=0.05)
ok("walk/certifies on ample data", fit["certifiable"] is True)
ok("walk/bound respects the budget", fit["risk_bound"] <= 0.05,
   "bound %.4f" % fit["risk_bound"])
ok("walk/empirical joint risk is under the bound",
   fit["empirical_joint_risk"] <= fit["risk_bound"] + 1e-12)
ok("walk/threshold lies on the pre-specified grid",
   abs(fit["threshold"] * 200 - round(fit["threshold"] * 200)) < 1e-9,
   "threshold %r is not a multiple of 0.005" % fit["threshold"])
ok("walk/accepts a real share of traffic", 0.0 < fit["coverage"] <= 1.0)
ok("walk/joint risk caps conditional at alpha/coverage",
   fit["empirical_selective_risk"] <= 0.05 / fit["coverage"] + 1e-9)

# Tightening alpha may never buy more coverage: the guarantee is monotone in what it costs.
prev_cov, prev_tau = 1.01, -1.0
for alpha in (0.20, 0.10, 0.05, 0.02, 0.01):
    f = selective_threshold(scores, correct, alpha=alpha, delta=0.05)
    ok("walk/coverage is non-increasing in strictness at alpha=%g" % alpha,
       f["coverage"] <= prev_cov + 1e-12, "%.4f then %.4f" % (prev_cov, f["coverage"]))
    ok("walk/threshold is non-decreasing in strictness at alpha=%g" % alpha,
       f["threshold"] >= prev_tau - 1e-12)
    prev_cov, prev_tau = f["coverage"], f["threshold"]

# Below the floor nothing is certifiable, and the gate has to say why rather than guess.
tiny_s, tiny_c = draw_selective(40, 13)
tiny = selective_threshold(tiny_s, tiny_c, alpha=0.05, delta=0.05)
ok("shortfall/not certifiable below the floor", tiny["certifiable"] is False)
check("shortfall/abstains on everything", tiny["threshold"], float("inf"))
check("shortfall/coverage is zero", tiny["coverage"], 0.0)
ok("shortfall/reports the floor it missed",
   tiny["shortfall"]["min_calibration"] == 59 and tiny["shortfall"]["n_calibration"] == 40)
ok("shortfall/explains itself", "at least 59" in tiny["shortfall"]["reason"],
   tiny["shortfall"]["reason"])

raises("walk/rejects mismatched lengths",
       lambda: selective_threshold(np.zeros(5), np.zeros(4, dtype=bool), 0.05),
       "same length")
raises("walk/rejects an empty calibration set",
       lambda: selective_threshold(np.zeros(0), np.zeros(0, dtype=bool), 0.05))

# A perfect model should be gated at the most permissive grid point, not a cautious one.
perfect = selective_threshold(np.full(500, 0.99), np.ones(500, dtype=bool), 0.05, 0.05)
check("walk/perfect record keeps everything", perfect["coverage"], 1.0)
check("walk/perfect record sits at grid floor", perfect["threshold"], 0.0)

# ------------------------------------------------------------------ selective validity
# The contract is P(true joint risk > alpha) <= delta. Labels here are drawn from the
# model's own probabilities, so true risk at a threshold is a population quantity with no
# label noise: E[1{p >= tau} * (1 - p)] over a large frozen pool.
pool = 0.25 + 0.75 * np.random.default_rng(101).beta(5.0, 1.5, size=200_000)
pool_wrong = 1.0 - pool
TRIALS, ALPHA = 120, 0.05
breaches = 0
for t in range(TRIALS):
    s, c = draw_selective(600, 5_000 + t)
    tau = selective_threshold(s, c, alpha=ALPHA, delta=0.05)["threshold"]
    r_true = 0.0 if not np.isfinite(tau) else float(np.mean((pool >= tau) * pool_wrong))
    breaches += r_true > ALPHA
# 120 trials at delta=0.05 expects 6; allow to the 99.9% binomial tail so CI does not flake.
ok("validity/selective breaches stay inside delta", breaches <= 16,
   "%d/%d trials exceeded alpha" % (breaches, TRIALS))

# ------------------------------------------------------------------ miss control
def draw_miss(n, seed, prevalence=0.25):
    r = np.random.default_rng(seed)
    y = r.random(n) < prevalence
    return np.where(y, r.beta(6.0, 2.0, size=n), r.beta(2.0, 6.0, size=n)), y


ms, my = draw_miss(3000, 21)
mfit = miss_threshold(ms, my, alpha=0.05, delta=0.05)
ok("miss/certifies on ample positives", mfit["certifiable"] is True)
ok("miss/bound respects the budget", mfit["risk_bound"] <= 0.05)
ok("miss/denominator is the positive class", mfit["n_positive"] == int(my.sum()))
ok("miss/blocks a real share", 0.0 < mfit["block_rate"] <= 1.0)
ok("miss/a stricter budget blocks more",
   miss_threshold(ms, my, 0.01, 0.05)["block_rate"]
   >= miss_threshold(ms, my, 0.10, 0.05)["block_rate"])
raises("miss/rejects a calibration set with no positives",
       lambda: miss_threshold(np.linspace(0, 1, 50), np.zeros(50, dtype=bool), 0.05),
       "at least one positive")

# Too few positives to certify: block everything rather than claim a bound.
few_s, few_y = draw_miss(200, 23, prevalence=0.05)
few = miss_threshold(few_s, few_y, alpha=0.01, delta=0.05)
ok("miss/uncertifiable blocks everything",
   few["certifiable"] is False and few["threshold"] == 0.0 and few["block_rate"] == 1.0)

# ------------------------------------------------------------------ conformal quantile
q, feasible = conformal_quantile(np.linspace(0, 1, 1000), 0.1)
ok("quantile/feasible on ample data", feasible is True and 0.0 < q < 1.0)
q2, feasible2 = conformal_quantile(np.linspace(0, 1, 5), 0.01)
ok("quantile/infeasible returns the trivial answer",
   feasible2 is False and q2 == float("inf"))
check("quantile/empty is infeasible", conformal_quantile(np.zeros(0), 0.1)[1], False)
ok("quantile/carries the (n+1) correction",
   conformal_quantile(np.arange(100.0), 0.1)[0] >= np.quantile(np.arange(100.0), 0.9))


# ------------------------------------------------------------------ synthetic Taut answers
def choice_answer(p, keys):
    return {"type": "choice", "choice": keys[int(np.argmax(p))],
            "probabilities": {k: float(v) for k, v in zip(keys, p)},
            "confidence": float(p.max())}


def noul_answer(p_true):
    return {"type": "noul", "noul": float(p_true),
            "confidence": float(max(p_true, 1.0 - p_true))}


def score_answer(p):
    return {"type": "score", "score": float((np.arange(p.size) * p).sum()),
            "legend": {str(i): "lvl%d" % i for i in range(p.size)},
            "probabilities": {str(i): float(v) for i, v in enumerate(p)}}


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


KEYS = ["billing", "technical", "account", "other"]


def make_dataset(n, seed, margin=2.2):
    r = np.random.default_rng(seed)
    results, labels = [], []
    for _ in range(n):
        gold = int(r.integers(0, len(KEYS)))
        lg = r.normal(0, 1.0, len(KEYS))
        lg[gold] += margin
        p = softmax(lg)

        unsafe = bool(r.random() < 0.25)
        p_true = float(r.beta(6, 2) if unsafe else r.beta(2, 6))

        sev_gold = int(r.integers(0, 5))
        slg = r.normal(0, 1.0, 5)
        slg[sev_gold] += 2.5

        results.append({"answers": {
            "intent": choice_answer(p, KEYS),
            "unsafe": noul_answer(p_true),
            "severity": score_answer(softmax(slg)),
        }})
        labels.append({"intent": KEYS[gold], "unsafe": unsafe, "severity": sev_gold})
    return results, labels


cal_r, cal_l = make_dataset(1200, 31)
test_r, test_l = make_dataset(3000, 32)

# ------------------------------------------------------------------ auto mode selection
auto = ConformalGate.calibrate(cal_r, cal_l, alpha=0.05, delta=0.05)
check("auto/choice becomes selective", auto["intent"].mode, "selective")
check("auto/noul becomes selective", auto["unsafe"].mode, "selective")
check("auto/score becomes interval", auto["severity"].mode, "interval")
check("auto/gate covers every labelled question", len(auto), 3)

# Per-question modes, including the guardrail control on the noul question.
gate = ConformalGate.calibrate(
    cal_r, cal_l, alpha=0.05, delta=0.05,
    mode={"intent": "set", "unsafe": "miss", "severity": "interval"})
check("mode-map/intent", gate["intent"].mode, "set")
check("mode-map/unsafe", gate["unsafe"].mode, "miss")
check("mode-map/severity", gate["severity"].mode, "interval")

raises("mode/miss rejects a choice question",
       lambda: ConformalGate.calibrate(cal_r, cal_l, mode={"intent": "miss"},
                                       questions=["intent"]),
       "noul questions")
raises("mode/interval rejects a noul question",
       lambda: ConformalGate.calibrate(cal_r, cal_l, mode={"unsafe": "interval"},
                                       questions=["unsafe"]),
       "score questions")
raises("mode/rejects an unknown name",
       lambda: QuestionGate("q", "choice", "nonsense", 0.05, 0.05, KEYS, {}),
       "not one of")

# ------------------------------------------------------------------ fresh-split coverage
def set_coverage(method, alpha):
    g = ConformalGate.calibrate(cal_r, cal_l, alpha=alpha, mode={"intent": "set"},
                                set_method=method, questions=["intent"])
    hits, sizes = 0, 0
    for r_, l_ in zip(test_r, test_l):
        blk = g.apply(r_)["answers"]["intent"]["gate"]
        hits += l_["intent"] in blk["prediction_set"]
        sizes += blk["set_size"]
    return hits / len(test_r), sizes / len(test_r)


for method in ("lac", "aps"):
    for alpha in (0.10, 0.05):
        cov, size = set_coverage(method, alpha)
        # Split conformal is valid, never exact: LAC lands on target, APS over-covers
        # because the unrandomised set includes the option that crosses qhat.
        ok("set/%s covers at alpha=%g" % (method, alpha), cov >= 1.0 - alpha - 0.02,
           "coverage %.4f against target %.2f" % (cov, 1.0 - alpha))
        ok("set/%s sets stay usable at alpha=%g" % (method, alpha), 1.0 <= size <= len(KEYS),
           "mean size %.2f" % size)

ival = ConformalGate.calibrate(cal_r, cal_l, alpha=0.10, mode={"severity": "interval"},
                               questions=["severity"])
hits = 0
for r_, l_ in zip(test_r, test_l):
    lo, hi = ival.apply(r_)["answers"]["severity"]["gate"]["interval"]
    hits += lo <= l_["severity"] <= hi
ok("interval/covers at alpha=0.10", hits / len(test_r) >= 0.88,
   "coverage %.4f" % (hits / len(test_r)))
ok("interval/half width is finite and useful",
   0.0 < ival["severity"].params["half_width"] < 4.0)
ok("interval/point estimate matches the answer's own expected score",
   abs(ival.apply(test_r[0])["answers"]["severity"]["gate"]["point"]
       - test_r[0]["answers"]["severity"]["score"]) < 1e-6)

# Infeasible interval: fall back to the range that always covers.
few_r, few_l = make_dataset(5, 41)
few_gate = ConformalGate.calibrate(few_r, few_l, alpha=0.01, mode={"severity": "interval"},
                                   questions=["severity"])
ok("interval/infeasible is flagged", few_gate["severity"].diagnostics["feasible"] is False)
check("interval/infeasible covers the full range",
      few_gate.apply(test_r[0])["answers"]["severity"]["gate"]["interval"], [0.0, 4.0])

# ------------------------------------------------------------------ applying a gate
applied = gate.apply(test_r[0])
ok("apply/does not mutate its input", "gate" not in test_r[0])
ok("apply/attaches a block to every gated answer",
   all("gate" in applied["answers"][q] for q in ("intent", "unsafe", "severity")))
ok("apply/keeps the original answer fields",
   applied["answers"]["intent"]["choice"] == test_r[0]["answers"]["intent"]["choice"])
check("apply/counts what it gated", applied["gate"]["questions_gated"], 3)
check("apply/counts what it did not", applied["gate"]["questions_ungated"], 0)

ungated = {"answers": {"intent": test_r[0]["answers"]["intent"],
                       "mystery": noul_answer(0.9)}}
passthrough = gate.apply(ungated)
ok("apply/ungated questions pass through untouched",
   "gate" not in passthrough["answers"]["mystery"])
check("apply/counts the ungated question", passthrough["gate"]["questions_ungated"], 1)
raises("apply/strict catches question drift", lambda: gate.apply(ungated, strict=True),
       "no gate fitted")

batch = gate.apply_batch(test_r[:5])
check("apply_batch/returns one result per input", len(batch), 5)
ok("apply_batch/matches apply", batch[0] == gate.apply(test_r[0]))

sel = ConformalGate.calibrate(cal_r, cal_l, alpha=0.05, delta=0.05, questions=["intent"])
blk = sel.apply(test_r[0])["answers"]["intent"]["gate"]
ok("accepts/agrees with the gate block",
   sel.accepts(test_r[0], "intent") == blk["accepted"])
check("accepts/ungated question accepts", sel.accepts(test_r[0], "unsafe"), True)
raises("accepts/raises on a missing answer",
       lambda: sel.accepts({"answers": {}}, "intent"), "no answer")
ok("selective/abstain is the complement of accepted", blk["abstain"] is not blk["accepted"])
ok("selective/decision matches the threshold",
   blk["accepted"] == (blk["top_probability"] >= blk["threshold"]))

# ------------------------------------------------------------------ the family-wise bound
# Accepting a record is not the per-question guarantee: five gates at 2% union to 10%.
fam = gate.family_risk()
ok("family/alpha unions over gated questions", abs(fam["family_alpha"] - 0.15) < 1e-12,
   "got %r" % fam["family_alpha"])
# Only selective and miss spend delta; split-conformal set and interval carry none.
ok("family/delta unions only over the modes that spend it",
   abs(fam["family_delta"] - 0.05) < 1e-12, "got %r" % fam["family_delta"])
ok("family/record block carries the union bound",
   applied["gate"]["family_alpha"] == fam["family_alpha"])
ok("family/record alpha differs from per-question alpha",
   applied["gate"]["family_alpha"] > applied["gate"]["alpha"])
single = ConformalGate.calibrate(cal_r, cal_l, alpha=0.05, questions=["intent"])
ok("family/one gated question is just alpha",
   abs(single.family_risk()["family_alpha"] - 0.05) < 1e-12)
many = ConformalGate({str(i): QuestionGate(str(i), "noul", "selective", 0.05, 0.05,
                                           ["false", "true"], {"threshold": 0.9})
                      for i in range(40)}, 0.05, 0.05)
ok("family/saturates rather than exceeding one", many.family_risk()["family_alpha"] == 1.0)
ok("family/says so when the bound goes vacuous",
   "vacuous" in many.family_risk()["family_guarantee"])
ok("family/empty is not a claim",
   "not a claim" in gate.family_risk([])["family_guarantee"])

# ------------------------------------------------------------------ guards on apply
mismatch = {"type": "choice", "choice": "a",
            "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.7}
raises("guard/option drift is refused", lambda: gate["intent"].apply(mismatch),
       "only valid for the question it was calibrated on")
raises("guard/type drift is refused", lambda: gate["intent"].apply(noul_answer(0.8)),
       "fitted on a choice question")
raises("guard/unknown answer type", lambda: gate["intent"].apply({"type": "essay"}),
       "unknown answer type")
raises("guard/empty probabilities",
       lambda: gate["intent"].apply({"type": "choice", "probabilities": {}}),
       "no probabilities")
raises("guard/result without answers", lambda: gate.apply({"nope": 1}), "answers")

# ------------------------------------------------------------------ label spellings
bad_label = [dict(l) for l in cal_l]
bad_label[0]["intent"] = "not-a-criterion"
raises("labels/unknown choice criterion",
       lambda: ConformalGate.calibrate(cal_r, bad_label, questions=["intent"]),
       "not one of the criteria")

bad_score = [dict(l) for l in cal_l]
bad_score[0]["severity"] = 9
raises("labels/score level outside the declared range",
       lambda: ConformalGate.calibrate(cal_r, bad_score, mode={"severity": "interval"},
                                       questions=["severity"]),
       "outside the 5 declared levels")

bad_bool = [dict(l) for l in cal_l]
bad_bool[0]["severity"] = True
raises("labels/bool is not a score level",
       lambda: ConformalGate.calibrate(cal_r, bad_bool, mode={"severity": "interval"},
                                       questions=["severity"]),
       "not a bool")

spelled = [dict(l, unsafe="yes" if l["unsafe"] else "no") for l in cal_l]
spelled_gate = ConformalGate.calibrate(cal_r, spelled, alpha=0.05,
                                       mode={"unsafe": "miss"}, questions=["unsafe"])
plain_gate = ConformalGate.calibrate(cal_r, cal_l, alpha=0.05,
                                     mode={"unsafe": "miss"}, questions=["unsafe"])
check("labels/yes-no spells the same gate as a bool",
      spelled_gate["unsafe"].params["threshold"], plain_gate["unsafe"].params["threshold"])
raises("labels/rejects a nonsense boolean spelling",
       lambda: ConformalGate.calibrate(cal_r, [dict(l, unsafe="perhaps") for l in cal_l],
                                       mode={"unsafe": "miss"}, questions=["unsafe"]),
       "not a boolean spelling")

# Partial labelling: a question labelled on only some rows still fits on those rows.
partial = [dict(l) if i % 3 == 0 else {k: v for k, v in l.items() if k != "intent"}
           for i, l in enumerate(cal_l)]
part_gate = ConformalGate.calibrate(cal_r, partial, alpha=0.05, questions=["intent"])
check("labels/partial labelling fits on the labelled rows",
      part_gate["intent"].diagnostics["n_calibration"], len(range(0, len(cal_l), 3)))

raises("calibrate/rejects mismatched lengths",
       lambda: ConformalGate.calibrate(cal_r, cal_l[:-1]), "same length")
raises("calibrate/rejects an empty split", lambda: ConformalGate.calibrate([], []),
       "at least one")
raises("calibrate/rejects alpha out of range",
       lambda: ConformalGate.calibrate(cal_r, cal_l, alpha=1.5), "alpha must lie")
raises("calibrate/rejects delta out of range",
       lambda: ConformalGate.calibrate(cal_r, cal_l, delta=0.0), "delta must lie")
raises("calibrate/rejects a question nobody labelled",
       lambda: ConformalGate.calibrate(cal_r, [{} for _ in cal_l]),
       "no question id appears in both")
raises("calibrate/rejects an unlabelled named question",
       lambda: ConformalGate.calibrate(cal_r, cal_l, questions=["ghost"]),
       "no labelled rows")

# ------------------------------------------------------------------ serialisation
path = os.path.join(tempfile.mkdtemp(), "gate.json")
gate.save(path)
loaded = ConformalGate.load(path)
check("json/round trip preserves every gate", sorted(loaded.gates), sorted(gate.gates))
check("json/round trip preserves alpha", loaded.alpha, gate.alpha)
for q in gate.gates:
    check("json/%s mode survives" % q, loaded[q].mode, gate[q].mode)
    check("json/%s params survive" % q, loaded[q].params, gate[q].params)
check("json/applies identically after a round trip",
      loaded.apply(test_r[0])["answers"], gate.apply(test_r[0])["answers"])
raw = json.load(open(path, encoding="utf-8"))
ok("json/is plain serialisable data", isinstance(raw, dict) and "gates" in raw)
ok("json/carries no model weights", "state_dict" not in json.dumps(raw))

# Non-finite values have to survive JSON, which has no literal for them.
inf_gate = ConformalGate.calibrate(tiny_dataset := make_dataset(40, 51)[0],
                                   make_dataset(40, 51)[1], alpha=0.05, delta=0.05,
                                   questions=["intent"])
ipath = os.path.join(tempfile.mkdtemp(), "inf.json")
inf_gate.save(ipath)
check("json/an infinite threshold round trips",
      ConformalGate.load(ipath)["intent"].params["threshold"], float("inf"))
ok("json/a NaN diagnostic round trips as null",
   ConformalGate.load(ipath)["intent"].diagnostics["empirical_joint_risk"] is None)

future = gate.to_dict()
future["schema_version"] = 99
raises("json/refuses a newer schema", lambda: ConformalGate.from_dict(future),
       "newer Taut")

# ------------------------------------------------------------------ reporting and dunders
rep = gate.report()
ok("report/names every question", all(q[:22] in rep for q in gate.gates))
ok("report/states the guarantee", "miss <=" in rep and "coverage >=" in rep)
ok("report/flags an uncertifiable gate",
   "NOT CERTIFIABLE" in ConformalGate.calibrate(*make_dataset(40, 61), alpha=0.05,
                                                questions=["intent"]).report())
check("dunder/len", len(gate), 3)
check("dunder/contains", "intent" in gate, True)
check("dunder/contains a stranger", "ghost" in gate, False)
ok("dunder/getitem", isinstance(gate["intent"], QuestionGate))
ok("dunder/repr names the size", "3 questions" in repr(gate))
ok("dunder/question repr names the mode", "mode=set" in repr(gate["intent"]))

# ------------------------------------------------------------------ package surface
import taut  # noqa: E402

for name in ("ConformalGate", "QuestionGate", "selective_threshold", "miss_threshold",
             "min_calibration_size", "binomial_upper_bound"):
    ok("export/%s is in __all__" % name, name in taut.__all__)
    ok("export/%s resolves" % name, getattr(taut, name) is not None)
check("export/ConformalGate is the real class", taut.ConformalGate, ConformalGate)

# The module is pure NumPy: a gate must be usable in a process with no torch at all.
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
import taut
g = taut.ConformalGate.load(%r)
out = g.apply({"answers": {"unsafe": {"type": "noul", "noul": 0.9, "confidence": 0.9}}})
assert "torch" not in sys.modules, "applying a gate pulled in torch"
print("ok", out["answers"]["unsafe"]["gate"]["mode"])
''' % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
check("torch-free/a gate applies with torch blocked", proc.returncode, 0)
ok("torch-free/it really applied the gate", proc.stdout.strip() == "ok miss",
   proc.stdout.strip() + proc.stderr[-400:])

# ------------------------------------------------------------------ report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all conformal risk-control tests passed")
sys.exit(1 if FAIL else 0)
