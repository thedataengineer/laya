"""Does the certified risk bound actually hold? Population-risk validation.

Run: ``python research/conformal/validate_risk.py`` (~3 minutes, NumPy only).

The earlier run scored each trial against a 5000-row sampled test set, so the
"breach" indicator carried the sampling noise of the estimate on top of the
question being asked.  Here the generator is explicit -- correctness is drawn
Bernoulli(c(p)) for a known link c -- so the TRUE risk at any threshold is a
population quantity, evaluated once on a frozen two-million-row score pool with
no label sampling at all.

A breach is r_true(tau_hat) > alpha.  The guarantee says that happens with
probability at most delta, whatever the distribution, so each configuration is
run against a calibrated link and two miscalibrated ones.
"""
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taut.conformal import miss_threshold, selective_threshold  # noqa: E402

POOL = 2_000_000
TRIALS = 1000
DELTA = 0.05

# Correctness links.  "calibrated" is the honest model; the other two are the
# distribution-free claim under stress -- validity must not depend on them.
LINKS = {
    "calibrated":    lambda p: p,
    "overconfident": lambda p: np.clip(p ** 1.6, 0.0, 1.0),
    "underconfident": lambda p: np.clip(p ** 0.6, 0.0, 1.0),
}


def draw_scores(rng, n, k=4):
    """Max-probability of a k-option choice question: floor 1/k, skewed high."""
    return 1.0 / k + (1.0 - 1.0 / k) * rng.beta(5.0, 1.5, size=n)


def selective_trials(link_name, alpha, n_cal, seed=0):
    link = LINKS[link_name]
    rng = np.random.default_rng(seed)
    pool = draw_scores(rng, POOL)
    pool_wrong = 1.0 - link(pool)                     # P(wrong | score) -- no labels drawn

    def r_true(tau):
        if not np.isfinite(tau):
            return 0.0
        return float(np.mean((pool >= tau) * pool_wrong))

    breaches = 0
    abstains = 0
    covs, risks = [], []
    for t in range(TRIALS):
        r = np.random.default_rng(seed * 100_003 + t + 1)
        s = draw_scores(r, n_cal)
        correct = r.random(n_cal) < link(s)
        out = selective_threshold(s, correct, alpha=alpha, delta=DELTA)
        tau = out["threshold"]
        if not np.isfinite(tau):
            abstains += 1
        rt = r_true(tau)
        risks.append(rt)
        covs.append(out["coverage"])
        if rt > alpha:
            breaches += 1
    return dict(breaches=breaches, abstains=abstains,
                mean_cov=float(np.mean(covs)), mean_risk=float(np.mean(risks)))


def draw_miss(rng, n, prevalence=0.2):
    """noul guardrail: label first, then a class-conditional P(unsafe) score."""
    y = rng.random(n) < prevalence
    s = np.where(y, rng.beta(6.0, 2.0, size=n), rng.beta(2.0, 6.0, size=n))
    return s, y


def miss_trials(alpha, n_cal, prevalence=0.2, seed=0):
    rng = np.random.default_rng(seed + 7919)
    ps, py = draw_miss(rng, POOL, prevalence)
    pos = ps[py]                                       # the conditioning event

    def r_true(tau):
        return float(np.mean(pos < tau))               # missed = not blocked

    breaches = 0
    blocks, risks = [], []
    for t in range(TRIALS):
        r = np.random.default_rng(seed * 100_019 + t + 1)
        s, y = draw_miss(r, n_cal, prevalence)
        if y.sum() == 0:
            continue
        out = miss_threshold(s, y, alpha=alpha, delta=DELTA)
        rt = r_true(out["threshold"])
        risks.append(rt)
        blocks.append(out["block_rate"])
        if rt > alpha:
            breaches += 1
    return dict(breaches=breaches, abstains=0,
                mean_cov=float(np.mean(blocks)), mean_risk=float(np.mean(risks)))


def budget(trials, delta):
    """Upper end of a 99% binomial interval on the allowed breach count."""
    mu = trials * delta
    return mu + 2.576 * math.sqrt(trials * delta * (1 - delta))


if __name__ == "__main__":
    cap = budget(TRIALS, DELTA)
    print("trials=%d delta=%.2f  breach budget <= %.0f/%d (99%% binomial)\n"
          % (TRIALS, DELTA, cap, TRIALS))

    print("=== selective: joint accepted-and-wrong rate ===")
    print("%-15s %6s %7s %9s %9s %9s %8s" %
          ("link", "alpha", "n_cal", "breaches", "mean_cov", "mean_risk", "verdict"))
    fails = []
    for link in LINKS:
        for alpha, n_cal in [(0.10, 500), (0.05, 500), (0.05, 3000), (0.02, 3000), (0.01, 5000)]:
            res = selective_trials(link, alpha, n_cal, seed=1)
            ok = res["breaches"] <= cap
            fails.append(ok)
            print("%-15s %6.2f %7d %9d %9.3f %9.4f %8s" %
                  (link, alpha, n_cal, res["breaches"], res["mean_cov"],
                   res["mean_risk"], "OK" if ok else "BREACH"))

    print("\n=== miss: false-negative rate among positives ===")
    print("%-15s %6s %7s %9s %9s %9s %8s" %
          ("prevalence", "alpha", "n_cal", "breaches", "mean_blk", "mean_risk", "verdict"))
    for prev in (0.2, 0.05):
        for alpha, n_cal in [(0.10, 2000), (0.05, 2000), (0.01, 5000)]:
            res = miss_trials(alpha, n_cal, prev, seed=2)
            ok = res["breaches"] <= cap
            fails.append(ok)
            print("%-15.2f %6.2f %7d %9d %9.3f %9.4f %8s" %
                  (prev, alpha, n_cal, res["breaches"], res["mean_cov"],
                   res["mean_risk"], "OK" if ok else "BREACH"))

    print("\nVERDICT: %s" % ("all configurations within budget" if all(fails)
                             else "%d of %d configurations breached" % (fails.count(False), len(fails))))
