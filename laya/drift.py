"""Does the gate still hold? Label-free drift monitoring for a fitted ConformalGate.

A conformal gate's guarantee is distribution-free but not distribution-*proof*. It needs
exchangeability: calibration traffic and live traffic drawn from the same distribution. A
gate fitted on last quarter's tickets carries no guarantee on this quarter's if the mix
moved, and nothing in :mod:`laya.conformal` notices. That is the gap this module closes --
a gate with no expiry date is a guarantee someone will keep quoting long after it stopped
being true.

The hard constraint is that production has no labels. You cannot measure risk live; if you
could, you would not need a gate. What you *can* measure, with no labels at all, is whether
the distribution the gate was fitted against still describes what is arriving:

``acceptance``
    The gate was fitted to accept a known share of traffic. If it was calibrated to keep
    51% and is now keeping 78%, the input distribution moved -- whatever the labels would
    have said. Tested with an **exact two-sided binomial** test against the calibrated
    coverage. No approximation, no sketch, no tuning constant.

``scores``
    A two-sample Kolmogorov-Smirnov test of the live score distribution against the
    calibration sketch stored in the gate. Catches shifts that leave the acceptance rate
    unchanged -- mass moving *within* the accepted region, which the binomial test cannot
    see.

    The reference is a 101-quantile sketch, so the reference CDF is resolved to about
    0.01. The statistic is floored accordingly rather than reported as though the sketch
    were the calibration set.

**What this cannot tell you, and the distinction matters.** Both tests watch the *scores*.
A shift in the score-to-correctness link -- same confidences, worse answers -- moves real
risk while leaving both tests quiet. Stable scores are necessary for the guarantee to
survive, not sufficient for it. A clean report means "no evidence the gate has expired",
never "the gate still holds". The only thing that re-establishes the guarantee is refitting
on freshly labelled data, and this module's job is to tell you *when* that is overdue, not
to excuse you from it.

Pure NumPy, like the gate itself.

    from laya import ConformalGate, GateMonitor

    gate = ConformalGate.load("triage_gate.json")
    monitor = GateMonitor(gate)

    for ticket in stream:
        result = gate.apply(agent.predict(ticket, questions))
        monitor.observe(result)

    status = monitor.check()
    if status["status"] == "expired":
        page_whoever_owns_the_labels(monitor.report())
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .conformal import ConformalGate, QuestionGate, gate_score

__all__ = ["GateMonitor", "binomial_two_sided_p", "ks_two_sample", "ks_p_value"]

# Below this the tests have no power worth reporting, so the monitor says "watching"
# rather than producing a verdict from a handful of requests.
_MIN_OBSERVATIONS = 100

# The sketch stores quantiles at 1% spacing, which is the floor on how finely the
# reference CDF can be resolved. A KS statistic under this is sketch granularity, not drift.
_SKETCH_RESOLUTION = 0.01


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def binomial_two_sided_p(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial p-value for ``k`` successes in ``n`` at rate ``p``.

    The method of small p-values: sum the probability of every outcome no more likely
    than the one observed. Exact rather than normal-approximate because an acceptance
    rate near 0 or 1 -- exactly where a strict gate operates -- is where the normal
    approximation is worst.
    """
    if n <= 0:
        return 1.0
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0

    i = np.arange(n + 1)
    log_coeff = np.concatenate(([0.0], np.cumsum(np.log(np.arange(n, 0, -1))
                                                 - np.log(np.arange(1, n + 1)))))
    log_pmf = log_coeff + i * math.log(p) + (n - i) * math.log1p(-p)
    pmf = np.exp(log_pmf - log_pmf.max())
    pmf /= pmf.sum()
    # A relative tolerance keeps ties on the far side of the mode from being dropped by
    # floating-point noise, which would understate the p-value.
    return float(pmf[pmf <= pmf[k] * (1.0 + 1e-9)].sum())


def _sketch_cdf(sketch: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    """Reference CDF at ``x``, interpolated from a stored quantile sketch."""
    qs = np.asarray(sketch["quantiles"], dtype=np.float64)
    levels = np.asarray(sketch["levels"], dtype=np.float64)
    # np.interp needs an increasing x-table; quantiles are non-decreasing, and ties (a
    # spike in the score distribution) must map to the *highest* level at that value so
    # the CDF stays right-continuous.
    return np.interp(x, qs, levels, left=0.0, right=1.0)


def ks_two_sample(scores: Sequence[float], sketch: Mapping[str, Any]) -> Dict[str, Any]:
    """Two-sample KS statistic of live ``scores`` against a calibration ``sketch``."""
    x = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = x.size
    n_ref = int(sketch.get("n", 0))
    if n == 0 or n_ref == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": n, "n_reference": n_ref,
                "effective_n": 0.0}

    ref = _sketch_cdf(sketch, x)
    # Compare against both edges of each empirical step, or the statistic is understated
    # by up to 1/n.
    upper = np.arange(1, n + 1, dtype=np.float64) / n
    lower = np.arange(0, n, dtype=np.float64) / n
    d = float(max(np.max(np.abs(upper - ref)), np.max(np.abs(lower - ref))))

    n_eff = n * n_ref / float(n + n_ref)
    return {"statistic": d, "p_value": ks_p_value(d, n_eff), "n": n,
            "n_reference": n_ref, "effective_n": n_eff}


def ks_p_value(d: float, n_eff: float) -> float:
    """P(D > d) under the null, from the asymptotic Kolmogorov distribution.

    Asymptotic, not exact: valid once ``n_eff`` is in the high tens, which
    ``_MIN_OBSERVATIONS`` enforces before any verdict is issued.
    """
    if n_eff <= 0 or d <= 0:
        return 1.0
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    total = 0.0
    for k in range(1, 101):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return float(min(1.0, max(0.0, total)))


# --------------------------------------------------------------------------------------
# The monitor
# --------------------------------------------------------------------------------------

def _expected_rate(gate: QuestionGate) -> Optional[float]:
    """The share of traffic this gate was calibrated to accept, if that is defined."""
    d = gate.diagnostics
    if gate.mode == "selective":
        return float(d["coverage"]) if "coverage" in d else None
    if gate.mode == "miss":
        # `accepted` on a miss gate means "not blocked".
        return 1.0 - float(d["block_rate"]) if "block_rate" in d else None
    if gate.mode == "set":
        return float(d["singleton_rate"]) if "singleton_rate" in d else None
    return None  # an interval gate accepts on its width, which does not vary per request


class GateMonitor:
    """Watches live traffic for evidence that a fitted gate has stopped applying.

    Holds a bounded window of recent scores per question -- ``window`` requests, oldest
    dropped -- so memory is fixed whatever the traffic, and the verdict reflects recent
    traffic rather than everything since the process started.
    """

    def __init__(self, gate: ConformalGate, window: int = 2000, alpha_test: float = 0.01):
        """
        Args:
            gate: The fitted gate to watch.
            window: How many recent requests to keep per question.
            alpha_test: Significance for the drift tests, Bonferroni-corrected across
                questions and across the two tests. Deliberately stricter than a
                conventional 0.05: this fires an operational alarm, and a monitor that
                cries wolf gets muted, which is worse than not having one.
        """
        if window < 1:
            raise ValueError("window must be at least 1; got %r" % (window,))
        if not (0.0 < alpha_test < 1.0):
            raise ValueError("alpha_test must lie in (0, 1); got %r" % (alpha_test,))
        self.gate = gate
        self.window = int(window)
        self.alpha_test = float(alpha_test)
        self._scores: Dict[str, List[float]] = {q: [] for q in gate.gates}
        self._accepted: Dict[str, List[bool]] = {q: [] for q in gate.gates}
        self.n_seen = 0

    # -- ingestion ----------------------------------------------------------------

    def observe(self, result: Mapping[str, Any]) -> None:
        """Record one Laya result. Gated or ungated -- the gate is applied if needed."""
        answers = result.get("answers", result)
        if not isinstance(answers, Mapping):
            raise ValueError("expected a Laya result with an 'answers' key")
        self.n_seen += 1
        for qid, qgate in self.gate.gates.items():
            answer = answers.get(qid)
            if answer is None:
                continue
            block = answer.get("gate") if isinstance(answer, Mapping) else None
            if block is None:
                block = qgate.apply(answer)
            self._push(self._scores[qid], gate_score(qgate, answer))
            self._push(self._accepted[qid], bool(block.get("accepted", True)))

    def observe_batch(self, results: Sequence[Mapping[str, Any]]) -> None:
        for r in results:
            self.observe(r)

    def _push(self, buf: List[Any], value: Any) -> None:
        buf.append(value)
        if len(buf) > self.window:
            del buf[:len(buf) - self.window]

    # -- verdict ------------------------------------------------------------------

    def check(self) -> Dict[str, Any]:
        """Per-question drift evidence and one overall status.

        ``ok`` -- no evidence the gate has expired. ``watching`` -- too little traffic to
        say. ``expired`` -- at least one test rejects, and the gate should be refit on
        freshly labelled data before its guarantee is quoted again.
        """
        questions: Dict[str, Any] = {}
        # Two tests per question, so the per-test level is split accordingly.
        n_tests = max(1, 2 * len(self.gate.gates))
        level = self.alpha_test / n_tests

        for qid, qgate in self.gate.gates.items():
            scores = self._scores[qid]
            accepted = self._accepted[qid]
            n = len(scores)
            entry: Dict[str, Any] = {"mode": qgate.mode, "n_observed": n}

            if n < _MIN_OBSERVATIONS:
                entry["status"] = "watching"
                entry["reason"] = ("%d of %d requests needed before a verdict carries any "
                                   "power" % (n, _MIN_OBSERVATIONS))
                questions[qid] = entry
                continue

            drifted = []

            expected = _expected_rate(qgate)
            if expected is not None:
                k = int(sum(accepted))
                p = binomial_two_sided_p(k, n, expected)
                entry["acceptance"] = {
                    "calibrated_rate": expected,
                    "observed_rate": k / n,
                    "p_value": p,
                    "drifted": bool(p < level),
                }
                if p < level:
                    drifted.append("acceptance rate moved from %.1f%% to %.1f%% (p=%.2g)"
                                   % (100 * expected, 100 * k / n, p))

            sketch = qgate.diagnostics.get("score_sketch")
            if sketch:
                ks = ks_two_sample(scores, sketch)
                # Anything under the sketch's own resolution is granularity, not drift.
                ks_drifted = bool(ks["p_value"] < level
                                  and ks["statistic"] > _SKETCH_RESOLUTION)
                entry["scores"] = dict(ks, drifted=ks_drifted,
                                       resolution_floor=_SKETCH_RESOLUTION)
                if ks_drifted:
                    drifted.append("score distribution shifted (KS=%.3f, p=%.2g)"
                                   % (ks["statistic"], ks["p_value"]))
            else:
                # Gates written before the sketch existed still monitor on acceptance.
                entry["scores"] = {"unavailable": "gate carries no calibration sketch; "
                                                  "refit to enable the score test"}

            entry["status"] = "expired" if drifted else "ok"
            if drifted:
                entry["reason"] = "; ".join(drifted)
            questions[qid] = entry

        statuses = {e["status"] for e in questions.values()}
        if "expired" in statuses:
            overall = "expired"
        elif statuses == {"watching"} or not statuses:
            overall = "watching"
        else:
            overall = "ok"

        return {
            "status": overall,
            "n_seen": self.n_seen,
            "window": self.window,
            "alpha_test": self.alpha_test,
            "per_test_level": level,
            "questions": questions,
            "caveat": ("these tests watch scores, not correctness; a clean result is "
                       "no evidence of expiry, not proof the guarantee still holds"),
        }

    def report(self) -> str:
        """The check as a table, for a log line or an alert body."""
        res = self.check()
        lines = ["Laya gate drift  status=%s  seen=%d  window=%d  level=%.2g per test"
                 % (res["status"], res["n_seen"], res["window"], res["per_test_level"])]
        lines.append("")
        header = "%-22s %-10s %-9s %-16s %s" % ("question", "mode", "status", "acceptance", "scores")
        lines.append(header)
        lines.append("-" * len(header))
        for qid, e in res["questions"].items():
            acc = e.get("acceptance")
            acc_s = ("%.1f%% vs %.1f%%" % (100 * acc["observed_rate"],
                                           100 * acc["calibrated_rate"])) if acc else "n/a"
            sc = e.get("scores") or {}
            sc_s = ("KS=%.3f p=%.2g" % (sc["statistic"], sc["p_value"])) if "statistic" in sc else "n/a"
            lines.append("%-22s %-10s %-9s %-16s %s" % (qid[:22], e["mode"], e["status"], acc_s, sc_s))
        for qid, e in res["questions"].items():
            if e.get("reason") and e["status"] == "expired":
                lines.append("")
                lines.append("  %s: %s" % (qid, e["reason"]))
        if res["status"] == "expired":
            lines.append("")
            lines.append("  Refit on freshly labelled traffic before quoting the guarantee again.")
        lines.append("")
        lines.append("  %s" % res["caveat"])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "GateMonitor(%d questions, seen=%d, window=%d)" % (
            len(self.gate.gates), self.n_seen, self.window)
