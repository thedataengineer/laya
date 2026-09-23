"""Does the gate still hold? Label-free drift monitoring for a fitted ConformalGate.

A conformal gate's guarantee is distribution-free but not distribution-*proof*. It needs
exchangeability: calibration traffic and live traffic drawn from the same distribution. A
gate fitted on last quarter's tickets carries no guarantee on this quarter's if the mix
moved, and nothing in :mod:`taut.conformal` notices. That is the gap this module closes --
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

Both of those watch the *scores*, and a shift in the score-to-correctness link -- same
confidences, worse answers -- moves real risk while leaving them quiet. Stable scores are
necessary for the guarantee to survive, not sufficient for it. Closing that half needs
labels, so the third test asks for a few:

``risk``
    Feed ``audit()`` a small, **randomly sampled** slice of live traffic with gold labels
    and it tests the realised loss directly against ``alpha`` -- an exact one-sided
    binomial test, in the same terms the gate certified. This is the only test here that
    can say the guarantee itself has broken rather than that its inputs moved.

    :func:`audit_size` says how many labels that takes. Detecting a doubling of a 2%
    budget at 80% power needs 424 audited rows, and a rise from 2% to 10% needs 54 --
    because the question is coarse: has risk left the budget, not what exactly is it now.

Without an audit a clean report means "no evidence the gate has expired", never "the gate
still holds". With one it means rather more. Either way the only thing that re-establishes
the guarantee is refitting on freshly labelled data; this module tells you *when* that is
overdue, not how to avoid it.

Pure NumPy, like the gate itself.

    from taut import ConformalGate, GateMonitor

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

from .conformal import ConformalGate, QuestionGate, gate_loss, gate_score

__all__ = ["GateMonitor", "audit_size", "binomial_two_sided_p",
           "binomial_upper_tail_p", "ks_two_sample", "ks_p_value"]

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


def binomial_upper_tail_p(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Bin(n, p)``: evidence that the true rate exceeds ``p``.

    One-sided on purpose. An audit asks a single question -- has realised loss left the
    budget -- and risk *below* ``alpha`` is the gate working, not a finding.
    """
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    if p >= 1.0:
        return 1.0
    i = np.arange(n + 1)
    log_coeff = np.concatenate(([0.0], np.cumsum(np.log(np.arange(n, 0, -1))
                                                 - np.log(np.arange(1, n + 1)))))
    log_pmf = log_coeff + i * math.log(p) + (n - i) * math.log1p(-p)
    pmf = np.exp(log_pmf - log_pmf.max())
    pmf /= pmf.sum()
    return float(pmf[k:].sum())


def audit_size(alpha: float, detect: float, power: float = 0.8,
               level: float = 0.05) -> int:
    """Audited rows needed to notice that risk has risen from ``alpha`` to ``detect``.

    Exact, by searching upward for the smallest ``n`` whose rejection region -- the
    smallest ``k`` with ``P(X >= k | alpha) <= level`` -- is reached with probability at
    least ``power`` when the true rate is ``detect``.

    The number is usually smaller than people expect, and that is the argument for
    auditing at all: a doubling of a 2% budget takes 424 labels at 80% power, and a jump
    from 2% to 10% takes 54 -- against the thousands that refitting a gate would need.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1); got %r" % (alpha,))
    if not (0.0 < power < 1.0) or not (0.0 < level < 1.0):
        raise ValueError("power and level must both lie in (0, 1)")
    if detect <= alpha:
        raise ValueError("detect must exceed alpha; there is nothing to notice at %r"
                         % (detect,))
    for n in range(1, 100_001):
        k = next((j for j in range(n + 1) if binomial_upper_tail_p(j, n, alpha) <= level),
                 None)
        if k is None:
            continue
        if 1.0 - binomial_upper_tail_p(k, n, detect) <= 1.0 - power:
            return n
    raise ValueError("no practical sample size detects %r against %r at power %r"
                     % (detect, alpha, power))


def ks_two_sample(scores: Sequence[float], sketch: Mapping[str, Any]) -> Dict[str, Any]:
    """Two-sample KS statistic of live ``scores`` against a calibration ``sketch``.

    The sketch's 101 quantiles are treated as what they are -- a 101-point sample from the
    calibration distribution -- and compared with a textbook two-sample ECDF statistic.

    That framing is not cosmetic. Reconstructing a *continuous* reference CDF and checking
    both edges of each empirical step is the usual recipe, and it is wrong here: a
    confident decision model piles mass on a handful of scores (p=1.000 over and over),
    and against a tied reference the lower step edge compares quantities that are not the
    same thing. It reported a statistic of 0.40 on a sample against its own sketch. The
    pooled-ECDF form handles ties exactly and gives 0.006 on that case.

    The p-value uses the true calibration size, not 101: for any realistic ``n_ref`` the
    sketch's 1% discretisation is smaller than the sampling noise it is summarising, and
    the caller floors the statistic at that resolution anyway.
    """
    x = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = x.size
    n_ref = int(sketch.get("n", 0))
    y = np.sort(np.asarray(sketch.get("quantiles", []), dtype=np.float64).ravel())
    if n == 0 or n_ref == 0 or y.size == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": n, "n_reference": n_ref,
                "effective_n": 0.0}

    pooled = np.union1d(x, y)
    fx = np.searchsorted(x, pooled, side="right") / float(n)
    fy = np.searchsorted(y, pooled, side="right") / float(y.size)
    d = float(np.max(np.abs(fx - fy)))

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
        # Audited rows are kept separately and are NOT windowed: labels are scarce, and
        # silently ageing them out would shrink the one test that can see real risk.
        self._audit: Dict[str, List[bool]] = {q: [] for q in gate.gates}
        self.n_seen = 0
        self.n_audited = 0

    # -- ingestion ----------------------------------------------------------------

    def observe(self, result: Mapping[str, Any]) -> None:
        """Record one Taut result. Gated or ungated -- the gate is applied if needed."""
        answers = result.get("answers", result)
        if not isinstance(answers, Mapping):
            raise ValueError("expected a Taut result with an 'answers' key")
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

    def audit(self, result: Mapping[str, Any], labels: Mapping[str, Any]) -> None:
        """Record one labelled row, so realised risk can be tested against ``alpha``.

        The row must come from a **random sample of live traffic**. Labelling the cases
        that looked wrong, or only the ones the gate accepted, biases the estimate in the
        direction that makes the guarantee look worst or best respectively, and the test
        below has no way to detect that it happened. Sample first, label second.

        Audited rows also count as observed, so a caller need not feed the same result to
        both methods.
        """
        if not isinstance(labels, Mapping):
            raise ValueError("labels must be a mapping of question id to gold answer")
        self.observe(result)
        answers = result.get("answers", result)
        audited = False
        for qid, qgate in self.gate.gates.items():
            answer = answers.get(qid)
            if answer is None or qid not in labels or labels[qid] is None:
                continue
            loss, counted = gate_loss(qgate, answer, labels[qid])
            if counted:
                self._audit[qid].append(bool(loss))
            audited = True
        if audited:
            self.n_audited += 1

    def audit_batch(self, results: Sequence[Mapping[str, Any]],
                    labels: Sequence[Mapping[str, Any]]) -> None:
        if len(results) != len(labels):
            raise ValueError("results and labels must be the same length; got %d and %d"
                             % (len(results), len(labels)))
        for r, l in zip(results, labels):
            self.audit(r, l)

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
        # Up to three tests per question, so the per-test level is split accordingly.
        # Questions with no audited rows still count: the correction must not depend on
        # how the data happened to arrive, or it would drift with the traffic.
        n_tests = max(1, 3 * len(self.gate.gates))
        level = self.alpha_test / n_tests

        for qid, qgate in self.gate.gates.items():
            scores = self._scores[qid]
            accepted = self._accepted[qid]
            n = len(scores)
            entry: Dict[str, Any] = {"mode": qgate.mode, "n_observed": n}

            audited = self._audit[qid]
            risk_block = self._risk_test(qgate, audited, level)
            if risk_block is not None:
                entry["risk"] = risk_block

            if n < _MIN_OBSERVATIONS:
                if risk_block is not None and risk_block.get("drifted"):
                    # Realised risk outranks "not enough traffic to say": a breach the
                    # labels already prove does not become tentative for want of volume.
                    entry["status"] = "expired"
                    entry["reason"] = risk_block["reason"]
                    questions[qid] = entry
                    continue
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

            if risk_block is not None and risk_block.get("drifted"):
                drifted.insert(0, risk_block["reason"])

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
            "n_audited": self.n_audited,
            "caveat": ("the acceptance and score tests watch scores, not correctness; "
                       "without audited rows a clean result is no evidence of expiry, "
                       "not proof the guarantee still holds"),
        }

    def _risk_test(self, qgate: QuestionGate, audited: Sequence[bool],
                   level: float) -> Optional[Dict[str, Any]]:
        """Exact one-sided binomial test of realised loss against the gate's own alpha."""
        n = len(audited)
        if n == 0:
            return None
        k = int(sum(audited))
        alpha = float(qgate.alpha)
        p = binomial_upper_tail_p(k, n, alpha)
        drifted = bool(p < level)
        block = {
            "n_audited": n,
            "observed_loss": k,
            "observed_risk": k / n,
            "alpha": alpha,
            "p_value": p,
            "drifted": drifted,
            "denominator": ("positives" if qgate.mode == "miss" else "audited rows"),
        }
        if drifted:
            block["reason"] = ("realised risk %.3g exceeds the %.3g budget on %d audited "
                               "rows (p=%.2g)" % (k / n, alpha, n, p))
        else:
            # Absence of evidence is not evidence of absence, and an auditor deserves the
            # difference spelled out rather than implied by a green status.
            detect = min(0.999, max(alpha * 2.0, alpha + 0.01))
            try:
                need = audit_size(alpha, detect, power=0.8, level=level)
            except ValueError:
                need = None
            block["note"] = ("no evidence risk has left the budget; %s"
                             % ("%d audited rows would give 80%% power to notice it "
                                "doubling, and there are %d" % (need, n) if need
                                else "power at this budget is limited by sample size"))
        return block

    def report(self) -> str:
        """The check as a table, for a log line or an alert body."""
        res = self.check()
        lines = ["Taut gate drift  status=%s  seen=%d  window=%d  level=%.2g per test"
                 % (res["status"], res["n_seen"], res["window"], res["per_test_level"])]
        lines.append("")
        header = "%-22s %-10s %-9s %-16s %-18s %s" % ("question", "mode", "status", "acceptance", "scores", "audited risk")
        lines.append(header)
        lines.append("-" * len(header))
        for qid, e in res["questions"].items():
            acc = e.get("acceptance")
            acc_s = ("%.1f%% vs %.1f%%" % (100 * acc["observed_rate"],
                                           100 * acc["calibrated_rate"])) if acc else "n/a"
            sc = e.get("scores") or {}
            sc_s = ("KS=%.3f p=%.2g" % (sc["statistic"], sc["p_value"])) if "statistic" in sc else "n/a"
            rk = e.get("risk")
            rk_s = ("%.2f%% of %d (a=%.3g)" % (100 * rk["observed_risk"], rk["n_audited"],
                                               rk["alpha"])) if rk else "not audited"
            lines.append("%-22s %-10s %-9s %-16s %-18s %s"
                         % (qid[:22], e["mode"], e["status"], acc_s, sc_s, rk_s))
        for qid, e in res["questions"].items():
            if e.get("reason") and e["status"] == "expired":
                lines.append("")
                lines.append("  %s: %s" % (qid, e["reason"]))
        if res["status"] == "expired":
            lines.append("")
            lines.append("  Refit on freshly labelled traffic before quoting the guarantee again.")
        if not any(e.get("risk") for e in res["questions"].values()):
            lines.append("")
            lines.append("  No rows audited. The tests above watch scores, so they cannot "
                         "see the model")
            lines.append("  getting worse at unchanged confidence. audit() closes that; "
                         "audit_size() prices it.")
        lines.append("")
        lines.append("  %s" % res["caveat"])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "GateMonitor(%d questions, seen=%d, window=%d)" % (
            len(self.gate.gates), self.n_seen, self.window)
