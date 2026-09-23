"""Distribution-free risk control for Laya decisions.

Laya reports a calibrated probability. This module turns that probability into an
*operational contract*: a threshold, a prediction set, or an interval carrying a
finite-sample, distribution-free guarantee that holds for any underlying data
distribution, for any model, without asymptotics.

Three guarantees are on offer, each fitted from a held-out labelled sample:

``selective``
    Abstain below a threshold so that the share of *all* incoming decisions that
    are both auto-handled and wrong is at most ``alpha``, with confidence
    ``1 - delta``. Answers "how much of the queue can I auto-route while keeping
    mistakes under 2% of everything that arrives?" Fitted by fixed-sequence
    testing over a pre-specified threshold grid with exact Clopper-Pearson upper
    bounds, so the bound is valid at finite n and needs no multiplicity
    correction.

    Note the denominator: the certified quantity is joint, not the error rate
    *among accepted* decisions. That conditional rate is reported as
    ``empirical_selective_risk`` but deliberately left uncertified -- see
    :func:`selective_threshold` for why bounding it is not sound here. Since
    accepted-and-wrong is a subset of accepted, the joint bound also caps the
    conditional rate at ``alpha / coverage``.

``set``
    Emit a *set* of options guaranteed to contain the true option with marginal
    probability at least ``1 - alpha``. Split conformal prediction: LAC (smallest
    average set) or APS (better conditional coverage). Applies to ``choice``;
    ``noul`` is handled as the two-option case.

``interval``
    For ``score`` questions, a symmetric interval around the expected score that
    covers the true level with probability at least ``1 - alpha``.

Plus one guardrail-specific control, ``miss``: pick the *most permissive* block
threshold on a ``noul`` question whose false-negative rate -- unsafe content that
slips through -- is bounded by ``alpha`` at confidence ``1 - delta``.

The module is pure NumPy. It imports neither torch nor transformers, so a gate
can be fitted, serialised and applied in a lightweight process, and a gate JSON
can be shipped to an edge client that never loads a model.

    from laya import Agent, ConformalGate, triage_questions

    agent = Agent()
    questions = triage_questions()
    results = agent.predict_batch(dev_states, questions)

    gate = ConformalGate.calibrate(results, dev_labels, alpha=0.02, delta=0.05)
    gate.save("triage_gate.json")

    decision = gate.apply(agent.predict(ticket, questions))
    if decision["answers"]["intent"]["gate"]["accepted"]:
        auto_route(decision["answers"]["intent"]["choice"])
    else:
        escalate(ticket)

``gate.report()`` prints what was bought: coverage, the risk bound, and the
empirical risk on the calibration split.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "ConformalGate",
    "QuestionGate",
    "binomial_upper_bound",
    "selective_threshold",
    "miss_threshold",
    "min_calibration_size",
    "conformal_quantile",
    "SUPPORTED_MODES",
]

# Modes a question can be gated with. `auto` picks by question type.
SUPPORTED_MODES = ("selective", "set", "interval", "miss")

# `miss` control tests this fixed grid, chosen before any data is seen, which is what
# keeps fixed-sequence testing honest: a grid read off the observed scores would be
# data-dependent and the family-wise error bound would not hold. Its denominator is
# the full positive class at every threshold, so unlike selective risk it has no
# shrinking-sample problem and a value grid is the natural sequence.
_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 5)

_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------------------
# Exact binomial bound
# --------------------------------------------------------------------------------------

def _log_binom_coeffs(n: int, k: int) -> np.ndarray:
    """log C(n, i) for i = 0..k, built by cumulative products so no lgamma is needed."""
    if k <= 0:
        return np.zeros(1, dtype=np.float64)
    i = np.arange(1, k + 1, dtype=np.float64)
    steps = np.log(n - i + 1.0) - np.log(i)
    return np.concatenate(([0.0], np.cumsum(steps)))


def _binom_cdf(k: int, n: int, p: float, log_coeffs: np.ndarray) -> float:
    """P(Bin(n, p) <= k), summed in log space so small p does not underflow."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    i = np.arange(0, k + 1, dtype=np.float64)
    logs = log_coeffs + i * math.log(p) + (n - i) * math.log1p(-p)
    return float(min(1.0, np.exp(logs).sum()))


def binomial_upper_bound(k: int, n: int, delta: float = 0.05) -> float:
    """Clopper-Pearson upper confidence bound on a rate: k failures out of n trials.

    Returns the smallest ``p`` for which observing at most ``k`` failures has
    probability at most ``delta``. The true rate exceeds it with probability at
    most ``delta``. Exact at every n, with no normal approximation, so it stays
    honest on the small calibration sets people actually have.
    """
    if n <= 0:
        return 1.0
    k = int(max(0, min(k, n)))
    if k >= n:
        return 1.0
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1); got %r" % (delta,))

    log_coeffs = _log_binom_coeffs(n, k)
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k, n, mid, log_coeffs) > delta:
            lo = mid
        else:
            hi = mid
    return float(hi)


# --------------------------------------------------------------------------------------
# Threshold fitting
# --------------------------------------------------------------------------------------

def min_calibration_size(alpha: float, delta: float) -> int:
    """Fewest calibration points that could ever certify risk ``alpha`` at confidence ``delta``.

    Even with a flawless record -- zero losses -- an exact binomial bound on ``n`` points
    cannot fall below ``alpha`` until ``n >= log(delta) / log(1 - alpha)``. Certifying 5%
    at 95% confidence therefore needs 59 points, 2% needs 149, and 1% needs 299. A gate
    that reports "not enough data" instead of quietly returning a threshold it cannot
    stand behind is the difference between a guarantee and a number.
    """
    if not (0.0 < alpha < 1.0) or not (0.0 < delta < 1.0):
        raise ValueError("alpha and delta must both lie in (0, 1)")
    return int(math.ceil(math.log(delta) / math.log(1.0 - alpha)))


def selective_threshold(scores: np.ndarray, correct: np.ndarray, alpha: float,
                        delta: float = 0.05) -> Dict[str, Any]:
    """Lowest threshold whose accepted-and-wrong rate is provably at most ``alpha``.

    The certified quantity is the *joint* loss -- a decision that is both accepted and
    wrong -- measured over every calibration point: "at most alpha of all incoming
    decisions are auto-handled incorrectly". That is the number an operations owner
    budgets against, and it is the one that can be certified honestly.

    The conditional rate, error *among* accepted decisions, is reported alongside as
    ``empirical_selective_risk`` but deliberately not certified. Its denominator shrinks
    as the threshold rises, so bounding it needs a data-dependent threshold whose
    accepted set is chosen by the same order statistics being tested; an exact binomial
    bound is not valid there, and measurement confirms it -- fitting on that quantity
    breached a 5% budget on 4 of 14 certified trials. The joint loss keeps the
    denominator at the full calibration size, where the binomial model holds exactly.

    Fitting walks a threshold grid fixed before any data is read, from strictest to most
    permissive, testing each with an exact Clopper-Pearson upper bound and stopping at
    the first failure. The loss is non-increasing in the threshold, so nothing past that
    point can recover, and stopping at the first failure over a pre-specified sequence
    holds the family-wise error rate at ``delta`` with no multiplicity correction.

    Args:
        scores: Confidence per calibration decision, higher meaning more confident.
        correct: Whether each decision was right.
        alpha: The share of all decisions allowed to be accepted and wrong.
        delta: Confidence on the bound. 0.05 means it holds 95% of the time.

    Returns:
        The threshold and what it bought: coverage, the certified bound, the observed
        joint and conditional rates, and ``certifiable``. When no threshold on the grid
        passes -- which at ``alpha`` and ``delta`` needs at least
        ``min_calibration_size(alpha, delta)`` points -- the threshold is ``inf``,
        the gate abstains on everything, and ``shortfall`` says what would change that.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=bool).ravel()
    if scores.shape != correct.shape:
        raise ValueError("scores and correct must be the same length; got %d and %d"
                         % (scores.size, correct.size))
    n = scores.size
    if n == 0:
        raise ValueError("selective calibration needs at least one scored decision")

    wrong = ~correct
    best: Optional[Dict[str, Any]] = None
    tested = 0

    # Descending grid: the strictest threshold accepts least and so loses least. Testing
    # runs from there towards permissiveness and stops at the first threshold that fails.
    for tau in _GRID[::-1]:
        accepted = scores >= tau
        n_bad = int((accepted & wrong).sum())
        bound = binomial_upper_bound(n_bad, n, delta)
        tested += 1
        if bound <= alpha:
            n_acc = int(accepted.sum())
            best = {
                "threshold": float(tau),
                "coverage": n_acc / n,
                "empirical_joint_risk": n_bad / n,
                "empirical_selective_risk": (n_bad / n_acc) if n_acc else float("nan"),
                "risk_bound": bound,
                "n_accepted": n_acc,
                "n_errors": n_bad,
            }
        else:
            break

    if best is None:
        floor = min_calibration_size(alpha, delta)
        return {
            "threshold": float("inf"),
            "coverage": 0.0,
            "empirical_joint_risk": float("nan"),
            "empirical_selective_risk": float("nan"),
            "risk_bound": 0.0,
            "n_accepted": 0,
            "n_errors": 0,
            "certifiable": False,
            "candidates_tested": tested,
            "shortfall": {
                "n_calibration": n,
                "min_calibration": floor,
                "reason": ("%d calibration decisions cannot certify alpha=%g at delta=%g; "
                           "an exact bound needs at least %d even with a flawless record"
                           % (n, alpha, delta, floor)) if n < floor else
                          ("no threshold on the grid holds the accepted-and-wrong rate "
                           "under alpha=%g at delta=%g on %d points" % (alpha, delta, n)),
            },
        }
    best["certifiable"] = True
    best["candidates_tested"] = tested
    return best


def miss_threshold(scores: np.ndarray, positive: np.ndarray, alpha: float,
                   delta: float = 0.05) -> Dict[str, Any]:
    """Most permissive block threshold whose false-negative rate is at most ``alpha``.

    ``scores`` is P(true) from a ``noul`` question, ``positive`` the gold label.
    An item is blocked when ``score >= threshold``, so raising the threshold blocks
    less and misses more. The walk therefore runs from threshold 0 (block
    everything, miss nothing) upward and stops at the first threshold whose
    missed-positive bound exceeds ``alpha``.

    This is the number a guardrail owner actually needs: not "how accurate is the
    classifier" but "what fraction of unsafe content reaches the model, at worst".

    The bound is label-conditional: the calibration sample is the positive examples, and
    its size is fixed at every threshold, so the binomial model holds exactly and the
    guarantee reads "of future unsafe items, at most ``alpha`` pass". It says nothing
    about how much benign traffic is blocked -- read ``block_rate`` for that cost, and
    raise ``alpha`` if it is too high to run.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(positive, dtype=bool).ravel()
    if scores.shape != positive.shape:
        raise ValueError("scores and positive must be the same length; got %d and %d"
                         % (scores.size, positive.size))
    n_pos = int(positive.sum())
    n = scores.size
    if n_pos == 0:
        raise ValueError("miss control needs at least one positive example in the "
                         "calibration set; none of the %d labels were true" % n)

    best: Optional[Dict[str, Any]] = None
    for tau in _GRID:
        blocked = scores >= tau
        n_missed = int((positive & ~blocked).sum())
        bound = binomial_upper_bound(n_missed, n_pos, delta)
        if bound <= alpha:
            best = {
                "threshold": float(tau),
                "block_rate": float(blocked.mean()),
                "empirical_miss_rate": n_missed / n_pos,
                "risk_bound": bound,
                "n_positive": n_pos,
                "n_missed": n_missed,
                "n_calibration": n,
            }
        else:
            break

    if best is None:
        return {
            "threshold": 0.0,
            "block_rate": 1.0,
            "empirical_miss_rate": 0.0,
            "risk_bound": binomial_upper_bound(0, n_pos, delta),
            "n_positive": n_pos,
            "n_missed": 0,
            "n_calibration": n,
            "certifiable": False,
        }
    best["certifiable"] = True
    return best


def conformal_quantile(scores: np.ndarray, alpha: float) -> Tuple[float, bool]:
    """Split-conformal quantile with the finite-sample ``(n + 1)`` correction.

    Returns ``(qhat, feasible)``. ``feasible`` is False when the calibration set is
    too small for the requested ``alpha`` -- fewer than ``ceil(1/alpha) - 1`` points
    -- in which case the only valid answer is the trivial one that always covers,
    and saying so is better than quietly returning an under-covering quantile.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        return float("inf"), False
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf"), False
    return float(np.quantile(scores, level, method="higher")), True


# --------------------------------------------------------------------------------------
# Pulling scores out of Laya answers
# --------------------------------------------------------------------------------------

def _probs_choice(answer: Mapping[str, Any]) -> Tuple[List[str], np.ndarray]:
    probs = answer.get("probabilities") or {}
    keys = list(probs.keys())
    return keys, np.asarray([float(probs[k]) for k in keys], dtype=np.float64)


def _probs_noul(answer: Mapping[str, Any]) -> Tuple[List[str], np.ndarray]:
    p_true = float(answer.get("noul", 0.0))
    return ["false", "true"], np.asarray([1.0 - p_true, p_true], dtype=np.float64)


def _probs_score(answer: Mapping[str, Any]) -> Tuple[List[str], np.ndarray]:
    probs = answer.get("probabilities") or {}
    keys = sorted(probs.keys(), key=lambda s: int(s))
    return keys, np.asarray([float(probs[k]) for k in keys], dtype=np.float64)


def _answer_distribution(answer: Mapping[str, Any]) -> Tuple[str, List[str], np.ndarray]:
    """(question type, option keys, probability vector) for any Laya answer."""
    qtype = answer.get("type")
    if qtype == "choice":
        keys, p = _probs_choice(answer)
    elif qtype == "noul":
        keys, p = _probs_noul(answer)
    elif qtype == "score":
        keys, p = _probs_score(answer)
    else:
        raise ValueError("unknown answer type %r; expected choice, score or noul" % (qtype,))
    if p.size == 0:
        raise ValueError("answer for a %s question carries no probabilities" % qtype)
    return qtype, keys, p


def _normalise_label(qtype: str, keys: Sequence[str], label: Any) -> int:
    """Index of the gold option, accepting the spellings a caller's data actually uses."""
    if qtype == "noul":
        if isinstance(label, str):
            low = label.strip().lower()
            if low in ("true", "yes", "1", "t", "y"):
                return 1
            if low in ("false", "no", "0", "f", "n"):
                return 0
            raise ValueError("noul label %r is not a boolean spelling" % (label,))
        return 1 if bool(label) else 0

    if qtype == "score":
        if isinstance(label, bool):
            raise ValueError("score label must be an ordinal level, not a bool")
        if isinstance(label, (int, np.integer)):
            idx = int(label)
        elif isinstance(label, float) and float(label).is_integer():
            idx = int(label)
        elif isinstance(label, str) and label.strip().lstrip("-").isdigit():
            idx = int(label.strip())
        else:
            raise ValueError("score label %r is not an integer level" % (label,))
        if not (0 <= idx < len(keys)):
            raise ValueError("score label %d is outside the %d declared levels" % (idx, len(keys)))
        return idx

    # choice
    label = str(label)
    try:
        return list(keys).index(label)
    except ValueError:
        raise ValueError("choice label %r is not one of the criteria %s"
                         % (label, list(keys))) from None


def _collect(results: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]],
             qid: str) -> Tuple[str, List[str], np.ndarray, np.ndarray]:
    """Stack one question's probability rows and gold indices across the calibration set.

    Rows whose label is missing are skipped, so a partially labelled set still fits
    the questions it does cover instead of failing outright.
    """
    rows: List[np.ndarray] = []
    gold: List[int] = []
    qtype: Optional[str] = None
    keys: Optional[List[str]] = None

    for i, (res, lab) in enumerate(zip(results, labels)):
        answers = res.get("answers", res) if isinstance(res, Mapping) else {}
        if qid not in answers:
            continue
        if not isinstance(lab, Mapping) or qid not in lab or lab[qid] is None:
            continue
        t, k, p = _answer_distribution(answers[qid])
        if qtype is None:
            qtype, keys = t, k
        elif t != qtype:
            raise ValueError("question %r changes type between rows: %s then %s" % (qid, qtype, t))
        elif list(k) != list(keys or []):
            raise ValueError("question %r changes its options at row %d: %s then %s"
                             % (qid, i, keys, k))
        rows.append(p)
        gold.append(_normalise_label(qtype, keys or [], lab[qid]))

    if qtype is None or not rows:
        raise ValueError("no labelled rows for question %r" % (qid,))
    return qtype, list(keys or []), np.vstack(rows), np.asarray(gold, dtype=np.int64)


# --------------------------------------------------------------------------------------
# Per-question gate
# --------------------------------------------------------------------------------------

class QuestionGate:
    """The fitted guarantee for one question id.

    Built by :meth:`ConformalGate.calibrate`; construct one directly only to hand-fit
    a threshold you already trust.
    """

    def __init__(self, qid: str, qtype: str, mode: str, alpha: float, delta: float,
                 options: Sequence[str], params: Mapping[str, Any],
                 diagnostics: Optional[Mapping[str, Any]] = None):
        if mode not in SUPPORTED_MODES:
            raise ValueError("mode %r is not one of %s" % (mode, list(SUPPORTED_MODES)))
        self.qid = qid
        self.qtype = qtype
        self.mode = mode
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.options = list(options)
        self.params = dict(params)
        self.diagnostics = dict(diagnostics or {})

    # -- fitting ------------------------------------------------------------------

    @classmethod
    def fit(cls, qid: str, qtype: str, options: Sequence[str], probs: np.ndarray,
            gold: np.ndarray, mode: str, alpha: float, delta: float,
            set_method: str = "lac") -> "QuestionGate":
        n = int(probs.shape[0])
        if mode == "selective":
            scores = probs.max(axis=1)
            correct = probs.argmax(axis=1) == gold
            fit = selective_threshold(scores, correct, alpha, delta)
            diagnostics = {
                "n_calibration": n,
                "coverage": fit["coverage"],
                "empirical_joint_risk": fit["empirical_joint_risk"],
                "empirical_selective_risk": fit["empirical_selective_risk"],
                "risk_bound": fit["risk_bound"],
                "certifiable": fit["certifiable"],
                "base_accuracy": float(correct.mean()),
                "min_calibration": min_calibration_size(alpha, delta),
            }
            if not fit["certifiable"]:
                diagnostics["shortfall"] = fit["shortfall"]
            params = {"threshold": fit["threshold"]}

        elif mode == "miss":
            if qtype != "noul":
                raise ValueError("miss control applies to noul questions; %r is %s" % (qid, qtype))
            fit = miss_threshold(probs[:, 1], gold == 1, alpha, delta)
            diagnostics = {
                "n_calibration": n,
                "block_rate": fit["block_rate"],
                "empirical_miss_rate": fit["empirical_miss_rate"],
                "risk_bound": fit["risk_bound"],
                "n_positive": fit["n_positive"],
                "certifiable": fit["certifiable"],
            }
            params = {"threshold": fit["threshold"]}

        elif mode == "set":
            nonconf = _set_scores(probs, gold, set_method)
            qhat, feasible = conformal_quantile(nonconf, alpha)
            sizes = _set_sizes(probs, qhat, set_method)
            covered = _set_contains(probs, gold, qhat, set_method)
            diagnostics = {
                "n_calibration": n,
                "mean_set_size": float(sizes.mean()),
                "empirical_coverage": float(covered.mean()),
                "singleton_rate": float((sizes == 1).mean()),
                "feasible": feasible,
                "min_calibration_for_alpha": int(math.ceil(1.0 / alpha) - 1),
            }
            params = {"qhat": qhat, "method": set_method}

        elif mode == "interval":
            if qtype != "score":
                raise ValueError("interval control applies to score questions; %r is %s"
                                 % (qid, qtype))
            levels = np.arange(probs.shape[1], dtype=np.float64)
            expected = (probs * levels).sum(axis=1)
            residuals = np.abs(expected - gold.astype(np.float64))
            qhat, feasible = conformal_quantile(residuals, alpha)
            covered = residuals <= qhat
            diagnostics = {
                "n_calibration": n,
                "half_width": qhat,
                "empirical_coverage": float(covered.mean()),
                "median_abs_error": float(np.median(residuals)),
                "feasible": feasible,
                "min_calibration_for_alpha": int(math.ceil(1.0 / alpha) - 1),
            }
            params = {"half_width": qhat, "levels": int(probs.shape[1])}

        else:  # pragma: no cover - guarded by SUPPORTED_MODES
            raise ValueError("unsupported mode %r" % (mode,))

        return cls(qid, qtype, mode, alpha, delta, options, params, diagnostics)

    # -- application --------------------------------------------------------------

    def apply(self, answer: Mapping[str, Any]) -> Dict[str, Any]:
        """The ``gate`` block for one answer: what was decided, and on what guarantee."""
        qtype, keys, p = _answer_distribution(answer)
        if qtype != self.qtype:
            raise ValueError("gate for %r was fitted on a %s question but got a %s answer"
                             % (self.qid, self.qtype, qtype))
        if list(keys) != self.options:
            raise ValueError(
                "gate for %r was fitted on options %s but the answer carries %s; a gate is "
                "only valid for the question it was calibrated on"
                % (self.qid, self.options, list(keys)))

        if self.mode == "selective":
            tau = float(self.params["threshold"])
            top = float(p.max())
            accepted = bool(top >= tau)
            return {
                "mode": "selective",
                "accepted": accepted,
                "abstain": not accepted,
                "top_probability": round(top, 6),
                "threshold": tau,
                "alpha": self.alpha,
                "delta": self.delta,
                "guarantee": ("at most %.4g of all decisions are accepted and wrong, "
                              "with confidence %.4g" % (self.alpha, 1.0 - self.delta)),
                "certifiable": bool(self.diagnostics.get("certifiable", True)),
            }

        if self.mode == "miss":
            tau = float(self.params["threshold"])
            p_true = float(p[1])
            blocked = bool(p_true >= tau)
            return {
                "mode": "miss",
                "blocked": blocked,
                "accepted": not blocked,
                "abstain": blocked,
                "probability": round(p_true, 6),
                "threshold": tau,
                "alpha": self.alpha,
                "delta": self.delta,
                "guarantee": ("at most %.4g of true positives pass this gate, with "
                              "confidence %.4g" % (self.alpha, 1.0 - self.delta)),
                "certifiable": bool(self.diagnostics.get("certifiable", True)),
            }

        if self.mode == "set":
            qhat = float(self.params["qhat"])
            method = str(self.params.get("method", "lac"))
            members = _members(p, qhat, method)
            labels = [keys[i] for i in members]
            return {
                "mode": "set",
                "method": method,
                "prediction_set": labels,
                "set_size": len(labels),
                "accepted": len(labels) == 1,
                "abstain": len(labels) != 1,
                "alpha": self.alpha,
                "guarantee": ("the true option is in this set with probability at least "
                              "%.4g" % (1.0 - self.alpha,)),
                "certifiable": bool(self.diagnostics.get("feasible", True)),
            }

        # interval
        half = float(self.params["half_width"])
        levels = np.arange(p.size, dtype=np.float64)
        expected = float((p * levels).sum())
        lo = max(0.0, expected - half)
        hi = min(float(p.size - 1), expected + half)
        return {
            "mode": "interval",
            "point": round(expected, 6),
            "interval": [round(lo, 6), round(hi, 6)],
            "half_width": half,
            "accepted": bool(half <= 0.5),
            "abstain": bool(half > 0.5),
            "alpha": self.alpha,
            "guarantee": ("the true level falls in this interval with probability at least "
                          "%.4g" % (1.0 - self.alpha,)),
            "certifiable": bool(self.diagnostics.get("feasible", True)),
        }

    # -- serialisation ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "qtype": self.qtype,
            "mode": self.mode,
            "alpha": self.alpha,
            "delta": self.delta,
            "options": list(self.options),
            "params": _jsonable(self.params),
            "diagnostics": _jsonable(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuestionGate":
        return cls(
            qid=payload["qid"],
            qtype=payload["qtype"],
            mode=payload["mode"],
            alpha=payload["alpha"],
            delta=payload["delta"],
            options=payload.get("options", []),
            params=payload.get("params", {}),
            diagnostics=payload.get("diagnostics", {}),
        )

    def __repr__(self) -> str:
        return "QuestionGate(%r, %s, mode=%s, alpha=%g)" % (
            self.qid, self.qtype, self.mode, self.alpha)


# --------------------------------------------------------------------------------------
# Conformal set helpers
# --------------------------------------------------------------------------------------

def _set_scores(probs: np.ndarray, gold: np.ndarray, method: str) -> np.ndarray:
    """Non-conformity score of the gold label on each calibration row."""
    rows = np.arange(probs.shape[0])
    if method == "lac":
        return 1.0 - probs[rows, gold]
    if method == "aps":
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        rank = np.argmax(order == gold[:, None], axis=1)
        return cum[rows, rank]
    raise ValueError("set method must be 'lac' or 'aps'; got %r" % (method,))


def _members(p: np.ndarray, qhat: float, method: str) -> List[int]:
    """Indices in the prediction set for one probability row."""
    if not np.isfinite(qhat):
        return list(range(p.size))
    if method == "lac":
        idx = np.nonzero(p >= 1.0 - qhat)[0]
        if idx.size == 0:
            # An empty set is valid but useless to a caller: fall back to the top option,
            # which only ever shrinks coverage on rows the guarantee already allowed to miss.
            return [int(p.argmax())]
        return [int(i) for i in idx]
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    # Smallest prefix whose mass reaches qhat, always at least one option.
    take = int(np.searchsorted(cum, qhat, side="left")) + 1
    take = max(1, min(take, p.size))
    return [int(i) for i in order[:take]]


def _set_sizes(probs: np.ndarray, qhat: float, method: str) -> np.ndarray:
    return np.asarray([len(_members(row, qhat, method)) for row in probs], dtype=np.int64)


def _set_contains(probs: np.ndarray, gold: np.ndarray, qhat: float, method: str) -> np.ndarray:
    return np.asarray([g in _members(row, qhat, method) for row, g in zip(probs, gold)], dtype=bool)


def _jsonable(obj: Any) -> Any:
    """NumPy scalars and non-finite floats survive a JSON round trip."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return obj


def _unjson(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _unjson(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unjson(v) for v in obj]
    if obj == "Infinity":
        return float("inf")
    if obj == "-Infinity":
        return float("-inf")
    return obj


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------

class ConformalGate:
    """A risk contract over a whole question set, fitted once and applied per request."""

    def __init__(self, gates: Mapping[str, QuestionGate], alpha: float, delta: float,
                 meta: Optional[Mapping[str, Any]] = None):
        self.gates: Dict[str, QuestionGate] = dict(gates)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.meta = dict(meta or {})

    # -- fitting ------------------------------------------------------------------

    @classmethod
    def calibrate(cls, results: Sequence[Mapping[str, Any]],
                  labels: Sequence[Mapping[str, Any]],
                  alpha: float = 0.05,
                  delta: float = 0.05,
                  mode: Union[str, Mapping[str, str]] = "auto",
                  set_method: str = "lac",
                  questions: Optional[Iterable[str]] = None) -> "ConformalGate":
        """Fit a gate from model outputs and gold labels.

        Args:
            results: What ``Agent.predict``/``predict_batch``/``Router.predict`` returned
                for a held-out split. A bare ``{qid: answer}`` mapping works too.
            labels: Gold answers, aligned with ``results`` by index. Each entry maps a
                question id to its gold value: the criterion key for ``choice``, a bool
                for ``noul``, the integer level for ``score``. Missing ids are skipped,
                so one labelling pass can cover a subset of the questions.
            alpha: The risk you are willing to carry. 0.05 means "at most 5%".
            delta: Confidence on the risk bound, for the modes that use one
                (``selective`` and ``miss``). 0.05 means the bound holds 95% of the time.
            mode: ``"auto"``, one mode for every question, or a per-question mapping.
                ``auto`` picks ``selective`` for ``choice`` and ``noul``, ``interval``
                for ``score`` -- the thresholds most callers want on first contact.
            set_method: ``"lac"`` for the smallest sets, ``"aps"`` for better coverage
                on hard inputs at the cost of larger ones.
            questions: Restrict fitting to these ids. Defaults to every id that appears
                in both ``results`` and ``labels``.

        Raises:
            ValueError: if the two sequences differ in length, or a requested question
                has no labelled rows.
        """
        results = list(results)
        labels = list(labels)
        if len(results) != len(labels):
            raise ValueError("results and labels must be the same length; got %d and %d"
                             % (len(results), len(labels)))
        if not results:
            raise ValueError("calibration needs at least one labelled example")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1); got %r" % (alpha,))
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must lie in (0, 1); got %r" % (delta,))

        if questions is None:
            seen: List[str] = []
            for res in results:
                answers = res.get("answers", res) if isinstance(res, Mapping) else {}
                for qid in answers:
                    if qid not in seen:
                        seen.append(qid)
            labelled = {qid for lab in labels if isinstance(lab, Mapping) for qid in lab}
            qids = [q for q in seen if q in labelled]
        else:
            qids = list(questions)
        if not qids:
            raise ValueError("no question id appears in both results and labels")

        fitted: Dict[str, QuestionGate] = {}
        for qid in qids:
            qtype, options, probs, gold = _collect(results, labels, qid)
            qmode = mode.get(qid, "auto") if isinstance(mode, Mapping) else mode
            if qmode == "auto":
                qmode = "interval" if qtype == "score" else "selective"
            fitted[qid] = QuestionGate.fit(qid, qtype, options, probs, gold,
                                           qmode, alpha, delta, set_method)

        meta = {
            "schema_version": _SCHEMA_VERSION,
            "n_calibration": len(results),
            "set_method": set_method,
        }
        return cls(fitted, alpha, delta, meta)

    # -- application --------------------------------------------------------------

    def apply(self, result: Mapping[str, Any], strict: bool = False) -> Dict[str, Any]:
        """Attach a ``gate`` block to every answer the gate covers.

        Returns a new result dict; the input is not mutated. Answers with no fitted
        gate pass through untouched unless ``strict`` is set, which raises instead --
        use it in production to catch a question set that drifted away from the gate
        it was calibrated against.

        The record-level ``gate`` block carries ``family_alpha``: the union bound over
        every gated question. It is *not* ``alpha`` unless exactly one question is gated.
        See :meth:`family_risk`.
        """
        answers = result.get("answers")
        if answers is None:
            raise ValueError("expected a Laya result with an 'answers' key")

        out_answers: Dict[str, Any] = {}
        accepted_all = True
        gated_any = False
        for qid, answer in answers.items():
            gate = self.gates.get(qid)
            if gate is None:
                if strict:
                    raise ValueError("no gate fitted for question %r; calibrate it or drop "
                                     "it from the request" % (qid,))
                out_answers[qid] = dict(answer)
                continue
            block = gate.apply(answer)
            merged = dict(answer)
            merged["gate"] = block
            out_answers[qid] = merged
            gated_any = True
            accepted_all = accepted_all and bool(block.get("accepted", True))

        out = dict(result)
        out["answers"] = out_answers
        gated_ids = [q for q in answers if q in self.gates]
        out["gate"] = dict(
            {
                "accepted": accepted_all if gated_any else None,
                "alpha": self.alpha,
                "delta": self.delta,
                "questions_gated": len(gated_ids),
                "questions_ungated": sum(1 for q in answers if q not in self.gates),
            },
            **self.family_risk(gated_ids)
        )
        return out

    def family_risk(self, qids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """The risk carried by accepting a *whole record*, not one answer.

        Each question's gate is certified at its own ``alpha``. Reading the record-level
        ``accepted`` flag as though it inherited that number is the single easiest way to
        misuse this module: accepting five answers at 2% each risks a record that is
        accepted and wrong somewhere at up to 10%, by the union bound. The bound is loose
        when errors co-occur and tight when they are disjoint, and it is the only
        distribution-free statement available without modelling the dependence between
        questions -- so it is the one reported.

        ``family_delta`` unions only over the modes that carry a confidence parameter
        (``selective`` and ``miss``). Split-conformal ``set`` and ``interval`` coverage is
        marginal over the calibration draw and has no ``delta`` to spend.

        Returns a block with ``family_alpha``, ``family_delta`` and a plain-language
        ``family_guarantee``; both are capped at 1.0, where the honest reading is that the
        record-level flag carries no usable guarantee and the per-question blocks should
        be read individually.
        """
        ids = list(self.gates) if qids is None else [q for q in qids if q in self.gates]
        alpha = sum(self.gates[q].alpha for q in ids)
        delta = sum(self.gates[q].delta for q in ids
                    if self.gates[q].mode in ("selective", "miss"))
        alpha = min(1.0, alpha)
        delta = min(1.0, delta)
        if not ids:
            guarantee = "no gated questions in this record; the record-level flag is not a claim"
        elif alpha >= 1.0:
            guarantee = ("%d gated questions at these alphas union to a vacuous record-level "
                         "bound; read each question's gate block instead" % len(ids))
        else:
            guarantee = ("at most %.4g of accepted records are wrong on at least one of the "
                         "%d gated questions, with confidence %.4g (union bound)"
                         % (alpha, len(ids), 1.0 - delta))
        return {
            "family_alpha": alpha,
            "family_delta": delta,
            "family_guarantee": guarantee,
        }

    def apply_batch(self, results: Sequence[Mapping[str, Any]],
                    strict: bool = False) -> List[Dict[str, Any]]:
        """``apply`` across a ``predict_batch`` result, in order."""
        return [self.apply(r, strict=strict) for r in results]

    def accepts(self, result: Mapping[str, Any], qid: str) -> bool:
        """Whether one question's answer clears its gate. Ungated questions accept."""
        gate = self.gates.get(qid)
        if gate is None:
            return True
        answers = result.get("answers", result)
        if qid not in answers:
            raise KeyError("result carries no answer for question %r" % (qid,))
        return bool(gate.apply(answers[qid]).get("accepted", True))

    # -- reporting ----------------------------------------------------------------

    def report(self) -> str:
        """A table of what each gate bought, for a commit message or a design review."""
        lines = ["Laya conformal gate  alpha=%g  delta=%g  n=%s"
                 % (self.alpha, self.delta, self.meta.get("n_calibration", "?"))]
        lines.append("")
        header = "%-22s %-9s %-10s %-24s %s" % ("question", "type", "mode", "operating point", "guarantee")
        lines.append(header)
        lines.append("-" * len(header))
        for qid, gate in self.gates.items():
            d = gate.diagnostics
            if gate.mode == "selective":
                op = "keep %.1f%% @ tau=%.3f" % (100.0 * float(d.get("coverage", 0.0)),
                                                 float(gate.params["threshold"]))
                guar = "accepted-and-wrong <= %.3g of all" % gate.alpha
            elif gate.mode == "miss":
                op = "block %.1f%% @ tau=%.3f" % (100.0 * float(d.get("block_rate", 0.0)),
                                                  float(gate.params["threshold"]))
                guar = "miss <= %.3g of positives" % gate.alpha
            elif gate.mode == "set":
                op = "set size %.2f, %.0f%% single" % (float(d.get("mean_set_size", 0.0)),
                                                       100.0 * float(d.get("singleton_rate", 0.0)))
                guar = "coverage >= %.3g" % (1.0 - gate.alpha)
            else:
                op = "+/- %.2f levels" % float(gate.params["half_width"])
                guar = "coverage >= %.3g" % (1.0 - gate.alpha)
            if not _certified(gate):
                guar += "  [NOT CERTIFIABLE: too little calibration data]"
            lines.append("%-22s %-9s %-10s %-24s %s" % (qid[:22], gate.qtype, gate.mode, op, guar))
        return "\n".join(lines)

    # -- serialisation ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "alpha": self.alpha,
            "delta": self.delta,
            "meta": _jsonable(self.meta),
            "gates": {qid: g.to_dict() for qid, g in self.gates.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformalGate":
        version = payload.get("schema_version")
        if version is not None and int(version) > _SCHEMA_VERSION:
            raise ValueError("gate was written by a newer Laya (schema %s > %s); upgrade laya "
                             "to load it" % (version, _SCHEMA_VERSION))
        payload = _unjson(dict(payload))
        gates = {qid: QuestionGate.from_dict(g) for qid, g in payload.get("gates", {}).items()}
        return cls(gates, payload.get("alpha", 0.05), payload.get("delta", 0.05),
                   payload.get("meta", {}))

    def save(self, path: str) -> None:
        """Write the gate to JSON. It carries no weights, so it ships anywhere."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> "ConformalGate":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __contains__(self, qid: object) -> bool:
        return qid in self.gates

    def __getitem__(self, qid: str) -> QuestionGate:
        return self.gates[qid]

    def __len__(self) -> int:
        return len(self.gates)

    def __repr__(self) -> str:
        return "ConformalGate(%d questions, alpha=%g, delta=%g)" % (
            len(self.gates), self.alpha, self.delta)


def _certified(gate: QuestionGate) -> bool:
    d = gate.diagnostics
    return bool(d.get("certifiable", d.get("feasible", True)))
