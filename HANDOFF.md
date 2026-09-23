# Handoff: certified risk control

Branch `feat/conformal-risk-control`. `laya/conformal.py` plus its wiring into the package,
the HTTP server, the CLI, the test suite and CI. Version bumped to 0.4.0.

## Why this exists

Laya's headline claim is calibration: ECE 0.081 against Jev's 0.246. On its own that number
is marketing — a caller still has to guess a confidence threshold by hand, and nothing tells
them what the guess costs. This module converts the calibrated probability into a contract
someone can be held to: "at most 2% of the queue is auto-handled incorrectly, at 95%
confidence."

Nothing in the competitive set ships this. Jev returns a score. Llama Guard and ShieldGemma
return a score. semantic-router and SetFit return a score. **The gate is the differentiator,
not the encoder.**

Pure NumPy — no torch, no transformers — so it keeps the lightweight-import property, and a
fitted gate serialises to JSON an edge client can apply without a model.

## The open question from the previous handoff is closed

The prior round reported 3 of 7 configurations breaching their `delta` budget, verdict
"inconclusive, do not ship the guarantee language". That verdict was correct to make and
the suspected cause was the right one.

It was measurement. Fixed-sequence testing returns the *most permissive* threshold that
still passes, so true risk sits just under `alpha` by construction; scoring that against a
5,000-row sampled test set, where the estimate carries a standard error near 0.003, puts a
true risk of 0.048 above 0.05 a large fraction of the time regardless of validity. The tell
was that apparent breaches got *worse* as calibration data grew.

`research/conformal/validate_risk.py` re-runs it against population risk — labels drawn
`Bernoulli(c(p))` for a known link, so `R(tau) = E[1{p >= tau}(1 - c(p))]` is exact on a
frozen two-million-row pool with no label sampling anywhere in the loop. 1,000 trials per
configuration, across calibrated, overconfident and underconfident models, because a
distribution-free bound that only holds for honest models is not distribution-free.

**21 of 21 configurations hold inside budget. Worst cell: 49/1000, a 4.9% breach rate
against a 5% allowance.** Full table and method in `research/conformal/README.md`.

The earlier suspicion that the fixed-sequence walk returns the wrong endpoint was
unfounded, and worth not re-litigating: returning the *last* passing threshold is exactly
what FWER control licenses. Let `j*` be the first index whose true risk exceeds `alpha`; a
false rejection requires reaching and rejecting there, which happens with probability at
most `delta`. The guarantee language is safe to ship.

## What was fixed while wiring it up

- **The module docstring promised the conditional rate** — error among accepted — which
  `selective_threshold` explicitly refuses to certify and says so in its own docstring. The
  top-level claim was the one a reader meets first. Now states the joint rate, and notes
  that the joint bound caps the conditional at `alpha / coverage`.
- **`min_calibration_size` was defined twice**, the second silently shadowing the first.
  Identical bodies, so benign, but it is the function the "not certifiable" path depends on.
- **`ConformalGate.apply` inflated the record-level claim.** Five questions gated at 2% each
  is a record-level risk of up to 10% by the union bound, not 2%. The block now carries
  `family_alpha`, `family_delta` and `family_guarantee`; `family_delta` unions only over
  `selective` and `miss`, since split-conformal `set` and `interval` spend no `delta`.
  Inheriting the per-question number here is the easiest way to misuse risk control.
- **`miss_threshold` was public and documented but missing from `__all__`.**
- **CI never ran `tests/test_serve.py`.** It is a pure pytest module, so `python
  tests/test_serve.py` defines its tests, calls none, and exits 0 — which is why it was
  never added to the direct-invocation list. The HTTP surface was untested in CI. There is
  now a pytest step covering it and `test_truncation_direction.py`, and
  `test_load_errors.py` joined the direct list.

## What shipped

| | |
|---|---|
| `laya/conformal.py` | four modes: `selective`, `miss`, `set`, `interval` |
| `laya/__init__.py` | lazy exports; `import laya` stays torch-free, and so does applying a gate |
| `laya/serve.py` | `LAYA_GATE`, `LAYA_GATE_STRICT`, contract advertised on `GET /health` |
| `laya/cli.py` | `--gate PATH`, `--report` |
| `tests/test_conformal.py` | 152 checks, runs in ~1.2 s |
| `tests/test_serve.py` | 7 new gating tests |
| `tests/test_cli.py` | 11 new gating tests |
| `laya/drift.py` | `GateMonitor`: label-free expiry detection over a fitted gate |
| `tests/test_drift.py` | 76 checks, runs in ~0.7 s |
| `research/conformal/` | validation harness and its results |
| `README.md`, `BENCHMARKS.md` | the competitive claim, with the evidence behind it |

`interval` mode, never exercised before, now has fresh-split coverage measured: 0.9152 at
target 0.90, 0.9536 at target 0.95, and the infeasible path degrades to the full range,
which always covers.

Full suite: 29 of 31 files pass. The two that do not are environmental and pre-existing —
`test_fast.py` needs CUDA, `test_local_e2e.py` needs locally trained weights at
`~/laya_models`. Neither is in CI.

## Where the risk actually sits now

**Exchangeability, and now it is watched.** The bound holds for any distribution, but
calibration and serving traffic have to be drawn from the same one. `laya/drift.py` closes
this with two label-free tests: an exact two-sided binomial test of the observed acceptance
rate against the calibrated coverage, and a two-sample KS test of live scores against the
101-quantile sketch now stored in each gate's diagnostics.

Measured, because a monitor that cries wolf gets muted: **0.3% false positives** on
undrifted traffic against a 1% test level, and it catches a 2.2 → 2.0 shift in logit
separation in 136 of 150 windows, 150/150 at 1.8 and beyond. Both numbers are asserted in
`tests/test_drift.py`, not just measured once.

The limit is stated everywhere it is reported, including in `report()` itself: both tests
watch *scores*. A shift in the score-to-correctness link — same confidences, worse answers
— moves real risk while leaving both quiet. A clean report is "no evidence of expiry",
never "the guarantee still holds".

**Every number in `research/conformal/` is synthetic.** The bound is distribution-free, so
that is sound for validating the guarantee — validity cannot depend on the generator. It is
*not* evidence about coverage on real traffic. What `alpha = 0.02` costs in kept traffic on
an actual ticket queue is unmeasured, and that is the number a buyer will ask for first.

## Not started

- **Correctness-link drift**, the half `GateMonitor` cannot see. Detecting it needs labels,
  but not many: a small periodically-labelled audit sample would let the monitor test
  observed risk against `alpha` directly, turning "no evidence of expiry" into a real
  statement about the guarantee. This is the highest-value remaining item.
- `Agent.predict(gate=...)` convenience wiring (the `Router`/serve path covers the real use).
- A `LayaGate` LangChain runnable, alongside the existing `LayaRouter` / `LayaGuardrail`.
- Real-traffic coverage numbers on a public dataset with labels, to replace the synthetic
  coverage column in `BENCHMARKS.md`.
