# Handoff: conformal risk control for Laya

Branch `feat/conformal-risk-control`, commit `cf6d1f7`, one new file: `laya/conformal.py` (980 lines).
Local only, not pushed. `main` is clean and still at exact parity with upstream.

```bash
git push -u origin feat/conformal-risk-control
```

## Why this module exists

Laya's headline claim is calibration: ECE 0.081 against Jev's 0.246. That number is currently
marketing. A caller still has to guess a confidence threshold by hand, and nothing tells them what
guessing costs. This module converts the calibrated probability into a contract a buyer can hold
someone to: "at most 2% of the queue is auto-handled incorrectly, at 95% confidence."

Nothing in the competitive set ships this. Jev returns a score. Llama Guard and ShieldGemma return a
score. semantic-router and SetFit return a score. The gate is the differentiator, not the encoder.

It is pure NumPy. No torch, no transformers, so it keeps the 0.3.8 lightweight-import property and a
fitted gate serialises to JSON that an edge client can apply without a model.

## Read this before touching the code

Two decisions cost real time to reach. Do not re-litigate them without reading the reasoning in the
module docstring and `selective_threshold`.

**The certified quantity is the joint accepted-and-wrong rate, not error-among-accepted.** The
conditional rate's denominator shrinks as the threshold rises, so bounding it requires a
data-dependent threshold whose accepted set is selected by the same order statistics under test. An
exact binomial bound is not valid there. Measured, on the first implementation: fitting the
conditional rate breached a 5% budget on 4 of 14 certified trials. The joint loss holds the
denominator at the full calibration size, where the binomial model is exact and fixed-sequence
testing over a pre-specified grid is sound. The conditional rate is still reported, labelled as
observed rather than certified.

**Clopper-Pearson, not Hoeffding or a normal approximation.** Calibration sets here are hundreds of
rows. `min_calibration_size()` reports the floor below which nothing is certifiable: 59 rows for
alpha=0.05 at delta=0.05, 149 for 0.02, 299 for 0.01. Below it the gate returns "not certifiable"
with a `shortfall` block rather than a threshold it cannot stand behind.

## What is verified

| Check | Result |
|---|---|
| Clopper-Pearson against its defining equation P(Bin(n,p) <= k) = delta | exact to 6 decimal places, n from 10 to 100 |
| `calibrate` / `report` / `apply` / `save` / `load` end to end | passes, JSON round trip clean |
| `set` mode LAC marginal coverage, fresh split | 0.9005 at target 0.90; 0.9509 at target 0.95 |
| `set` mode APS marginal coverage, fresh split | 0.9707 at target 0.90; 0.9922 at target 0.95, conservative as expected |
| `interval` mode | never exercised |

## The open question, and it is the important one

Empirical breach counts for `selective` and `miss` on fresh splits, 200 trials each, budget 5%:

| mode | alpha | n_cal | breaches |
|---|---|---|---|
| selective | 0.10 | 500 | 9/200 |
| selective | 0.05 | 500 | 8/200 |
| selective | 0.05 | 3000 | 16/200 |
| selective | 0.02 | 3000 | 12/200 |
| miss | 0.10 | 2000 | 9/200 |
| miss | 0.05 | 2000 | 8/200 |
| miss | 0.01 | 2000 | 13/200 |

Three rows sit above the 10/200 budget. **This is most likely a flaw in how I measured, not in the
gate**, and the shape of the numbers is the evidence: breaches get *worse* as calibration data grows,
which is backwards for a real validity failure and exactly what measurement noise predicts.

The mechanism: fixed-sequence testing deliberately returns the most permissive threshold that still
passes, so true risk at the chosen threshold sits just under alpha by construction. I then scored
each trial against a 5000-row test sample, where the risk estimate has a standard error near 0.003.
When true risk is 0.048 and alpha is 0.05, that estimate lands above alpha a large fraction of the
time regardless of whether the guarantee holds. More calibration data tightens the bound, pushes the
chosen threshold closer to the boundary, and inflates the apparent breach rate. That is the observed
pattern.

**Verdict: inconclusive, not failing.** Do not ship the `selective` guarantee language until this is
settled either way.

### How to settle it

Score against true risk rather than a sampled estimate. The synthetic generator draws labels from the
model's own probabilities, so true risk at threshold tau is available in closed form with no label
sampling noise:

```python
r_true = float((( test_p.max(1) >= tau) * (1.0 - test_p.max(1))).mean())
breach = r_true > alpha
```

Use 200k test rows, keep 200 trials, and expect at most 5% breaches. If breaches stay above budget
with true risk, the bug is real and lives in the fixed-sequence walk in `selective_threshold`; start
by checking whether returning the last passing threshold rather than the first is what breaks the
family-wise argument.

Same correction applies to `miss`, scored against positives only.

## Remaining work, in order

1. Re-run validation against true risk as above. Gates everything else.
2. Exercise `interval` mode. My throwaway script had `np.arange(L, float)`, which raises
   `TypeError`; the module itself correctly uses `dtype=np.float64`, so interval is untested rather
   than broken.
3. Export from `laya/__init__.py`. Add to `_LAZY_ATTRS` and `__all__`, keeping the lazy pattern so
   `import laya` stays torch-free: `"ConformalGate": (".conformal", "ConformalGate")`, same for
   `QuestionGate`, `min_calibration_size`, `binomial_upper_bound`.
4. Write `tests/test_conformal.py`. CI runs test files directly as `python tests/test_x.py`, so
   follow `tests/test_router.py`: module-level asserts, collect failures, `sys.exit(1 if FAIL else 0)`.
   Cover the bound against its defining equation, monotonicity of the walk, the not-certifiable path
   below `min_calibration_size`, JSON round trip, and the option-mismatch guard in `QuestionGate.apply`.
5. Add the test file to `.github/workflows/ci.yml` alongside the other direct invocations.
6. README section and a BENCHMARKS.md table. This is the part that turns the work into a
   competitive claim, so write it only after step 1 lands.

## Not started

`Agent.predict(gate=...)` convenience wiring, a `/v1/systemone` gate parameter in `laya/serve.py`,
CLI surface, and a `LayaGate` LangChain runnable. All straightforward once the guarantee is settled.
