# Is the risk bound real?

`taut.conformal` makes a falsifiable claim: fit a gate at `alpha`, and the risk it
certifies is exceeded at most `delta` of the time — for any data distribution, at finite
sample size, with no asymptotics. A claim like that is worth exactly as much as the
evidence that it survives an attempt to break it.

`validate_risk.py` is that attempt. `python research/conformal/validate_risk.py`, about
three minutes, NumPy only.

## How it is measured, and why the method matters

The obvious experiment — fit on a calibration split, score on a test split, count how
often measured risk exceeds `alpha` — does not answer the question. Fixed-sequence
testing deliberately returns the *most permissive* threshold that still passes, so true
risk at the chosen threshold sits just under `alpha` by construction. Score that against
a 5,000-row test sample, where the risk estimate carries a standard error near 0.003, and
a true risk of 0.048 lands above 0.05 a large fraction of the time whether or not the
guarantee holds. The tell that this is measurement and not failure: apparent breaches get
*worse* as calibration data grows, which is backwards for a real validity failure and
exactly what estimator noise predicts.

So the generator here is explicit. Correctness is drawn `Bernoulli(c(p))` for a known
link `c`, which makes true risk at any threshold a population quantity:

```
R(tau) = E[ 1{p >= tau} * (1 - c(p)) ]
```

evaluated once on a frozen two-million-row score pool with no label sampling anywhere in
the loop. A breach is `R(tau_hat) > alpha`, full stop — no estimator between the
guarantee and the verdict.

Validity must not depend on the model being well calibrated, so each configuration runs
against three links: `c(p) = p` (calibrated), `p^1.6` (overconfident), and `p^0.6`
(underconfident). A distribution-free bound that only holds for honest models is not
distribution-free.

## Result

1,000 trials per configuration at `delta = 0.05`. The budget is 68/1000, the upper end of
a 99% binomial interval around the 50 breaches `delta` permits.

| mode | link / prevalence | alpha | n_cal | breaches | coverage | verdict |
|---|---|---|---|---|---|---|
| selective | calibrated | 0.10 | 500 | 44/1000 | 0.690 | OK |
| selective | calibrated | 0.05 | 500 | 42/1000 | 0.440 | OK |
| selective | calibrated | 0.05 | 3000 | 49/1000 | 0.511 | OK |
| selective | calibrated | 0.02 | 3000 | 10/1000 | 0.289 | OK |
| selective | calibrated | 0.01 | 5000 | 8/1000 | 0.189 | OK |
| selective | overconfident | 0.10 | 500 | 30/1000 | 0.547 | OK |
| selective | overconfident | 0.05 | 500 | 30/1000 | 0.341 | OK |
| selective | overconfident | 0.05 | 3000 | 35/1000 | 0.399 | OK |
| selective | overconfident | 0.02 | 3000 | 5/1000 | 0.221 | OK |
| selective | overconfident | 0.01 | 5000 | 8/1000 | 0.143 | OK |
| selective | underconfident | 0.10 | 500 | 30/1000 | 0.867 | OK |
| selective | underconfident | 0.05 | 500 | 24/1000 | 0.571 | OK |
| selective | underconfident | 0.05 | 3000 | 30/1000 | 0.660 | OK |
| selective | underconfident | 0.02 | 3000 | 19/1000 | 0.385 | OK |
| selective | underconfident | 0.01 | 5000 | 13/1000 | 0.254 | OK |
| miss | prevalence 0.20 | 0.10 | 2000 | 26/1000 | block 0.229 | OK |
| miss | prevalence 0.20 | 0.05 | 2000 | 36/1000 | block 0.287 | OK |
| miss | prevalence 0.20 | 0.01 | 5000 | 16/1000 | block 0.451 | OK |
| miss | prevalence 0.05 | 0.10 | 2000 | 17/1000 | block 0.129 | OK |
| miss | prevalence 0.05 | 0.05 | 2000 | 23/1000 | block 0.250 | OK |
| miss | prevalence 0.05 | 0.01 | 5000 | 0/1000 | block 1.000 | OK |

21 of 21 configurations inside budget. The worst cell is 49/1000, a 4.9% breach rate
against a 5% allowance.

Two things in that table are worth reading carefully.

**The bound is tight, not slack.** Several cells sit just under budget rather than far
below it. That is the procedure working as designed: fixed-sequence testing returns the
most permissive threshold that still passes, so it spends the risk allowance it is given
rather than hoarding it. A gate that always breached at 0.5% would be leaving coverage on
the table for no gain in safety.

**The last row is the gate refusing to answer.** At `alpha = 0.01` with 5% prevalence,
5,000 calibration rows carry only ~250 positives, below the 299 that
`min_calibration_size(0.01, 0.05)` requires. Rather than return a threshold it cannot
stand behind, the gate blocks everything — zero breaches, and a block rate of 1.0 that
tells the operator plainly that the budget is unaffordable at this sample size. Refusing
is a feature; it is the difference between a guarantee and a number.

## What this does not establish

The synthetic generator draws each row independently. Exchangeability is the one
assumption conformal prediction genuinely needs, and real traffic violates it under drift
— a gate fitted on last quarter's tickets carries no guarantee on this quarter's if the
mix moved. Refit on a recent labelled split; the JSON gate is cheap to regenerate because
it holds no weights.

`set` and `interval` modes are not in this table. They are split-conformal, whose
finite-sample validity is a theorem about order statistics rather than a bound to be
stress-tested, and `tests/test_conformal.py` checks their fresh-split coverage directly.
