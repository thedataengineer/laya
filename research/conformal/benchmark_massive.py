"""Coverage against risk budget on real traffic, not a generator.

`validate_risk.py` answers "does the bound hold", and has to be synthetic to do it -- the
whole point is that validity cannot depend on the data. This answers the other question,
the one a buyer asks first: **what does the guarantee cost on real inputs?**

    python research/conformal/benchmark_massive.py --per-lang 600 --langs en,de,fr

Dataset and prompt follow `research/eval/taut_eval.py`: `mteb/amazon_massive_intent`, test
split, the same "What is the user asking for in `utterance`?" rendering.

One deliberate difference. That harness draws a fresh option subset per case, which is the
right design for measuring raw accuracy. A gate cannot be fitted on it: a gate is valid
only for the question it was calibrated against, and `ConformalGate` refuses a question
whose options move between rows. So this uses the `--n-opts` most frequent intents as a
single **fixed** taxonomy and keeps the rows whose gold label is in it -- which is also
what a deployed decision actually looks like.

Method, and the part that matters: the scored cases are split in half by a fixed seed. A
gate is fitted on one half and every number reported comes from the **other**. A
coverage-versus-alpha curve measured on the calibration split is a description of that
split, not a forecast, and it is exactly the mistake this module exists to stop.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taut.conformal import ConformalGate  # noqa: E402

SEED = 20240617
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)


def render_label(key: str) -> str:
    return key.replace("_", " ").replace(".", ": ")


def fixed_taxonomy(ds, n_opts):
    """The `n_opts` most frequent intents, as one option set shared by every case."""
    counts: Dict[str, int] = {}
    for row in ds:
        counts[row["label_text"]] = counts.get(row["label_text"], 0) + 1
    ranked = sorted(counts, key=lambda k: (-counts[k], k))
    return sorted(ranked[:n_opts])


def build_cases(rows, taxonomy, limit):
    """One typed `choice` question per utterance, over a taxonomy that never moves."""
    question = {"intent": {
        "type": "choice",
        "instructions": "What is the user asking for in `utterance`?",
        "criteria": {o: render_label(o) for o in taxonomy},
    }}
    allowed = set(taxonomy)
    cases = []
    for row in rows:
        if row["label_text"] not in allowed:
            continue
        cases.append({"text": row["text"], "gold": row["label_text"],
                      "questions": question})
        if len(cases) >= limit:
            break
    return cases


def score(cases, model, device, batch):
    """Score every case. The taxonomy is fixed, so the whole batch shares one question
    and rides a shared forward pass rather than one pass per row."""
    import taut

    router = taut.Router(device=device, preload=False)
    questions = cases[0]["questions"]
    out = []
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        states = [{"text": c["text"]} for c in chunk]
        try:
            out.extend(router.predict_batch(states, questions, model=model))
        except AttributeError:
            out.extend(router.predict(st, questions, model=model) for st in states)
        print("  scored %d/%d" % (min(i + batch, len(cases)), len(cases)),
              end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return out


def measure(gate, results, labels, qid="intent"):
    """Held-out behaviour: what the gate did to data it had never seen."""
    n = accepted = wrong_accepted = 0
    covered = sizes = 0
    qgate = gate[qid]
    for res, gold in zip(results, labels):
        answer = res["answers"][qid]
        block = qgate.apply(answer)
        n += 1
        if qgate.mode == "selective":
            top = max(answer["probabilities"], key=answer["probabilities"].get)
            if block["accepted"]:
                accepted += 1
                wrong_accepted += top != gold
        else:
            covered += gold in block["prediction_set"]
            sizes += block["set_size"]
    if qgate.mode == "selective":
        return {"coverage": accepted / n, "joint_risk": wrong_accepted / n,
                "selective_risk": (wrong_accepted / accepted) if accepted else float("nan"),
                "n": n}
    return {"set_coverage": covered / n, "mean_set_size": sizes / n, "n": n}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--langs", default="en", help="comma list of MASSIVE languages")
    p.add_argument("--per-lang", type=int, default=600)
    p.add_argument("--n-opts", type=int, default=20)
    p.add_argument("--model", default=None, help="checkpoint name; default auto-routes")
    p.add_argument("--device", default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", default=None, help="write the table as JSON here")
    args = p.parse_args(argv)

    from datasets import load_dataset

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    taxonomy = None
    all_results: List[Dict[str, Any]] = []
    all_gold: List[str] = []
    per_lang_index: Dict[str, List[int]] = {}

    for lang in langs:
        print("== %s" % lang, file=sys.stderr)
        ds = load_dataset("mteb/amazon_massive_intent", lang, split="test")
        if taxonomy is None:
            # Fixed once, from the first language, so every language answers the same
            # question and the gate stays valid across all of them.
            taxonomy = fixed_taxonomy(ds, args.n_opts)
            print("   taxonomy: %s" % ", ".join(taxonomy), file=sys.stderr)
        cases = build_cases(list(ds), taxonomy, args.per_lang)
        results = score(cases, args.model, args.device, args.batch)
        start = len(all_results)
        all_results.extend(results)
        all_gold.extend(c["gold"] for c in cases)
        per_lang_index[lang] = list(range(start, len(all_results)))

    order = list(range(len(all_results)))
    random.Random(SEED).shuffle(order)
    half = len(order) // 2
    cal_i, test_i = order[:half], order[half:]
    cal_r = [all_results[i] for i in cal_i]
    cal_l = [{"intent": all_gold[i]} for i in cal_i]
    test_r = [all_results[i] for i in test_i]
    test_g = [all_gold[i] for i in test_i]

    base_acc = sum(
        max(r["answers"]["intent"]["probabilities"],
            key=r["answers"]["intent"]["probabilities"].get) == g
        for r, g in zip(test_r, test_g)) / len(test_r)

    print("\nMASSIVE intent, fixed %d-option taxonomy, %d scored (%s)"
          % (args.n_opts, len(all_results), ", ".join(langs)))
    print("calibration %d / held out %d   ungated accuracy %.3f"
          % (len(cal_r), len(test_r), base_acc))

    table: Dict[str, Any] = {"langs": langs, "n_opts": args.n_opts,
                             "taxonomy": taxonomy,
                             "n_total": len(all_results), "base_accuracy": base_acc,
                             "selective": [], "set": []}

    print("\nselective -- 'at most alpha of ALL decisions are auto-handled and wrong'")
    print("%-7s %-10s %-12s %-14s %-14s %s"
          % ("alpha", "tau", "kept", "held-out risk", "among kept", "within budget"))
    for alpha in ALPHAS:
        gate = ConformalGate.calibrate(cal_r, cal_l, alpha=alpha, delta=0.05,
                                       mode={"intent": "selective"})
        g = gate["intent"]
        if not g.diagnostics.get("certifiable", True):
            print("%-7s %-10s %-12s %-14s %-14s %s"
                  % (alpha, "n/a", "0.0%", "n/a", "n/a", "NOT CERTIFIABLE"))
            table["selective"].append({"alpha": alpha, "certifiable": False})
            continue
        m = measure(gate, test_r, test_g)
        print("%-7s %-10.3f %-12s %-14s %-14s %s"
              % (alpha, g.params["threshold"], "%.1f%%" % (100 * m["coverage"]),
                 "%.4f" % m["joint_risk"], "%.4f" % m["selective_risk"],
                 "yes" if m["joint_risk"] <= alpha else "over (one draw)"))
        table["selective"].append({"alpha": alpha, "threshold": g.params["threshold"],
                                   "certifiable": True,
                                   "fit_coverage": g.diagnostics["coverage"], **m})

    print("\nset -- 'the true option is in the set with probability at least 1 - alpha'")
    print("%-7s %-16s %-16s %s" % ("alpha", "held-out coverage", "mean set size", "of 20"))
    for alpha in (0.01, 0.05, 0.10, 0.20):
        gate = ConformalGate.calibrate(cal_r, cal_l, alpha=alpha, mode={"intent": "set"})
        m = measure(gate, test_r, test_g)
        print("%-7s %-16s %-16.2f %s"
              % (alpha, "%.3f" % m["set_coverage"], m["mean_set_size"],
                 "target %.2f" % (1 - alpha)))
        table["set"].append({"alpha": alpha, **m})

    print("\nOne held-out sample is one draw; the guarantee is over repeated fits, so a")
    print("cell landing just over its budget is noise. validate_risk.py is what tests the")
    print("bound itself, across 1000 fits per configuration.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(table, fh, indent=2)
            fh.write("\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
