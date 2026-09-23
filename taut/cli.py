"""Command-line interface for testing Taut locally.

    taut "I was charged twice, please refund"           # routing decision only (no model download)
    taut "Refactor this service" --predict              # full answers, loads the checkpoint
    taut                                                # interactive mode
    taut "Mein Konto wurde zweimal belastet" --lang de  # explicit language
    taut "..." --predict --gate triage_gate.json         # apply a certified risk gate
    taut --gate triage_gate.json --report                # print what a saved gate certifies

    taut calibrate data.jsonl -q questions.json -o gate.json --alpha 0.02
                                                        # fit a certified gate from labelled data

Routing (the default) never downloads a checkpoint, so it works offline and
returns in milliseconds. --predict loads the routed checkpoint on first use,
which needs network access to the Hugging Face hub.
"""

import argparse
import json
import sys
from collections.abc import Mapping as Mapping_

import taut


def build_parser():
    parser = argparse.ArgumentParser(
        prog="taut",
        description="Test Taut locally: route or answer a request from the command line.",
    )
    parser.add_argument("text", nargs="*", help="the request text (omit for interactive mode)")
    parser.add_argument("--predict", action="store_true",
                        help="run the full prediction, not just the routing decision (downloads the checkpoint on first use)")
    parser.add_argument("--model", choices=sorted(taut.DEFAULT_MODELS),
                        help="force a checkpoint instead of auto-routing")
    parser.add_argument("--lang", help="force a language, e.g. en or de, instead of detecting it")
    parser.add_argument("--task", help="force a typed-decisions workflow instead of detecting it")
    parser.add_argument("--device", help="torch device, e.g. cpu or cuda")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    parser.add_argument("--gate", metavar="PATH",
                        help="apply a saved ConformalGate (see taut.conformal); with --report, "
                             "describe it instead of running a prediction")
    parser.add_argument("--report", action="store_true",
                        help="with --gate, print what the gate certifies and exit")
    return parser


def load_gate(path):
    """A saved gate, or None. Pure NumPy -- this never loads a checkpoint."""
    from taut.conformal import ConformalGate

    return ConformalGate.load(path)


# --------------------------------------------------------------------------------------
# taut calibrate
# --------------------------------------------------------------------------------------

CALIBRATE_HELP = """Fit a certified risk gate from labelled data.

    taut calibrate tickets.jsonl -q questions.json -o gate.json --alpha 0.02

DATA is JSON Lines, one labelled example per line, with the labels under "labels":

    {"text": "I was charged twice", "labels": {"intent": "billing", "urgent": false}}
    {"state": {"subject": "...", "body": "..."}, "labels": {"intent": "billing"}}

A line that already carries model output under "answers" is used as-is and no checkpoint
is loaded, so predictions can be computed once and reused across risk budgets:

    {"answers": {"intent": {...}}, "labels": {"intent": "billing"}}

QUESTIONS is the same JSON object you would pass to predict(). It is not needed when every
line already carries "answers".

By default a random half is held out: the gate is fitted on one half and *measured* on the
other, so the printed report carries evidence rather than only the promise. --split 1.0
fits on everything and skips that check.
"""


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError("%s line %d is not valid JSON: %s" % (path, i, e)) from None
            if not isinstance(row, dict):
                raise ValueError("%s line %d is not a JSON object" % (path, i))
            if "labels" not in row:
                raise ValueError("%s line %d has no 'labels' key; every calibration row "
                                 "needs gold answers" % (path, i))
            rows.append(row)
    if not rows:
        raise ValueError("%s contains no labelled rows" % path)
    return rows


def _parse_cost(specs):
    """--cost error=40,abstain=1  or  --cost intent:error=40,abstain=1 (repeatable)."""
    if not specs:
        return None
    flat, per_question = {}, {}
    for spec in specs:
        target, _, body = spec.rpartition(":") if ":" in spec else ("", "", spec)
        pairs = {}
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, value = part.partition("=")
            if not sep:
                raise ValueError("--cost expects key=value pairs, got %r" % part)
            try:
                pairs[key.strip()] = float(value)
            except ValueError:
                raise ValueError("--cost value for %r is not a number: %r"
                                 % (key.strip(), value)) from None
        if target:
            per_question[target] = pairs
        else:
            flat.update(pairs)
    if per_question and flat:
        raise ValueError("--cost takes either one spec for every question or per-question "
                         "specs, not both")
    return per_question or flat


def _states_and_labels(rows):
    states, labels = [], []
    for row in rows:
        labels.append(row["labels"])
        if "state" in row:
            states.append(row["state"])
        elif "text" in row:
            states.append({"text": row["text"]})
        else:
            states.append(None)
    return states, labels


def calibrate(argv):
    """Fit a gate from labelled JSONL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="taut calibrate", description=CALIBRATE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="labelled data as JSON Lines")
    parser.add_argument("-q", "--questions", help="question schema as JSON (not needed "
                                                  "when every row carries 'answers')")
    parser.add_argument("-o", "--output", default="gate.json", help="where to write the gate")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="the risk you are willing to carry (default: %(default)s)")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="confidence on the bound (default: %(default)s)")
    parser.add_argument("--mode", default="auto",
                        help="auto, or one mode for every question, or "
                             "'qid=mode,qid=mode' per question")
    parser.add_argument("--set-method", default="lac", choices=["lac", "aps"])
    parser.add_argument("--cost", action="append", metavar="SPEC",
                        help="operating costs, e.g. 'error=40,abstain=1' or "
                             "'intent:error=40,abstain=1' (repeatable)")
    parser.add_argument("--split", type=float, default=0.5, metavar="FRACTION",
                        help="fraction used to fit; the rest measures the result "
                             "(default: %(default)s, 1.0 to fit on everything)")
    parser.add_argument("--seed", type=int, default=0, help="split seed")
    parser.add_argument("--model", choices=sorted(taut.DEFAULT_MODELS))
    parser.add_argument("--device")
    parser.add_argument("--quiet", action="store_true", help="write the gate, print nothing")
    args = parser.parse_args(argv)

    if not (0.0 < args.split <= 1.0):
        print("taut: --split must lie in (0, 1]", file=sys.stderr)
        return 2

    try:
        rows = _read_jsonl(args.data)
        cost = _parse_cost(args.cost)
    except (OSError, ValueError) as e:
        print("taut: %s" % e, file=sys.stderr)
        return 2

    mode = args.mode
    if "=" in args.mode:
        mode = dict(part.split("=", 1) for part in args.mode.split(",") if part)

    states, labels = _states_and_labels(rows)
    have_answers = all("answers" in r for r in rows)
    if have_answers:
        results = [{"answers": r["answers"]} for r in rows]
    else:
        if any("answers" in r for r in rows):
            print("taut: some rows carry 'answers' and some do not; provide predictions "
                  "for all rows or none", file=sys.stderr)
            return 2
        if not args.questions:
            print("taut: --questions is required unless every row already carries 'answers'",
                  file=sys.stderr)
            return 2
        if any(s is None for s in states):
            print("taut: every row needs 'text' or 'state' when predictions are computed here",
                  file=sys.stderr)
            return 2
        try:
            with open(args.questions, "r", encoding="utf-8") as fh:
                questions = json.load(fh)
        except (OSError, ValueError) as e:
            print("taut: could not read %s (%s)" % (args.questions, e), file=sys.stderr)
            return 2
        if not args.quiet:
            print("Scoring %d labelled rows..." % len(states), file=sys.stderr)
        try:
            router = taut.Router(device=args.device, preload=False)
            results = router.predict_batch(states, questions, model=args.model)
        except AttributeError:
            results = [router.predict(s, questions, model=args.model) for s in states]
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            print("taut: could not run the model (%s)" % e, file=sys.stderr)
            return 2

    # Fit on one part, measure on the other: a gate reported only on the data it was fitted
    # to is a promise, and the point of this module is not to make promises.
    import random

    order = list(range(len(results)))
    random.Random(args.seed).shuffle(order)
    n_fit = len(order) if args.split >= 1.0 else max(1, int(round(args.split * len(order))))
    fit_idx, held_idx = order[:n_fit], order[n_fit:]

    try:
        gate = taut.ConformalGate.calibrate(
            [results[i] for i in fit_idx], [labels[i] for i in fit_idx],
            alpha=args.alpha, delta=args.delta, mode=mode,
            set_method=args.set_method, cost=cost)
    except (ValueError, KeyError) as e:
        print("taut: could not fit a gate (%s)" % e, file=sys.stderr)
        return 2

    try:
        gate.save(args.output)
    except OSError as e:
        print("taut: could not write %s (%s)" % (args.output, e), file=sys.stderr)
        return 2

    if args.quiet:
        return 0

    print(gate.report())
    if held_idx:
        print()
        print(measure(gate, [results[i] for i in held_idx], [labels[i] for i in held_idx]))
    else:
        print()
        print("  Fitted on every row (--split 1.0), so nothing is held back to measure it.")
    print()
    print("Wrote %s (%d question%s). Apply it with:" % (args.output, len(gate),
                                                        "" if len(gate) == 1 else "s"))
    print("    taut \"...\" --predict --gate %s" % args.output)
    return 0


def measure(gate, results, labels):
    """Held-out behaviour of a fitted gate, as a table.

    This is the number that matters and the one a fitted-on-everything report cannot give:
    what the gate actually did to data it had never seen.
    """
    lines = ["Held-out check  n=%d" % len(results)]
    lines.append("")
    header = "%-22s %-10s %-24s %s" % ("question", "mode", "observed", "against")
    lines.append(header)
    lines.append("-" * len(header))
    for qid, qgate in gate.gates.items():
        seen = accepted = wrong_accepted = covered = total_set = 0
        for res, lab in zip(results, labels):
            answer = res.get("answers", {}).get(qid)
            if answer is None or not isinstance(lab, Mapping_) or lab.get(qid) is None:
                continue
            block = qgate.apply(answer)
            seen += 1
            if qgate.mode == "miss":
                # A miss gate's guarantee is label-conditional: of the *positives*, how
                # many were not blocked. Scoring it like a selective gate would measure
                # a different quantity and flatter the result on a low-prevalence set.
                positive = _is_positive(qgate, lab[qid])
                if not block.get("blocked"):
                    accepted += 1
                    wrong_accepted += positive
                covered += positive          # reused as the positive count
            elif qgate.mode == "selective":
                ok_ = _is_right(qgate, answer, lab[qid])
                if block.get("accepted"):
                    accepted += 1
                    wrong_accepted += (not ok_)
            elif qgate.mode == "set":
                covered += str(lab[qid]) in block["prediction_set"]
                total_set += block["set_size"]
            else:
                lo, hi = block["interval"]
                covered += lo <= float(lab[qid]) <= hi
        if not seen:
            lines.append("%-22s %-10s %-24s %s" % (qid[:22], qgate.mode, "no labelled rows", ""))
            continue
        if qgate.mode == "miss":
            n_pos = covered
            obs = ("%.2f%% of positives missed" % (100.0 * wrong_accepted / n_pos)
                   if n_pos else "no positives held out")
            against = "budget %.3g  (blocked %.1f%%)" % (qgate.alpha,
                                                         100.0 * (1 - accepted / seen))
        elif qgate.mode == "selective":
            obs = "%.2f%% accepted+wrong" % (100.0 * wrong_accepted / seen)
            against = "budget %.3g  (kept %.1f%%)" % (qgate.alpha, 100.0 * accepted / seen)
        elif qgate.mode == "set":
            obs = "%.1f%% covered" % (100.0 * covered / seen)
            against = "target %.1f%%  (mean size %.2f)" % (100.0 * (1 - qgate.alpha),
                                                           total_set / seen)
        else:
            obs = "%.1f%% covered" % (100.0 * covered / seen)
            against = "target %.1f%%" % (100.0 * (1 - qgate.alpha))
        lines.append("%-22s %-10s %-24s %s" % (qid[:22], qgate.mode, obs, against))
    lines.append("")
    lines.append("  One held-out sample is one draw: landing a little over the budget is "
                 "noise, not a breach.")
    return "\n".join(lines)


def _is_positive(qgate, gold):
    """Whether a noul gold label is the positive (unsafe) class."""
    from taut.conformal import _normalise_label

    return _normalise_label("noul", ["false", "true"], gold) == 1


def _is_right(qgate, answer, gold):
    """Whether the model's top answer matches gold, in the gate's own terms."""
    from taut.conformal import _answer_distribution, _normalise_label

    _, keys, p = _answer_distribution(answer)
    return int(p.argmax()) == _normalise_label(qgate.qtype, keys, gold)


def make_router(args):
    return taut.Router(device=args.device, preload=False)


def show_decision(decision):
    print("Model     :", decision["model"])
    print("Reason    :", decision["reason"])
    detection = decision.get("detection")
    if detection:
        print("Detected  :", json.dumps(detection, ensure_ascii=False))


def show_answers(result):
    routing = result.get("routing")
    if routing:
        show_decision(routing)
        print()
    for qid, answer in result.get("answers", {}).items():
        if "choice" in answer:
            # A choice answer carries `probabilities` (a dict over the criteria), never a
            # scalar `probability` -- reading the latter printed p=0.000 for every answer.
            chosen = answer["choice"]
            p = answer.get("probabilities", {}).get(chosen)
            if p is None:
                p = answer.get("confidence", 0.0)
            detail = "%s (p=%.3f)" % (chosen, p)
        elif "score" in answer:
            detail = "%.2f" % answer["score"]
        elif "noul" in answer:
            detail = "%.3f" % answer["noul"]
        else:
            detail = json.dumps(answer, ensure_ascii=False)
        print("%-12s: %s" % (qid, detail))


def show_gate(result):
    """The gate blocks, next to the answers they certify."""
    record = result.get("gate")
    for qid, answer in result.get("answers", {}).items():
        block = answer.get("gate")
        if not block:
            continue
        if block["mode"] == "set":
            verdict = "set %s" % (block["prediction_set"],)
        elif block["mode"] == "interval":
            verdict = "interval %s" % (block["interval"],)
        else:
            verdict = "ACCEPT" if block.get("accepted") else "ABSTAIN"
        flag = "" if block.get("certifiable", True) else "  [NOT CERTIFIABLE]"
        print("%-12s: %-10s %s%s" % (qid, verdict, block["guarantee"], flag))
    if record and record.get("family_guarantee"):
        print("\nrecord      : %s" % record["family_guarantee"])


def run(text, args, router=None, gate=None):
    """Route or predict one request; returns 0 on success, 2 on a handled error."""
    router = router or make_router(args)
    state = {"text": text}
    try:
        if args.predict:
            result = router.predict(state, taut.router_questions(),
                                    model=args.model, task=args.task, lang=args.lang)
            if gate is not None:
                result = gate.apply(result)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            else:
                show_answers(result)
                if gate is not None:
                    print()
                    show_gate(result)
        else:
            decision = router.route(state, model=args.model, task=args.task, lang=args.lang)
            if args.json:
                print(json.dumps(dict(decision), ensure_ascii=False, indent=2, default=str))
            else:
                show_decision(decision)
    except ValueError as error:
        print("taut: %s" % error, file=sys.stderr)
        return 2
    except (ImportError, OSError, RuntimeError) as error:
        print("taut: could not run Taut (%s)." % error, file=sys.stderr)
        print("Check that the dependencies are installed and the checkpoints can be "
              "downloaded from the Hugging Face hub (network access is needed on first use).",
              file=sys.stderr)
        return 2
    return 0


def interactive(args, gate=None):
    print("Taut interactive mode. Type a request and press Enter; Ctrl-D or 'quit' to exit.")
    router = make_router(args)
    while True:
        try:
            text = input("taut> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        run(text, args, router, gate)
    return 0


SUBCOMMANDS = {"calibrate": calibrate}


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in SUBCOMMANDS:
        return SUBCOMMANDS[raw[0]](raw[1:])

    parser = build_parser()
    args = parser.parse_args(raw)
    text = " ".join(args.text).strip()

    gate = None
    if args.gate:
        try:
            gate = load_gate(args.gate)
        except (OSError, ValueError, KeyError) as error:
            print("taut: could not load the gate at %s (%s)" % (args.gate, error), file=sys.stderr)
            return 2
        if args.report:
            print(gate.report())
            return 0
        if not args.predict:
            # Routing alone produces no probabilities, so there is nothing to certify.
            # Saying so beats silently ignoring the flag the caller asked for.
            print("taut: --gate applies to answers, so it needs --predict "
                  "(or --report to describe the gate)", file=sys.stderr)
            return 2
    elif args.report:
        print("taut: --report needs --gate PATH", file=sys.stderr)
        return 2

    if not text:
        return interactive(args, gate)
    return run(text, args, gate=gate)


if __name__ == "__main__":
    sys.exit(main())
