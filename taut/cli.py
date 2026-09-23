"""Command-line interface for testing Taut locally.

    taut "I was charged twice, please refund"           # routing decision only (no model download)
    taut "Refactor this service" --predict              # full answers, loads the checkpoint
    taut                                                # interactive mode
    taut "Mein Konto wurde zweimal belastet" --lang de  # explicit language
    taut "..." --predict --gate triage_gate.json         # apply a certified risk gate
    taut --gate triage_gate.json --report                # print what a saved gate certifies

Routing (the default) never downloads a checkpoint, so it works offline and
returns in milliseconds. --predict loads the routed checkpoint on first use,
which needs network access to the Hugging Face hub.
"""

import argparse
import json
import sys

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


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
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
