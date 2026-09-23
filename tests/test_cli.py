"""CLI tests. No model weights are loaded: the Router is stubbed throughout."""
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taut import cli  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
    else:
        FAIL.append("%s: %s" % (name, detail))


class StubDecision(dict):
    def __init__(self):
        super().__init__(model="multilingual", repo="thekarteek/taut",
                         reason="detected non-English text", detection={"lang": "de"}, workflow=None)


class StubRouter:
    def __init__(self):
        self.route_calls = []
        self.predict_calls = []

    def route(self, state, **kwargs):
        self.route_calls.append((state, kwargs))
        return StubDecision()

    def predict(self, state, questions, **kwargs):
        self.predict_calls.append((state, kwargs))
        return {"answers": {"difficulty": {"score": 1.4}}, "routing": dict(StubDecision())}


def run_cli(argv, router=None):
    stub = router or StubRouter()
    original = cli.make_router
    cli.make_router = lambda args: stub
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
    finally:
        cli.make_router = original
    return code, out.getvalue(), err.getvalue(), stub


# --------------------------------------------------------------------- routing (default)
code, out, err, stub = run_cli(["Ich wurde doppelt belastet"])
check("route: exit code", code == 0, "got %r" % code)
check("route: prints the chosen checkpoint", "multilingual" in out, out)
check("route: state carries the text", stub.route_calls[0][0] == {"text": "Ich wurde doppelt belastet"})
check("route: never loads a checkpoint", stub.predict_calls == [])

# --------------------------------------------------------------------- full prediction
code, out, err, stub = run_cli(["--predict", "Refactor this service"])
check("predict: exit code", code == 0, "got %r" % code)
check("predict: prints answers", "difficulty" in out, out)
check("predict: router.predict called once", len(stub.predict_calls) == 1)

# --------------------------------------------------------------------- json output
code, out, err, stub = run_cli(["--json", "hello there"])
check("json: exit code", code == 0, "got %r" % code)
check("json: raw decision printed", '"model": "multilingual"' in out, out)

# --------------------------------------------------------------------- friendly errors
class BrokenRouter:
    def route(self, state, **kwargs):
        raise OSError("connection failed")


code, out, err, stub = run_cli(["some text"], router=BrokenRouter())
check("error: exit code 2", code == 2, "got %r" % code)
check("error: names the failure", "could not run Taut" in err, err)
check("error: points at the fix", "Hugging Face hub" in err, err)

# --------------------------------------------------------------------- explicit flags
code, out, err, stub = run_cli(["--model", "english", "charged twice"])
check("flags: --model forwarded", stub.route_calls[0][1]["model"] == "english")

# --------------------------------------------------------------------- risk gating
# `--gate` applies a saved ConformalGate to the answers. It is pure NumPy and loads no
# checkpoint, so the gate paths are exercised against the same stub router.
import json as _json  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

from taut.conformal import ConformalGate  # noqa: E402


class GatedStubRouter(StubRouter):
    """Answers the question the fixture gate below is calibrated on."""

    def predict(self, state, questions, **kwargs):
        self.predict_calls.append((state, kwargs))
        return {"answers": {"dept": {"type": "choice", "choice": "billing",
                                     "probabilities": {"billing": 0.94, "tech": 0.06},
                                     "confidence": 0.94}},
                "routing": dict(StubDecision())}


def _gate_path(alpha=0.05):
    rng = np.random.default_rng(3)
    results, labels = [], []
    for _ in range(800):
        gold = "billing" if rng.random() < 0.5 else "tech"
        p = float(rng.beta(6, 2))
        p_billing = p if gold == "billing" else 1.0 - p
        results.append({"answers": {"dept": {
            "type": "choice", "choice": "billing" if p_billing >= 0.5 else "tech",
            "probabilities": {"billing": p_billing, "tech": 1.0 - p_billing},
            "confidence": max(p_billing, 1.0 - p_billing)}}})
        labels.append({"dept": gold})
    gate = ConformalGate.calibrate(results, labels, alpha=alpha, delta=0.05)
    path = os.path.join(tempfile.mkdtemp(), "gate.json")
    gate.save(path)
    return path


GATE = _gate_path()

# A choice answer carries `probabilities`, a dict over the criteria -- never a scalar
# `probability`. Reading the latter printed "p=0.000" for every choice answer ever shown.
code, out, err, stub = run_cli(["--predict", "billed twice"], router=GatedStubRouter())
check("answers: choice prints its real probability", "p=0.940" in out, out)
check("answers: choice prints the chosen criterion", "billing" in out, out)

code, out, err, stub = run_cli(["--gate", GATE, "--report"])
check("gate: --report exits clean", code == 0, "%r %s" % (code, err))
check("gate: --report names the question", "dept" in out, out)
check("gate: --report states the guarantee", "accepted-and-wrong <= 0.05" in out, out)
check("gate: --report loads no checkpoint", stub.predict_calls == [])

code, out, err, stub = run_cli(["--predict", "--gate", GATE, "billed twice"],
                               router=GatedStubRouter())
check("gate: applies to a prediction", code == 0, "%r %s" % (code, err))
check("gate: prints a verdict", "ACCEPT" in out or "ABSTAIN" in out, out)
check("gate: prints the guarantee", "with confidence 0.95" in out, out)
check("gate: prints the record-level union bound", "record" in out, out)

code, out, err, stub = run_cli(["--predict", "--json", "--gate", GATE, "billed twice"],
                               router=GatedStubRouter())
payload = _json.loads(out)
check("gate: json carries the per-answer block", "gate" in payload["answers"]["dept"], out)
check("gate: json carries the record block",
      payload["gate"]["family_alpha"] == 0.05, out)

code, out, err, stub = run_cli(["--gate", GATE, "billed twice"])
check("gate: without --predict is refused", code == 2, "got %r" % code)
check("gate: says what to do instead", "needs --predict" in err, err)

code, out, err, stub = run_cli(["--report", "billed twice"])
check("gate: --report without --gate is refused", code == 2, "got %r" % code)
check("gate: names the missing flag", "needs --gate" in err, err)

code, out, err, stub = run_cli(["--predict", "--gate", "/no/such/gate.json", "hi"])
check("gate: an unreadable gate exits 2", code == 2, "got %r" % code)
check("gate: reports the path it could not load", "/no/such/gate.json" in err, err)

bad = os.path.join(tempfile.mkdtemp(), "bad.json")
open(bad, "w", encoding="utf-8").write("{not json")
code, out, err, stub = run_cli(["--predict", "--gate", bad, "hi"])
check("gate: malformed JSON exits 2 rather than raising", code == 2, "got %r" % code)

# --------------------------------------------------------------------- taut calibrate
# Fitting a gate from labelled JSONL must not need a checkpoint when the rows already
# carry model output, which is the path a user reusing predictions across risk budgets
# takes -- and the only one testable without weights.


def _write_jsonl(rows):
    path = os.path.join(tempfile.mkdtemp(), "data.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(_json.dumps(r) + "\n")
    return path


def _labelled_rows(n=1500, seed=7):
    keys = ["billing", "technical", "account", "other"]
    r = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        g = int(r.integers(0, 4))
        lg = r.normal(0, 1, 4)
        lg[g] += 2.2
        e = np.exp(lg - lg.max())
        p = e / e.sum()
        unsafe = bool(r.random() < 0.25)
        pt = float(r.beta(6, 2) if unsafe else r.beta(2, 6))
        rows.append({"answers": {
            "intent": {"type": "choice", "choice": keys[int(p.argmax())],
                       "probabilities": {k: float(v) for k, v in zip(keys, p)},
                       "confidence": float(p.max())},
            "unsafe": {"type": "noul", "noul": pt, "confidence": max(pt, 1 - pt)}},
            "labels": {"intent": keys[g], "unsafe": unsafe}})
    return rows


def run_calibrate(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(["calibrate"] + argv)
    return code, out.getvalue(), err.getvalue()


DATA = _write_jsonl(_labelled_rows())
GATE_OUT = os.path.join(tempfile.mkdtemp(), "fitted.json")

code, out, err = run_calibrate([DATA, "-o", GATE_OUT, "--alpha", "0.05",
                               "--mode", "intent=selective,unsafe=miss"])
check("calibrate: exit code", code == 0, "%r %s" % (code, err))
check("calibrate: needs no checkpoint for precomputed rows", "could not run the model" not in err)
check("calibrate: writes the gate", os.path.exists(GATE_OUT))
check("calibrate: prints the fitted report", "accepted-and-wrong" in out, out[:300])
check("calibrate: applies the per-question modes", "miss <=" in out, out[:400])
check("calibrate: holds data out and measures it", "Held-out check" in out, out[:600])
check("calibrate: scores a miss gate on its own terms",
      "of positives missed" in out, out)
check("calibrate: warns that one sample is one draw", "noise, not a breach" in out)
check("calibrate: tells you how to apply it", "--gate " + GATE_OUT in out, out[-300:])

loaded = ConformalGate.load(GATE_OUT)
check("calibrate: the gate loads back", sorted(loaded.gates) == ["intent", "unsafe"])
check("calibrate: fitted on the split, not everything",
      loaded.meta["n_calibration"] == 750, loaded.meta.get("n_calibration"))

code, out, err = run_calibrate([DATA, "-o", GATE_OUT + "2", "--split", "1.0"])
check("calibrate: --split 1.0 fits on everything", code == 0, err)
check("calibrate: and says nothing was held back", "nothing is held back" in out, out[-400:])
check("calibrate: auto mode picks per question type", "selective" in out)

# costs reach the fit, and change the operating point
code, out, err = run_calibrate([DATA, "-o", GATE_OUT + "3", "--alpha", "0.05",
                                "--mode", "intent=selective,unsafe=miss",
                                "--cost", "intent:error=40,abstain=1",
                                "--cost", "unsafe:miss=800,block=1"])
check("calibrate: --cost is accepted", code == 0, err)
check("calibrate: --cost is priced into the report", "/decision, saves" in out, out[:500])
priced = ConformalGate.load(GATE_OUT + "3")
check("calibrate: --cost tightens the threshold",
      priced["intent"].params["threshold"] > loaded["intent"].params["threshold"],
      "%r vs %r" % (priced["intent"].params["threshold"], loaded["intent"].params["threshold"]))
check("calibrate: --cost survives into the saved gate",
      "cost" in priced["intent"].diagnostics)

code, out, err = run_calibrate([DATA, "-o", GATE_OUT + "4", "--cost", "error=40,abstain=1"])
check("calibrate: a blanket --cost works too", code == 0, err)

code, out, err = run_calibrate([DATA, "--quiet", "-o", GATE_OUT + "5"])
check("calibrate: --quiet writes but prints nothing", code == 0 and out.strip() == "", out)
check("calibrate: --quiet still wrote the gate", os.path.exists(GATE_OUT + "5"))

# failure paths, each with a message naming the fix
bad = _write_jsonl([{"answers": {}}])
code, out, err = run_calibrate([bad, "-o", GATE_OUT + "6"])
check("calibrate: a row without labels is refused", code == 2)
check("calibrate: and says which line", "line 1" in err and "labels" in err, err)

nolabels = os.path.join(tempfile.mkdtemp(), "broken.jsonl")
open(nolabels, "w", encoding="utf-8").write("{not json\n")
code, out, err = run_calibrate([nolabels, "-o", GATE_OUT + "7"])
check("calibrate: malformed JSON is refused", code == 2)
check("calibrate: and points at the line", "line 1" in err, err)

empty = os.path.join(tempfile.mkdtemp(), "empty.jsonl")
open(empty, "w", encoding="utf-8").write("")
code, out, err = run_calibrate([empty, "-o", GATE_OUT + "8"])
check("calibrate: an empty file is refused", code == 2 and "no labelled rows" in err, err)

code, out, err = run_calibrate(["/no/such/data.jsonl", "-o", GATE_OUT + "9"])
check("calibrate: a missing file exits 2", code == 2)

states_only = _write_jsonl([{"text": "hello", "labels": {"intent": "billing"}}])
code, out, err = run_calibrate([states_only, "-o", GATE_OUT + "10"])
check("calibrate: rows without answers need --questions",
      code == 2 and "--questions is required" in err, err)

mixed = _write_jsonl(_labelled_rows(5) + [{"text": "x", "labels": {"intent": "billing"}}])
code, out, err = run_calibrate([mixed, "-o", GATE_OUT + "11"])
check("calibrate: a half-predicted file is refused",
      code == 2 and "for all rows or none" in err, err)

code, out, err = run_calibrate([DATA, "-o", GATE_OUT + "12", "--split", "0"])
check("calibrate: --split 0 is refused", code == 2 and "--split must lie" in err, err)

code, out, err = run_calibrate([DATA, "-o", GATE_OUT + "13", "--cost", "error"])
check("calibrate: a malformed --cost is refused", code == 2 and "key=value" in err, err)
code, out, err = run_calibrate([DATA, "-o", GATE_OUT + "14", "--cost", "error=lots"])
check("calibrate: a non-numeric --cost is refused", code == 2 and "not a number" in err, err)
code, out, err = run_calibrate([DATA, "-o", GATE_OUT + "15",
                                "--cost", "error=1,abstain=1", "--cost", "intent:error=1"])
check("calibrate: mixing blanket and per-question costs is refused",
      code == 2 and "not both" in err, err)

code, out, err = run_calibrate([DATA, "-o", "/no/such/dir/gate.json"])
check("calibrate: an unwritable output exits 2", code == 2 and "could not write" in err, err)

# the flat CLI still works with a subcommand registered
code, out, err, stub = run_cli(["charged twice"])
check("calibrate: the bare CLI is unaffected", code == 0 and "multilingual" in out, out)

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all CLI tests passed")
sys.exit(1 if FAIL else 0)
