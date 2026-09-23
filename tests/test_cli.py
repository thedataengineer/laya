"""CLI tests. No model weights are loaded: the Router is stubbed throughout."""
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import cli  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
    else:
        FAIL.append("%s: %s" % (name, detail))


class StubDecision(dict):
    def __init__(self):
        super().__init__(model="multilingual", repo="convaiinnovations/laya",
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
check("error: names the failure", "could not run Laya" in err, err)
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

from laya.conformal import ConformalGate  # noqa: E402


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

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all CLI tests passed")
sys.exit(1 if FAIL else 0)
