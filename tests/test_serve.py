"""Server-shim tests: verify the Jev /v1/systemone surface without a GPU.

A fake Router is injected so nothing loads a checkpoint; we only assert that the
HTTP layer maps requests/responses and enforces auth as hs-jev expects.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from taut.serve import _apply_thread_limit, _env_bool, _resolve_model, create_app  # noqa: E402


class FakeRouter:
    """Records the last predict() call and returns a Jev-shaped payload."""

    loaded = ["english"]

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        return {
            "model": "taut-rl-agent",
            "answers": {
                "dept": {"type": "choice", "choice": "billing",
                         "probabilities": {"billing": 0.94, "tech": 0.06}, "confidence": 0.94},
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
            "routing": {"model": "english", "reason": "English Latin text"},
        }


def _client(monkeypatch, api_key=None):
    if api_key is None:
        monkeypatch.delenv("TAUT_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TAUT_API_KEY", api_key)
    fake = FakeRouter()
    return TestClient(create_app(router=fake)), fake


REQ = {
    "model": "jev-1",  # a non-Taut model id -> should be ignored, router auto-routes
    "state": {"body": "billed twice, refund please"},
    "questions": {"dept": {"type": "choice", "instructions": "which team?",
                           "criteria": {"billing": None, "tech": None}}},
}


def test_predict_passthrough_shape(monkeypatch):
    client, fake = _client(monkeypatch)
    r = client.post("/v1/systemone", json=REQ)
    assert r.status_code == 200
    body = r.json()
    # exactly the fields hs-jev's Response/Usage decoders require
    assert set(["answers", "usage"]).issubset(body)
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert body["answers"]["dept"]["choice"] == "billing"
    # unknown model id was dropped -> router asked to auto-route
    assert fake.calls[0]["model"] is None


def test_known_model_is_honoured(monkeypatch):
    client, fake = _client(monkeypatch)
    client.post("/v1/systemone", json={**REQ, "model": "multilingual"})
    assert fake.calls[0]["model"] == "multilingual"


@pytest.mark.parametrize(("model", "expected"), [
    ("thekarteek/taut-multilingual", "multilingual"),
    ("thekarteek/taut-typed-decisions", "typed-decisions"),
])
def test_published_model_id_is_honoured(monkeypatch, model, expected):
    client, fake = _client(monkeypatch)
    client.post("/v1/systemone", json={**REQ, "model": model})
    assert fake.calls[0]["model"] == expected


def test_missing_questions_is_400(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.post("/v1/systemone", json={"state": "hi"})
    assert r.status_code == 400


def test_auth_required_when_key_set(monkeypatch):
    client, _ = _client(monkeypatch, api_key="s3cret")
    assert client.post("/v1/systemone", json=REQ).status_code == 401
    ok = client.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_health(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_helpers():
    assert _resolve_model("multilingual") == "multilingual"
    assert _resolve_model("thekarteek/taut-multilingual") == "multilingual"
    assert _resolve_model("thekarteek/taut-typed-decisions") == "typed-decisions"
    assert _resolve_model("thekarteek/taut") is None
    assert _resolve_model("jev-1") is None
    assert _resolve_model(None) is None
    import os
    os.environ.pop("X_FLAG", None)
    assert _env_bool("X_FLAG", True) is True


def test_thread_limit(monkeypatch):
    monkeypatch.delenv("TAUT_THREADS", raising=False)
    assert _apply_thread_limit() is None  # unset -> no-op, no torch import
    for bad in ("0", "-4", "abc", ""):
        monkeypatch.setenv("TAUT_THREADS", bad)
        assert _apply_thread_limit() is None
    monkeypatch.setenv("TAUT_THREADS", "8")
    assert _apply_thread_limit() == 8
    import torch
    assert torch.get_num_threads() == 8


# The endpoint is `async def` and inference is synchronous torch, which on CPU takes
# hundreds of milliseconds to seconds. Calling it from the coroutine puts that work on
# the event loop, so every other client -- `GET /health` included -- waits for it.
# Driving the app directly on a loop (`httpx.ASGITransport`) makes the difference
# observable: offloaded work runs on a worker thread, inline work runs on the loop's own
# `MainThread`. `TestClient` cannot see this, because it runs the loop in a portal thread
# and hands each call its own, so a blocking endpoint still looks concurrent there.
class SlowRouter(FakeRouter):
    """Sleeps like a CPU forward pass and records the thread it ran on."""

    def __init__(self, seconds=0.25):
        super().__init__()
        self.seconds = seconds
        self.threads = []

    def predict(self, state, questions, model=None):
        import threading
        import time
        self.threads.append(threading.current_thread().name)
        time.sleep(self.seconds)
        return super().predict(state, questions, model=model)


def test_inference_runs_off_the_event_loop(monkeypatch):
    import asyncio
    import threading

    import httpx

    monkeypatch.delenv("TAUT_API_KEY", raising=False)
    fake = FakeRouter()
    seen = []
    real_predict = fake.predict

    def recording_predict(state, questions, model=None):
        seen.append(threading.current_thread().name)
        return real_predict(state, questions, model=model)

    fake.predict = recording_predict
    app = create_app(router=fake)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post("/v1/systemone", json=REQ)

    response = asyncio.run(drive())

    assert response.status_code == 200, response.text
    assert seen, "predict was never called"
    assert "MainThread" not in seen, (
        "predict ran on the event loop thread: %s -- one request would stall every "
        "other client, including GET /health" % seen)


def test_health_stays_available_during_inference(monkeypatch):
    """A request in flight must not stop the app answering `GET /health`."""
    import asyncio

    import httpx

    monkeypatch.delenv("TAUT_API_KEY", raising=False)
    fake = SlowRouter(seconds=0.25)
    app = create_app(router=fake)
    seen = {}

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await client.post("/v1/systemone", json=REQ)          # warm up

            async def slow_request():
                seen["slow"] = (await client.post("/v1/systemone", json=REQ)).status_code

            async def health():
                r = await client.get("/health")
                seen["health"] = r.status_code
                seen["payload"] = r.json()

            await asyncio.gather(slow_request(), health())

    asyncio.run(drive())

    assert seen["slow"] == 200
    assert seen["health"] == 200 and seen["payload"]["status"] == "ok"
    assert fake.threads and "MainThread" not in fake.threads, fake.threads


# ---------------------------------------------------------------- certified risk gating
# With TAUT_GATE set the server stops being a scorer and becomes a decision service.
# The gate is the operator's, loaded once at startup, and never a request field.

def _fitted_gate(alpha=0.05, delta=0.05, mode="selective"):
    """A real ConformalGate over the question FakeRouter answers."""
    import numpy as np

    from taut.conformal import ConformalGate

    rng = np.random.default_rng(3)
    results, labels = [], []
    for _ in range(800):
        gold = "billing" if rng.random() < 0.5 else "tech"
        p = float(rng.beta(6, 2))
        p_billing = p if gold == "billing" else 1.0 - p
        results.append({"answers": {"dept": {
            "type": "choice",
            "choice": "billing" if p_billing >= 0.5 else "tech",
            "probabilities": {"billing": p_billing, "tech": 1.0 - p_billing},
            "confidence": max(p_billing, 1.0 - p_billing),
        }}})
        labels.append({"dept": gold})
    return ConformalGate.calibrate(results, labels, alpha=alpha, delta=delta,
                                   mode={"dept": mode})


def _gate_file(tmp_path, gate):
    path = tmp_path / "gate.json"
    gate.save(str(path))
    return str(path)


def test_ungated_server_reports_no_gate(monkeypatch):
    monkeypatch.delenv("TAUT_GATE", raising=False)
    client, _ = _client(monkeypatch)
    assert client.get("/health").json()["gate"] is None
    assert "gate" not in client.post("/v1/systemone", json=REQ).json()


def test_gate_from_env_certifies_every_answer(monkeypatch, tmp_path):
    monkeypatch.delenv("TAUT_API_KEY", raising=False)
    monkeypatch.setenv("TAUT_GATE", _gate_file(tmp_path, _fitted_gate()))
    client = TestClient(create_app(router=FakeRouter()))

    body = client.post("/v1/systemone", json=REQ).json()
    blk = body["answers"]["dept"]["gate"]
    assert blk["mode"] == "selective"
    assert blk["alpha"] == 0.05 and blk["delta"] == 0.05
    assert blk["accepted"] is (blk["top_probability"] >= blk["threshold"])
    # the original answer survives alongside the guarantee
    assert body["answers"]["dept"]["choice"] == "billing"
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    # and the record-level block carries the union bound, not the per-question alpha
    assert body["gate"]["family_alpha"] == 0.05
    assert body["gate"]["questions_gated"] == 1


def test_health_advertises_the_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("TAUT_GATE", _gate_file(tmp_path, _fitted_gate(alpha=0.02)))
    client = TestClient(create_app(router=FakeRouter()))
    g = client.get("/health").json()["gate"]
    assert g["alpha"] == 0.02
    assert g["questions"] == ["dept"] and g["modes"] == {"dept": "selective"}
    assert "union bound" in g["family_guarantee"]
    assert g["strict"] is False


def test_injected_gate_overrides_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TAUT_GATE", _gate_file(tmp_path, _fitted_gate(alpha=0.02)))
    client = TestClient(create_app(router=FakeRouter(), risk_gate=_fitted_gate(alpha=0.10)))
    assert client.get("/health").json()["gate"]["alpha"] == 0.10


def test_strict_mode_rejects_uncalibrated_questions(monkeypatch, tmp_path):
    monkeypatch.setenv("TAUT_GATE", _gate_file(tmp_path, _fitted_gate()))
    monkeypatch.setenv("TAUT_GATE_STRICT", "1")
    client = TestClient(create_app(router=FakeRouter()))
    # FakeRouter always answers "dept"; a gate fitted on something else must 422 rather
    # than pass the answer through wearing no guarantee.
    other = TestClient(create_app(router=FakeRouter(),
                                  risk_gate=_fitted_gate()))  # sanity: same questions pass
    assert other.post("/v1/systemone", json=REQ).status_code == 200

    import numpy as np

    from taut.conformal import ConformalGate
    rng = np.random.default_rng(5)
    res = [{"answers": {"other_q": {"type": "noul", "noul": float(rng.beta(5, 2)),
                                    "confidence": 0.8}}} for _ in range(400)]
    lab = [{"other_q": bool(rng.random() < 0.5)} for _ in range(400)]
    mismatched = ConformalGate.calibrate(res, lab, alpha=0.05)
    strict = TestClient(create_app(router=FakeRouter(), risk_gate=mismatched))
    r = strict.post("/v1/systemone", json=REQ)
    assert r.status_code == 422
    assert "gate could not be applied" in r.json()["detail"]
    assert "no gate fitted" in r.json()["detail"]


def test_a_gate_that_will_not_load_stops_the_server(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("TAUT_GATE", str(bad))
    # Serving ungated while advertising a guarantee is the worst outcome available,
    # so a broken gate is fatal rather than a warning.
    with pytest.raises(RuntimeError, match="could not be loaded as a conformal gate"):
        create_app(router=FakeRouter())


def test_gate_is_not_a_request_field(monkeypatch, tmp_path):
    """A caller must not be able to name its own risk budget."""
    monkeypatch.setenv("TAUT_GATE", _gate_file(tmp_path, _fitted_gate(alpha=0.02)))
    client = TestClient(create_app(router=FakeRouter()))
    body = client.post("/v1/systemone", json=dict(REQ, alpha=0.5, gate={"alpha": 0.5})).json()
    assert body["answers"]["dept"]["gate"]["alpha"] == 0.02
