"""HTTP server exposing Taut over TypeSafe Jev's ``/v1/systemone`` wire protocol.

Taut's ``predict()`` output is already schema-compatible with the Jev decision
API -- ``choice`` / ``score`` / ``noul`` answers and a ``{input_tokens,
output_tokens}`` usage block -- so a client written against Jev (for example the
`hs-jev` Haskell client) can point its ``baseUrl`` at this server and keep
working unchanged. All this module adds is the HTTP surface Taut itself does not
ship: a ``POST /v1/systemone`` route, an optional bearer check, and a health
probe.

Configuration is entirely via environment variables so the same entry point
serves a laptop dev run and a systemd unit:

======================  ============================================  =========
env var                 meaning                                        default
======================  ============================================  =========
``TAUT_HOST``           bind address                                   0.0.0.0
``TAUT_PORT``           bind port                                      8000
``TAUT_DEVICE``         torch device for every checkpoint              (auto)
``TAUT_PRELOAD``        build the checkpoints at startup, not lazily   1
``TAUT_MODELS``         comma list to preload (english,multilingual,   (all)
                        typed-decisions); empty = every checkpoint
``TAUT_THREADS``        cap torch intra-op threads (CPU inference).    (torch
                        Keep <= physical cores; oversubscribing the     default)
                        logical/hyperthread count is a large regression.
``TAUT_AUTO_TASK``      auto-route to the typed-decisions checkpoint   0
``TAUT_API_KEY``        if set, require ``Authorization: Bearer <it>``  (none)
``TAUT_GATE``           path to a ConformalGate JSON; every answer     (none)
                        carries a certified ``gate`` block and the
                        response a record-level one
``TAUT_GATE_STRICT``    with ``TAUT_GATE``, reject a request whose     0
                        questions the gate was not calibrated on
``TAUT_LOG_LEVEL``      uvicorn log level                              info
======================  ============================================  =========

With ``TAUT_GATE`` set the server stops being a scorer and becomes a decision
service: the risk budget is the operator's, fixed at deploy time and visible in
``GET /health``, rather than a threshold each caller invents. Deliberately not a
request field -- a client that can name its own ``alpha`` can claim any guarantee
it likes.

Imports of heavy dependencies (fastapi, uvicorn, torch via Router) are all
deferred into the functions that need them, so ``import taut.serve`` stays cheap
and touches no GPU -- which is what keeps the Nix ``pythonImportsCheck`` honest.
"""
import os
from typing import Any, Dict, Optional

# The three checkpoint names the router understands; used to decide whether a
# client's `model` field names a Taut checkpoint (honour it) or is some other
# Jev model id (ignore it and let the router auto-select).
_KNOWN_MODELS = {"english", "multilingual", "typed-decisions"}

# Public Hugging Face ids, accepted so a client can name a checkpoint. The root bundle is
# deliberately absent: the documented ``thekarteek/taut`` value means
# "let the Router choose", rather than pinning the English checkpoint.
_PUBLISHED_MODEL_IDS = {
    "thekarteek/taut-multilingual": "multilingual",
    "thekarteek/taut-typed-decisions": "typed-decisions",
}


def load_gate() -> Optional[Any]:
    """The ``ConformalGate`` named by ``TAUT_GATE``, or None when unset.

    Loaded once at app creation and never reloaded: a gate that changed under a running
    server would silently move the guarantee the ``/health`` probe is advertising.
    Failure to load is fatal rather than a warning -- a server configured to certify its
    answers and then quietly not doing so is the worst of the available outcomes.
    """
    path = (os.environ.get("TAUT_GATE") or "").strip()
    if not path:
        return None
    from .conformal import ConformalGate

    try:
        return ConformalGate.load(path)
    except Exception as exc:  # noqa: BLE001 -- refuse to start rather than serve ungated
        raise RuntimeError("TAUT_GATE=%s could not be loaded as a conformal gate: %s"
                           % (path, exc)) from exc


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a client's `model` field onto a Taut checkpoint, or None to auto-route."""
    if not model:
        return None
    published = _PUBLISHED_MODEL_IDS.get(str(model).strip().lower())
    if published is not None:
        return published
    from .router import normalise_name

    # normalise_name raises ValueError on anything that is not a known checkpoint
    # or alias. A Jev client's `model` field (e.g. "jev-1") is expected to miss;
    # treat that as "no explicit checkpoint" and let the router auto-select.
    try:
        key = normalise_name(model)
    except Exception:
        return None
    return key if key in _KNOWN_MODELS else None


def _apply_thread_limit():
    """Honour TAUT_THREADS by capping torch's intra-op thread count for CPU
    inference. Returns the value applied, or None if unset/invalid. torch is
    imported only when a limit is actually requested."""
    raw = os.environ.get("TAUT_THREADS")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n <= 0:
        return None
    import torch

    torch.set_num_threads(n)
    return n


def build_router():
    """Build a Router from the environment, preloading unless told otherwise."""
    from .router import Router

    _apply_thread_limit()
    device = os.environ.get("TAUT_DEVICE") or None
    models_env = os.environ.get("TAUT_MODELS", "").strip()
    preload_names = [m.strip() for m in models_env.split(",") if m.strip()] or None
    router = Router(device=device, auto_task_detection=_env_bool("TAUT_AUTO_TASK", False))
    if _env_bool("TAUT_PRELOAD", True):
        router.preload(preload_names)
    return router


def create_app(router: Optional[Any] = None, risk_gate: Optional[Any] = None):
    """Build the FastAPI app. Pass a Router to inject one (tests); otherwise one
    is built from the environment (and preloaded) at app-creation time.

    ``risk_gate`` likewise overrides ``TAUT_GATE``; pass a fitted
    :class:`~taut.conformal.ConformalGate` to certify every answer this server returns.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, Header, HTTPException, Request

    if router is None:
        router = build_router()
    if risk_gate is None:
        risk_gate = load_gate()
    gate_strict = _env_bool("TAUT_GATE_STRICT", False)
    api_key = os.environ.get("TAUT_API_KEY") or None

    # Inference is synchronous torch, and a CPU call takes hundreds of milliseconds to
    # seconds, so it must not run on the event loop: one request would stall every
    # other client, `GET /health` included. One worker, because one forward pass at a
    # time is what a single CPU or GPU Agent wants (the Router already guards checkpoint
    # lifecycle, and leaves `Agent.system_one` unguarded deliberately so concurrent
    # predictions can share a checkpoint -- a GPU-shaped choice this endpoint does not
    # rely on). `loop.run_in_executor` is the API the issue asked for.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taut-infer")
    # Created on first request, not here: an `asyncio.Lock` binds to the loop that is
    # running when it is first awaited, and `create_app` may be called before that loop
    # exists (module scope, TestClient startup, a preload script).
    gate: Optional[asyncio.Lock] = None

    app = FastAPI(
        title="taut-serve",
        summary="Taut System-1 decisions over the TypeSafe Jev /v1/systemone protocol",
    )

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        if authorization != "Bearer " + api_key:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        payload = {
            "status": "ok",
            "loaded": router.loaded,
            "device": os.environ.get("TAUT_DEVICE") or "auto",
            "gate": None,
        }
        if risk_gate is not None:
            # The guarantee is part of the service contract, so a caller can read it
            # without having to trust a README.
            payload["gate"] = dict(
                {
                    "alpha": risk_gate.alpha,
                    "delta": risk_gate.delta,
                    "questions": sorted(risk_gate.gates),
                    "modes": {q: g.mode for q, g in sorted(risk_gate.gates.items())},
                    "strict": gate_strict,
                },
                **risk_gate.family_risk()
            )
        return payload

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        nonlocal gate
        _check_auth(authorization)
        body = await request.json()
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        state = body.get("state")
        questions = body["questions"]
        model = _resolve_model(body.get("model"))
        if gate is None:
            gate = asyncio.Lock()
        try:
            # Taut's result is already Jev-shaped: {model, answers, usage, routing}.
            # hs-jev decodes `answers` and `usage` and ignores the rest.
            async with gate:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    pool, lambda: router.predict(state, questions, model=model))
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 -- surface model/tokenizer errors as 422
            raise HTTPException(status_code=422, detail=str(e))

        if risk_gate is None:
            return result
        try:
            # Gating is pure NumPy over probabilities already computed, so it stays on
            # the event loop rather than costing a second executor hop.
            return risk_gate.apply(result, strict=gate_strict)
        except Exception as e:  # noqa: BLE001 -- a gate/question mismatch is the caller's
            raise HTTPException(status_code=422, detail="gate could not be applied: %s" % e)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("TAUT_HOST", "0.0.0.0"),
        port=int(os.environ.get("TAUT_PORT", "8000")),
        log_level=os.environ.get("TAUT_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
