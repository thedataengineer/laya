"""LangChain and LangGraph integration for Taut System 1 decision engine.

Provides fast (~33 ms), non-autoregressive routing, real-time guardrails, and
state evaluation nodes for LangChain Expression Language (LCEL) and LangGraph.

Supports both local in-process models (`Agent` / `Router`) and remote HTTP
deployments (your own `taut-serve`) without requiring PyTorch on edge clients.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Sequence, Union

# Optional LangChain base class integration
try:
    from langchain_core.runnables import RunnableConfig, RunnableSerializable
    _RUNNABLE_AVAILABLE = True
except ImportError:
    _RUNNABLE_AVAILABLE = False
    RunnableSerializable = object  # type: ignore
    RunnableConfig = Any  # type: ignore


class TautGuardrailError(ValueError):
    """Raised when an input violates a Taut guardrail policy."""

    def __init__(self, message: str, violations: Dict[str, Any], raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.violations = violations
        self.raw_decision = raw_decision


def _extract_text(input_val: Any, state_key: Optional[Union[str, Callable[[Any], Any]]] = None) -> Union[str, dict, list]:
    """Extract evaluatable text from arbitrary LangChain/LangGraph states or messages."""
    if state_key is not None:
        if callable(state_key):
            return state_key(input_val)
        if isinstance(input_val, dict) and state_key in input_val:
            return _extract_from_message_or_value(input_val[state_key])

    if isinstance(input_val, str):
        return input_val

    if isinstance(input_val, dict):
        for candidate in ("input", "text", "query", "prompt", "message", "body", "content"):
            if candidate in input_val:
                return _extract_from_message_or_value(input_val[candidate])
        if "messages" in input_val and isinstance(input_val["messages"], list):
            return _extract_from_messages_list(input_val["messages"])
        return input_val

    if isinstance(input_val, list):
        return _extract_from_messages_list(input_val)

    return str(input_val)


def _extract_from_message_or_value(val: Any) -> Any:
    if hasattr(val, "content"):
        return str(val.content)
    if isinstance(val, list):
        return _extract_from_messages_list(val)
    return val


def _extract_from_messages_list(msgs: Sequence[Any]) -> str:
    if not msgs:
        return ""
    # Search backwards for the most recent human/user message
    for m in reversed(msgs):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return str(getattr(m, "content", m))
    last = msgs[-1]
    return str(getattr(last, "content", last))


def _call_remote(
    base_url: str,
    state: Any,
    questions: Dict[str, Any],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Send decision request to a remote taut-serve HTTP instance using standard library urllib."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1/systemone"):
        url = f"{url}/v1/systemone"

    payload: Dict[str, Any] = {"state": state, "questions": questions}
    if model:
        payload["model"] = model

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Taut server error {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Taut server at {url}: {e}") from e


_DEFAULT_ROUTER = None


def _get_default_router():
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        from ..router import Router
        _DEFAULT_ROUTER = Router()
    return _DEFAULT_ROUTER


def _execute_decision(
    state: Any,
    questions: Dict[str, Any],
    agent: Optional[Any] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if base_url:
        return _call_remote(base_url, state, questions, api_key=api_key, model=model)
    runner = agent if agent is not None else _get_default_router()
    kwargs = {"model": model} if model else {}
    return runner.predict(state, questions, **kwargs)


class TautRouter(RunnableSerializable):
    """Zero-latency LangGraph conditional edge and LangChain LCEL routing runnable.

    Evaluates user input against typed criteria in ~33 ms without token generation.
    Supports confidence threshold gating and fallback routing.
    """

    criteria: Dict[str, str]
    instructions: str = "Which route should handle this request?"
    confidence_threshold: float = 0.0
    fallback: Optional[str] = None
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    question_id: str = "route"
    last_decision: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        criteria: Dict[str, str],
        instructions: str = "Which route should handle this request?",
        confidence_threshold: float = 0.0,
        fallback: Optional[str] = None,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                criteria=criteria,
                instructions=instructions,
                confidence_threshold=confidence_threshold,
                fallback=fallback,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.criteria = criteria
            self.instructions = instructions
            self.confidence_threshold = confidence_threshold
            self.fallback = fallback
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
        self.question_id = "route"
        self.last_decision: Optional[Dict[str, Any]] = None

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        """Route input to a destination branch label."""
        text = _extract_text(input, self.state_key)
        questions = {
            self.question_id: {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": self.criteria,
            }
        }
        res = _execute_decision(
            text,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        self.last_decision = res
        ans = res["answers"][self.question_id]
        choice = ans["choice"]
        confidence = ans.get("confidence", 1.0)

        if self.confidence_threshold > 0.0 and confidence < self.confidence_threshold:
            if self.fallback is not None:
                return self.fallback

        return choice

    def __call__(self, state: Any) -> str:
        """Callable protocol for direct use as a LangGraph conditional edge."""
        return self.invoke(state)


class TautGuardrail(RunnableSerializable):
    """Sub-40ms inline guardrail for LangChain chains and LangGraph nodes.

    Screens for prompt injections, jailbreaks, sensitive data, or custom harm
    criteria before passing inputs downstream.
    """

    questions: Optional[Dict[str, Any]] = None
    action: str = "raise"  # "raise", "filter", or "annotate"
    rejection_message: str = "I cannot fulfill this request because it violates safety guidelines."
    threshold: float = 0.5
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        questions: Optional[Dict[str, Any]] = None,
        action: str = "raise",
        rejection_message: str = "I cannot fulfill this request because it violates safety guidelines.",
        threshold: float = 0.5,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                questions=questions,
                action=action,
                rejection_message=rejection_message,
                threshold=threshold,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.questions = questions
            self.action = action
            self.rejection_message = rejection_message
            self.threshold = threshold
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model

    def _default_questions(self) -> Dict[str, Any]:
        from ..presets import guard_questions
        return guard_questions()

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Screen input against guardrail questions."""
        text = _extract_text(input, self.state_key)
        qdefs = self.questions if self.questions is not None else self._default_questions()

        res = _execute_decision(
            text,
            qdefs,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        answers = res.get("answers", {})

        violations: Dict[str, Any] = {}
        for qid, ans in answers.items():
            t = ans.get("type")
            if t == "noul" and ans.get("noul", 0.0) >= self.threshold:
                violations[qid] = {
                    "probability": ans["noul"],
                    "confidence": ans.get("confidence", 0.0),
                }
            elif t == "score" and ans.get("score", 0.0) >= self.threshold:
                violations[qid] = {
                    "score": ans["score"],
                    "confidence": ans.get("confidence", 0.0),
                }

        is_safe = len(violations) == 0

        if not is_safe and self.action == "raise":
            raise TautGuardrailError(
                f"Taut guardrail policy violation detected: {list(violations.keys())}",
                violations=violations,
                raw_decision=res,
            )

        if not is_safe and self.action == "filter":
            if isinstance(input, dict):
                filtered = dict(input)
                filtered["output"] = self.rejection_message
                return filtered
            return self.rejection_message

        if self.action == "annotate":
            if isinstance(input, dict):
                annotated = dict(input)
                annotated["guardrails"] = {
                    "passed": is_safe,
                    "violations": violations,
                    "answers": answers,
                }
                return annotated
            return {
                "input": input,
                "guardrails": {
                    "passed": is_safe,
                    "violations": violations,
                    "answers": answers,
                },
            }

        return input

    def __call__(self, state: Any) -> Any:
        return self.invoke(state)


class TautTriage(RunnableSerializable):
    """Customer support ticket and incoming message triage node for LangGraph.

    Analyzes intent, urgency, customer frustration, and churn risk in one single
    forward pass and enriches the graph state dictionary.
    """

    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model

    def invoke(self, state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Triage the state and return enriched fields."""
        from ..presets import triage_questions

        text = _extract_text(state, self.state_key)
        res = _execute_decision(
            text,
            triage_questions(),
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        ans = res.get("answers", {})

        triage_info = {
            "intent": ans.get("intent", {}).get("choice"),
            "intent_confidence": ans.get("intent", {}).get("confidence"),
            "is_urgent": ans.get("is_urgent", {}).get("noul", 0.0) >= 0.5,
            "frustration_score": ans.get("frustration", {}).get("score"),
            "churn_risk": ans.get("churn_risk", {}).get("noul", 0.0) >= 0.5,
            "refund_requested": ans.get("refund_requested", {}).get("noul", 0.0) >= 0.5,
        }

        if isinstance(state, dict):
            updated = dict(state)
            updated["triage"] = triage_info
            return updated

        return {"input": state, "triage": triage_info}

    def __call__(self, state: Any) -> Dict[str, Any]:
        return self.invoke(state)


class TautEvaluator(RunnableSerializable):
    """Rubric-based output grading and hallucination evaluation for LangChain.

    Evaluates LLM responses against criteria without generating text.
    """

    questions: Dict[str, Any]
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        questions: Dict[str, Any],
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                questions=questions,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.questions = questions
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model

    def evaluate_strings(self, *, prediction: str, input: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """LangChain standard string evaluation interface."""
        state = {"input": input, "prediction": prediction} if input else prediction
        res = _execute_decision(
            state,
            self.questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        return res.get("answers", {})

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        text = _extract_text(input, self.state_key)
        res = _execute_decision(
            text,
            self.questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        return res.get("answers", {})

    def __call__(self, state: Any) -> Dict[str, Any]:
        return self.invoke(state)


class TautGateEscalation(ValueError):
    """Raised when a gated decision does not clear its certified threshold.

    Carries the gate block that refused, so a handler can log *why* the guarantee
    declined to cover this decision rather than only that it did.
    """

    def __init__(self, message: str, abstained: Dict[str, Any], raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.abstained = abstained
        self.raw_decision = raw_decision


class TautGate(RunnableSerializable):
    """Put a certified risk bound on a decision inside a LangChain or LangGraph app.

    The other runnables here return a score and leave the threshold to you.
    ``TautGate`` carries a :class:`~taut.conformal.ConformalGate` fitted to a risk budget,
    so a node can act on "this decision is inside the 2% error budget I signed off" rather
    than on a number somebody picked. The gate is JSON and holds no weights, so it can be
    committed next to the graph and reviewed as a diff.

    Three actions, matching how graphs actually branch:

    ``annotate`` (default)
        Attach the per-question gate blocks and a summary to the state and pass it on.
    ``route``
        Return ``"accept"`` or ``"escalate"``, for use directly as a conditional edge.
    ``raise``
        Raise :class:`TautGateEscalation` when any gated answer abstains.

    ``accept_when`` decides what a record-level accept means: ``"all"`` (default, every
    gated question must clear) or ``"any"``. Read ``family_alpha`` in the output rather
    than ``alpha``: gating five questions at 2% each risks a record wrong somewhere at up
    to 10%, and the block says so.

    Set ``monitor=True`` and the runnable keeps a :class:`~taut.drift.GateMonitor` at
    ``.monitor``, so a long-running graph can be asked whether its gate has expired.
    """

    gate: Optional[Any] = None
    questions: Optional[Dict[str, Any]] = None
    action: str = "annotate"
    accept_when: str = "all"
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    output_key: str = "taut_gate"
    monitor: bool = False
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        gate: Optional[Any] = None,
        questions: Optional[Dict[str, Any]] = None,
        action: str = "annotate",
        accept_when: str = "all",
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        output_key: str = "taut_gate",
        monitor: bool = False,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if action not in ("annotate", "route", "raise"):
            raise ValueError("action must be 'annotate', 'route' or 'raise'; got %r" % (action,))
        if accept_when not in ("all", "any"):
            raise ValueError("accept_when must be 'all' or 'any'; got %r" % (accept_when,))
        gate = self._resolve_gate(gate)
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                gate=gate,
                questions=questions,
                action=action,
                accept_when=accept_when,
                state_key=state_key,
                output_key=output_key,
                monitor=monitor,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.gate = gate
            self.questions = questions
            self.action = action
            self.accept_when = accept_when
            self.state_key = state_key
            self.output_key = output_key
            self.monitor = monitor
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
        self._monitor = None
        if monitor:
            from ..drift import GateMonitor
            self._monitor = GateMonitor(gate)

    @staticmethod
    def _resolve_gate(gate: Any) -> Any:
        from ..conformal import ConformalGate

        if gate is None:
            raise ValueError(
                "TautGate needs a fitted gate: a ConformalGate, a path to one saved with "
                "gate.save(), or the dict it serialises to. Fit one with "
                "`taut calibrate` or ConformalGate.calibrate()."
            )
        if isinstance(gate, ConformalGate):
            return gate
        if isinstance(gate, str):
            return ConformalGate.load(gate)
        if isinstance(gate, dict):
            return ConformalGate.from_dict(gate)
        raise TypeError("gate must be a ConformalGate, a path, or a serialised dict; got %r"
                        % type(gate).__name__)

    def _decide(self, input: Any) -> Dict[str, Any]:
        """The gated result for one input, predicting only when it is not already given."""
        raw = input
        if isinstance(input, dict) and isinstance(input.get("answers"), dict):
            # Already scored -- by an upstream node, or by taut-serve. Do not pay twice.
            result = {k: v for k, v in input.items() if k != self.output_key}
        else:
            if not self.questions:
                raise ValueError(
                    "TautGate needs `questions` to score an input that does not already "
                    "carry an 'answers' block"
                )
            text = _extract_text(raw, self.state_key)
            result = _execute_decision(
                text, self.questions, agent=self.agent, base_url=self.base_url,
                api_key=self.api_key, model=self.model,
            )
        gated = self.gate.apply(result)
        if self._monitor is not None:
            self._monitor.observe(gated)
        return gated

    def _verdict(self, gated: Dict[str, Any]) -> Dict[str, Any]:
        blocks = {qid: ans["gate"] for qid, ans in gated.get("answers", {}).items()
                  if isinstance(ans, dict) and "gate" in ans}
        abstained = {qid: b for qid, b in blocks.items() if not b.get("accepted", True)}
        if not blocks:
            accepted = None                       # nothing gated: not a claim either way
        elif self.accept_when == "any":
            accepted = len(abstained) < len(blocks)
        else:
            accepted = not abstained
        record = dict(gated.get("gate", {}))
        record.update({"accepted": accepted, "abstained": sorted(abstained),
                       "questions": blocks, "accept_when": self.accept_when})
        return record

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        gated = self._decide(input)
        verdict = self._verdict(gated)

        if self.action == "route":
            # A conditional edge wants a branch name, not a payload.
            return "accept" if verdict["accepted"] else "escalate"

        if self.action == "raise" and verdict["accepted"] is False:
            raise TautGateEscalation(
                "Taut gate abstained on %s; the certified guarantee does not cover this "
                "decision" % (verdict["abstained"],),
                abstained={q: verdict["questions"][q] for q in verdict["abstained"]},
                raw_decision=gated,
            )

        if isinstance(input, dict):
            out = dict(input)
            out.update(gated)
            out[self.output_key] = verdict
            return out
        return {"input": input, "answers": gated.get("answers", {}), self.output_key: verdict}

    def report(self) -> str:
        """What the gate certifies, for a startup log line."""
        return self.gate.report()

    def drift(self) -> Dict[str, Any]:
        """The monitor's verdict. Requires ``monitor=True``."""
        if self._monitor is None:
            raise ValueError("TautGate was built with monitor=False; construct it with "
                             "monitor=True to track whether the gate has expired")
        return self._monitor.check()

    def __call__(self, state: Any) -> Any:
        return self.invoke(state)
