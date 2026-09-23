<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/thedataengineer/taut/main/assets/logo-lockup-dark.png" />
    <img src="https://raw.githubusercontent.com/thedataengineer/taut/main/assets/logo-lockup.png" alt="Taut" width="330" />
  </picture>
</p>

**Typed decisions in one forward pass, behind a risk bound you can certify.** Taut answers
typed questions over 100+ languages in 33 ms with no text generation — and then puts a
*contract* on the answer: a threshold fitted to your risk budget, carrying a
distribution-free, finite-sample guarantee, plus a monitor that tells you when that
guarantee has expired.

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/taut.svg)](https://pypi.org/project/taut/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-thekarteek%2Ftaut-blue)](https://huggingface.co/thekarteek/taut)
[![Multilingual](https://img.shields.io/badge/%F0%9F%A4%97%20Model-taut--multilingual-blue)](https://huggingface.co/thekarteek/taut-multilingual)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

> **Taut is a fork of [Laya](https://github.com/NandhaKishorM/laya)** by Convai Innovations,
> Apache-2.0. The encoder, the RLCD training recipe and the router are theirs. What Taut adds
> is the layer above the score: [certified risk control](#certified-risk-control), drift
> monitoring, and the operational surface around them. See [Attribution](#attribution).

<p align="center">
  <img src="https://raw.githubusercontent.com/thedataengineer/taut/main/assets/taut_vs_jev_full.png" alt="Taut versus TypeSafe Jev: accuracy on shared public datasets, every application workflow, all 51 languages, speed, calibration, and the cost of not preloading" width="100%" />
</p>

Taut evaluates typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket or JSON document) in **a single forward pass** — 33 ms for one question, 7.2 ms/question batched, measured on a T4. No text generation, so nothing to parse and nothing to hallucinate.

Three checkpoints, and a `Router` that picks between them per request:

| | encoder | params | context | use it for |
|---|---|---|---|---|
| [`taut`](https://huggingface.co/thekarteek/taut) | ModernBERT-large | 421M | 512 | English |
| [`taut-multilingual`](https://huggingface.co/thekarteek/taut-multilingual) | mmBERT-base | 322M | 1024 | 100+ languages, 2x faster |
| [`taut-typed-decisions`](https://huggingface.co/thekarteek/taut-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

### What's new in 0.4.0

* **Certified risk control.** `taut.conformal` turns a calibrated probability into an operating contract: a threshold, prediction set or interval carrying a distribution-free, finite-sample guarantee. Four modes — `selective`, `miss`, `set`, `interval` — fitted from a labelled split, serialised to JSON, applied without torch. It reports when your calibration set is too small to certify anything rather than returning a threshold it cannot stand behind. See [Certified Risk Control](#certified-risk-control).

### What's new in 0.3.9

* Documentation update only. There are no code or checkpoint changes from 0.3.8.

### What's new in 0.3.8

* **Batch scoring.** `agent.predict_batch(states, questions)` scores many states in shared forward passes, with answers identical to calling `predict` one state at a time. See [Batch Mode](#batch-mode-score-many-states-in-one-forward-pass).
* **Faster paths, all opt-in.** `taut.load(..., fast=True)` uses a TileLang GPU fast path that matches the stock bf16 forward within rounding (see [GPU Fast Path](#gpu-fast-path-tilelang)). `Agent(compile=True)` enables `torch.compile`, and `taut.onnx_agent.ONNXAgent` runs an exported model on ONNX Runtime. A second `Agent` for the same checkpoint reuses its parsed tokenizer, so it loads in about 0.5 s instead of about 3 s.
* **`import taut` no longer loads torch.** Routing, language detection and e-mail cleaning work in lightweight processes; torch loads on first use of a model.
* **New ways to call Taut.** A `taut` [command](#command-line) for quick local tests, an optional [MCP server](#mcp-server-optional) (`pip install "taut[mcp]"`), [LangChain and LangGraph](#langchain-and-langgraph-integration) routing, guardrails, triage and evaluation (`pip install "taut[langchain]"`), and `taut-ts/`, a TypeScript package for Node and the browser that gives the same answers as the Python package.
* **HTTP server fixes.** Inference runs off the event loop, so one request no longer stalls `/health` and other clients. The published Hugging Face ids select their checkpoint, and Docker Compose gains an HTTP service.
* **Routing.** CJK text containing Latin brand names, romanized Bangla, and Azerbaijani now reach the multilingual checkpoint. Checked on 20,000 English texts: at most 5 English sentences move, all quoting long native-script names.
* **Correctness fixes.** Long conversation lists keep the newest turn when truncated. Non-ASCII instructions reach the model as text instead of `\uXXXX` escapes. `truncate_left` with no room left keeps none of the state rather than all of it. `Router.preload([])` loads nothing. `noul` questions accept an opt-in `labels` override. Intel XPU devices are detected.
* **E-mail cleaning.** A body that mentions "confidential", or a line that starts with "Thanks for" or "Best", is no longer deleted as boilerplate, and more real sign-offs are removed.
* **Fine-tuning.** Decision-head activation checkpointing is honoured, so the notebook's memory setting takes effect.

### What's new in 0.3.7

* **About 10x faster loading.** Checkpoints are built without the throwaway random weight initialisation, so `taut.load()` drops from about 22 s to about 2 s on CPU with bit-identical answers. This also skips the pass that crashed on Windows with Python 3.14 (#123).
* **Better routing for non-English Latin text.** Plain-ASCII Spanish, Italian, Portuguese and French (accents stripped by mail clients and ticket systems) and Brazilian Portuguese support text now reach the multilingual checkpoint. Letters in scripts the router has no range for no longer fall through to English, and URLs, e-mail addresses and dotted names no longer count as words. Checked on 20,000 English texts, with no English prose moved.
* **A Jev-compatible HTTP server.** `pip install "taut[serve]"`, then `taut-serve`, speaks `POST /v1/systemone`, so existing TypeSafe clients work by changing `baseUrl`. See [Self-Hosting](#self-hosting-http-server-jev-compatible).
* **Router defaults and hooks.** `Router()` keeps two checkpoints resident, so alternating languages no longer reload a model on every request. You can pass your own language guess with `lang_guess=` and send undecided text to `Router(default=...)`. `Router` and `Agent` also work as context managers, and evicted models free their memory.
* **Clearer errors and safer edge cases.** A malformed question is rejected with a message naming the question and what to fix. An empty question set returns an empty answer, and invalid temperatures in a checkpoint no longer stop it loading.
* **E-mail cleaning for Portuguese and Spanish** replies, signatures and footers.
* **Fine-tuning notebook fixes.** Calibration is now fitted on a held-out slice rather than on training data (#186), and stale temperature overrides are cleared before a refit.
* **Docker quickstart** under `docs/docker.md`.

---

## Installation

Python 3.10 or newer. The dependencies set that floor: `huggingface_hub` 1.x, `transformers` 5.x and `torch` 2.14 all require 3.10.

**Optional PyTorch build selection:** If you need a CPU-only or GPU-specific PyTorch build, follow [PyTorch's installation guide](https://pytorch.org/get-started/locally/) after creating your virtual environment and before installing Taut. Replace `pip` or `pip3` in the selected command with the environment's Python executable followed by `-m pip`.

If you already use a virtual environment, install the PyPI release with:

```bash
python -m pip install taut
```

For a new environment, choose the commands for your platform below. Run them from your project directory; the explicit Python paths keep installation and verification in the same environment.

**macOS / Linux** (with Python 3.10 or newer):

On Debian/Ubuntu, the system Python may require `sudo apt install python3-venv` before creating a virtual environment. If `venv` reports that `ensurepip` is unavailable, install that package and retry.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install taut
.venv/bin/python -I -c "import taut; print(taut.__version__)"
```

**Windows PowerShell** (this example uses an installed Python 3.11):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install taut
.\.venv\Scripts\python.exe -I -c "import taut; print(taut.__version__)"
```

Both checks print the installed Taut version without loading a checkpoint. `-I` excludes the current directory from the import search path, so a local source copy cannot mask a missing installation. Keep using the same virtual environment's Python when running your application.

**Install from GitHub**

To use the development version instead of the PyPI release, create the virtual environment above and replace its installation command with the appropriate command below. Git must be installed.

```bash
# macOS / Linux
.venv/bin/python -m pip install "git+https://github.com/thedataengineer/taut.git"
```

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install "git+https://github.com/thedataengineer/taut.git"
```

Run the same version check afterward. The GitHub version follows the repository's default branch and may differ from the published release.

**Model setup and troubleshooting**

Continue with the [Router quickstart](#quickstart-route-mode-recommended) to run inference. Loading a Hub checkpoint requires access to Hugging Face on its first download; the quickstart's `Router(preload=True)` loads all three configured checkpoints at construction.

- **`ModuleNotFoundError: No module named 'taut'`:** run both installation and your script with the same virtual environment's Python executable shown above. In an editor, select that interpreter as well.
- **Missing `rl_agent_config.json`:** this file ships with a Taut checkpoint alongside `model.safetensors`; it is not a configuration file you need to create in the source repository. For a local model, pass the directory containing those checkpoint files.

---

### Command line

Installing the package also installs a `taut` command for quick local testing, no script needed:

```bash
taut "I was charged twice, please refund"            # routing decision only; works offline, no download
taut "Refactor this service" --predict               # full answers (downloads the checkpoint on first use)
taut "Mein Konto wurde zweimal belastet" --lang de   # force a language instead of detecting it
taut                                                 # interactive mode
```

Routing alone never downloads a checkpoint, so it returns in milliseconds. `--predict` loads the routed checkpoint, which needs network access to the Hugging Face hub the first time; if a checkpoint cannot be downloaded, the CLI says so instead of crashing.

---

### First run: publish the checkpoint mirrors

Taut loads its weights from `thekarteek/taut*` on Hugging Face — mirrors of upstream's
Apache-2.0 Laya checkpoints, republished unmodified so this package does not depend on
another account's repo names staying put. If you are running from a fresh clone of this
fork and those mirrors do not exist yet, publish them once:

```bash
huggingface-cli login
python scripts/mirror_checkpoints.py --dry-run    # show the plan
python scripts/mirror_checkpoints.py              # ask, then publish
```

Or skip mirroring entirely and load upstream's originals:

```bash
export TAUT_MODEL_ORG=convaiinnovations
```

Nothing is retrained or altered either way; they are the same weights.

---

## Quickstart: Route Mode (Recommended)

To try the Python SDK in a CPU container, see the
[Docker Compose quickstart](docs/docker.md). It runs a sample request and keeps
downloaded models between runs.

Taut ships three checkpoints. The built-in **`Router`** is the recommended entry point: it evaluates any state in any language, automatically detects scripts and languages in sub-milliseconds, and dispatches to the optimal checkpoint in a single forward pass.

```python
from taut import Router

# Preload checkpoints into memory for instant sub-35ms routing
router = Router(preload=True)

# 1. State in any language or schema
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 2. Define your typed questions
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

# 3. English state -> automatically routed to taut (ModernBERT-large, 39.5 ms)
res_en = router.predict(state, questions)
print("Department :", res_en["answers"]["department"]["choice"])  # -> billing (confidence: 0.94)
print("Routing    :", res_en["routing"]["model"])                 # -> english

# 4. Hindi state -> automatically routed to taut-multilingual (mmBERT-base, 32.8 ms)
res_hi = router.predict({"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions)
print("Department :", res_hi["answers"]["department"]["choice"])  # -> billing (confidence: 0.86)
print("Routing    :", res_hi["routing"]["model"])                 # -> multilingual

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")
```

Every result carries full routing metadata explaining why the choice was made:

```python
res_hi["routing"]
# {
#   'model': 'multilingual',
#   'repo': 'thekarteek/taut/multilingual',
#   'reason': 'non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it'
# }
```

Inspect a routing decision without running any forward pass:

```python
router.route({"body": "Der Kunde wurde zweimal belastet"}, questions).reason
# "Latin script but language looks like 'de', not English"
```

Very short Latin-script text often carries nothing that identifies its language (`"Quero cancelar"`, `"Esqueci minha senha"`). Such text goes to `default`, which is `"english"` unless you change it. If most of your traffic is not English, set:

```python
router = Router(default="multilingual")
router.route({"body": "Esqueci minha senha"}).model                 # -> multilingual
router.route({"body": "Please refund the duplicate charge"}).model  # -> english
```

### Why Route: The Evidence

On a shared benchmark (17,416 questions, one T4 GPU, identical questions per model):

| Benchmark / Task | English (`taut`) | Multilingual (`taut-multilingual`) | `Router` (Routed) |
|---|---|---|---|
| MASSIVE intent, English | **0.783** | 0.657 | **0.783** |
| MASSIVE intent, 13 other languages | 0.306 | **0.451** | **0.451** |
| XNLI, English | **0.860** | 0.843 | **0.860** |
| XNLI, 14 other languages | 0.521 | **0.731** | **0.731** |
| Languages usable (>3x random) | 23 / 51 | 45 / 51 | **45 / 51** |
| Latency, 1 question (T4 GPU) | 39.5 ms | **32.8 ms** | **32.8 ms** |
| Latency, 10 questions batched | 158.6 ms | **72.3 ms** | **72.3 ms** |

The English checkpoint collapses on non-Latin scripts (Khmer scores **0.000 accuracy at 0.952 confidence**). Because the model stays confident while being wrong, confidence gating cannot save you. `Router` detects the script in <0.5 ms pure Python before the forward pass.

### Production Preload & Memory

A cold checkpoint build costs seconds; language detection costs microseconds. The lazy default keeps **two** checkpoints resident — `english` and `multilingual`, the only two automatic routing chooses between — so a language flip costs detection only once each has been built. `max_loaded=1` rebuilds the checkpoint it just evicted on *every* switch (measured at a 7.4 s median reload on CPU and 10.3 s on T4), and traffic that only ever sees one language never builds the second, so the default costs a single-language deployment nothing.

For a server or production app, preload:

```python
# Every checkpoint resident in memory; language flips cost detection only (<1 ms)
router = Router(preload=True)
router = Router(preload=True, device="cuda")

# Or preload only the specific checkpoints you serve:
router.preload(["english", "multilingual"])

# If your app already built an agent, attach it to avoid duplicate VRAM:
router.attach("english", existing_agent)

# Manage resident memory (default keeps two hot: english + multilingual, LRU eviction)
router = Router(max_loaded=3)       # keep all three hot, e.g. with auto_task_detection
router = Router(max_loaded=1)       # memory-constrained host, reloads on every switch
router.unload()                     # free memory
```

| Deployment Mode | Per-Request Latency | Model Reloads |
|---|---|---|
| `Router()` (lazy, `max_loaded=2`) | detection only (<1 ms) on a switch, after each language's first load | 1 the first time a language appears |
| `Router(max_loaded=1)` | 7 to 10 s on every language switch | 1 per switch |
| `Router(preload=True)` | **32.8 ms (GPU) / 193–464 ms (CPU)** | **none** |

A rebuild still re-reads the checkpoint, but each checkpoint's tokenizer is parsed once per process
and reused by every `Agent` — including one the Router rebuilds after eviction. The multilingual
`tokenizer.json` alone is 34 MB / 256k vocab, several times the cost of applying its weights.
Preloading is still the right answer for a server: it removes the rebuild rather than making it
cheaper.

### Supplying Your Own Language Detection

Routing asks one question: *can the English checkpoint read this state?* The built-in detector answers it from the script and a function-word heuristic, and is deliberately dependency-free. That heuristic is best-effort on Latin-script languages it holds no word list for, so a short request can carry no usable signal:

```python
from taut.lang import analyse
analyse("Care este ora in Tokyo?")
# {'script': 'latin', 'language': 'en', 'is_english': True}   -> the English checkpoint
```

If you already run a language-identification model, hand routing the answer instead of relying on the heuristic. `lang_guess` takes a language code or a callable receiving the state, and is checked after an explicit `lang=` and before detection:

```python
# A code you already know
router.predict(state, questions, lang_guess="ro")

# A callable, e.g. wrapping fastText, CLD3 or a transformer LID
router.predict(state, questions, lang_guess=lambda s: my_lid(s))

# Or install one for every request on a server
router = Router(preload=True, lang_guess=my_lid)
```

The hint only decides *English or not*: a code whose primary subtag is `en`, `eng` or `english` routes to the English checkpoint and everything else routes to the multilingual one. `"en_US"` and `"en_US.UTF-8"` are read as English, so `$LANG` can be passed straight through. Returning `None`, or an empty code, makes it abstain and the built-in detector decides as before — so a LID model that is unsure does not force a checkpoint. An explicit `model=`, `task=` or `lang=` still wins, and the default path is unchanged.

---

## Self-Hosting: HTTP Server (Jev-compatible)

`taut.serve` exposes the `Router` over HTTP on the same `POST /v1/systemone`
wire protocol as TypeSafe's hosted Jev API. Taut's answer payload is already
schema-identical to what Jev returns (`choice`/`score`/`noul` answers and a
`{input_tokens, output_tokens}` usage block), so an existing Jev client — e.g.
the [`hs-jev`](https://github.com/getmissionctrl/hs-jev) Haskell client — just
needs its `baseUrl` repointed; nothing else changes.

```bash
pip install "taut[serve]"          # adds fastapi + uvicorn
TAUT_DEVICE=cuda TAUT_PRELOAD=1 taut-serve   # binds 0.0.0.0:8000, preloads all 3 checkpoints
```

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"body": "billed twice, refund please or we cancel"},
  "questions": {"dept": {"type": "choice", "instructions": "which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"}}}
}'
```

Configuration is by environment variable: `TAUT_HOST`, `TAUT_PORT`,
`TAUT_DEVICE`, `TAUT_PRELOAD`, `TAUT_MODELS` (comma list to preload),
`TAUT_THREADS` (cap torch intra-op threads for CPU inference — keep at or below
physical cores), `TAUT_AUTO_TASK`, and `TAUT_API_KEY` (when set, clients must
send `Authorization: Bearer <key>`). A client's `model` field is honoured when it
names a Taut checkpoint (`english`/`multilingual`/`typed-decisions`), otherwise
the router auto-selects by script/language.

### Nix / NixOS

This repo is a flake. On a machine with an NVIDIA GPU:

```bash
nix run .#taut-serve          # build (prebuilt CUDA torch, no compile) and serve
nix develop                   # dev shell: torch-bin, transformers, fastapi, pytest
```

For a NixOS host, import the module and enable the service:

```nix
# flake inputs:  taut.url = "github:<you>/taut";  # or path:/… on the same host
{
  imports = [ taut.nixosModules.default ];
  services.taut-serve = {
    enable = true;
    host = "0.0.0.0";           # or bind to the Tailscale/LAN address
    openFirewall = true;
    device = "cuda";
    models = [ "english" "multilingual" "typed-decisions" ];
    # apiKeyFile = config.age.secrets.taut-api-key.path;  # optional bearer auth
  };
}
```

The module runs a hardened `DynamicUser` systemd unit with CUDA device access,
caches weights under `/var/lib/taut-serve`, and reads the bearer token (if any)
via `LoadCredential` so it never enters the store.

---

## Single-Model Mode (Direct SDK)

If you only need a single checkpoint for a dedicated pipeline, you can load models directly:

```python
import taut

# 1. Load a specific checkpoint directly from the hub
agent = taut.load("thekarteek/taut")                           # English root
agent_ml = taut.load("thekarteek/taut", subfolder="multilingual") # 100+ languages
agent_td = taut.load("thekarteek/taut", subfolder="typed-decisions")

# 2. Run all questions in ONE single forward pass (~35 ms on GPU)
result = agent.predict(state, questions)
answers = result["answers"]

print("Department :", answers["department"]["choice"])   # -> billing (confidence: 0.94)
print("Urgency    :", answers["urgency"]["score"])        # -> 1.84 / 2.0
print("Churn Risk :", answers["churn_risk"]["noul"])       # -> 0.892 (89.2% probability)
```

Passing an empty question dictionary to `agent.predict(state, {})` or
`agent.system_one(state, {})` returns the standard response with `"answers": {}`
and `"usage": {"input_tokens": 0, "output_tokens": 0}`. The state is not tokenized
and no model forward pass runs.

### Batch Mode: score many states in one forward pass

`predict` handles one state per call, which leaves most of the GPU's batch dimension idle. When you
have a list of items to score against the *same* questions — a backlog of tickets, a table of rows,
a log slice — `predict_batch` packs them into shared forward passes:

```python
states = [{"body": t} for t in ticket_texts]           # a list of states

results = agent.predict_batch(states, questions)       # one forward pass for the whole list
# results[i] is exactly what agent.predict(states[i], questions) would return

# Bound peak memory when the list (or the texts) are large — chunk into passes of N:
results = agent.predict_batch(states, questions, batch_size=64)
```

Results are aligned with `states` by index and identical in shape to `predict`. Decisions match the
one-at-a-time path exactly (numbers are bit-identical on CPU; on GPU they can differ in the 4th
decimal because fp16 autocast reorders reductions across padding widths). Batching is a **GPU
throughput win** — on an RTX 5060 Ti, per-decision latency drops from ~10 ms one-by-one to ~1 ms
batched (measured ~9–10×). On CPU the model is already compute-bound, so batching does not speed it
up; use it there only for API convenience.

---

## GPU Fast Path (TileLang)

`pip install taut[fast]` adds an optional forward built from fused [TileLang](https://github.com/tile-ai/tilelang)
kernels: GEMM + bias/activation epilogues, GEMM + GEGLU, residual + LayerNorm, in-place RoPE, and a
sliding-window flash attention that reads the packed QKV buffer directly. Weights stay resident in bf16
and every (batch, length) bucket is captured as a CUDA graph, so a one-question call no longer pays
~200 kernel launches from Python.

```python
agent = taut.load("thekarteek/taut", fast=True)   # or: agent.accelerate()
agent.predict(state, questions)                            # same API, same answers
```

Numerics: on a fixed set of 60 states the fast path is at least as close to an fp32 forward as the stock bf16
path is (max |Δp| ≤ 0.05 vs fp32 on both checkpoints, argmax agreement ≥ 47/48 per question type; every per-option
probability is in `benchmarks/results/parity_*.json`) — see `benchmarks/parity_fast.py` and [BENCHMARKS.md](BENCHMARKS.md#gpu-fast-path).
Falls back to the stock forward on CPU/MPS or when `tilelang` is not installed; `agent.deaccelerate()`
restores it. Kernels compile once per shape bucket on first use (a few seconds, cached on disk).

---

## Automated Confidence Gating

Because Taut's probabilities are trained with strictly proper scoring rules (RLCD), confidence scores are statistically meaningful:

```python
dept = answers["department"]["choice"]
conf = answers["department"]["confidence"]

if conf >= 0.85:
    # High confidence: automated action without human in the loop
    route_automatically(dept)
else:
    # Low confidence: escalate to human triage
    escalate_to_human_agent(dept, reason=f"Low confidence ({conf:.2f})")
```

That `0.85` is a guess. It is a reasonable guess — calibrated probabilities are what make
it reasonable — but nothing here tells you what the guess costs, and nobody can sign off
on a number whose error rate is unknown. [Certified Risk Control](#certified-risk-control)
replaces it with a threshold that is fitted, not chosen, and that carries a bound.

---

## Certified Risk Control

Every model in this space returns a score. A score is not an operating decision, and the
gap between them is where the work actually is: someone still has to pick a cutoff, and
own what it lets through.

`taut.conformal` closes that gap. Give it a labelled held-out split and a risk budget, and
it returns a threshold with a **distribution-free, finite-sample guarantee** — valid for
any data distribution, at the sample size you actually have, with no asymptotics and no
assumption that the model is well calibrated:

> at most 2% of incoming tickets are auto-handled incorrectly, with 95% confidence

```python
from taut import Agent, ConformalGate, triage_questions

agent = Agent()
questions = triage_questions()

# 1. Score a labelled held-out split
results = agent.predict_batch(dev_states, questions)

# 2. Fit the gate to a risk budget you can defend in a review
gate = ConformalGate.calibrate(results, dev_labels, alpha=0.02, delta=0.05)
print(gate.report())
gate.save("triage_gate.json")

# 3. Apply it per request
decision = gate.apply(agent.predict(ticket, questions))
if decision["answers"]["intent"]["gate"]["accepted"]:
    auto_route(decision["answers"]["intent"]["choice"])
else:
    escalate(ticket)
```

```
Taut conformal gate  alpha=0.02  delta=0.05  n=1200

question               type      mode       operating point          guarantee
------------------------------------------------------------------------------
intent                 choice    selective  keep 45.7% @ tau=0.705   accepted-and-wrong <= 0.02 of all
unsafe                 noul      miss       block 52.2% @ tau=0.285  miss <= 0.02 of positives
severity               score     interval   +/- 2.13 levels          coverage >= 0.98
```

That table is the point. It states in one line what you are buying and what it costs —
the coverage you keep, the threshold that buys it, and the guarantee attached — which is
the artefact a risk owner signs, not a score.

### Four guarantees

| mode | question types | the contract |
|---|---|---|
| `selective` | `choice`, `noul` | at most `alpha` of **all** decisions are both auto-handled and wrong |
| `miss` | `noul` | at most `alpha` of **true positives** slip past the guardrail |
| `set` | `choice` | the returned **set of options** contains the truth with probability ≥ `1 - alpha` |
| `interval` | `score` | the true level falls in the returned **interval** with probability ≥ `1 - alpha` |

`selective` and `miss` are fitted by fixed-sequence testing over a threshold grid fixed
before any data is read, with exact **Clopper-Pearson** bounds — not Hoeffding, not a
normal approximation, because real calibration sets are hundreds of rows, not thousands.
`set` and `interval` are split conformal with the finite-sample `(n+1)` correction.

### It tells you when it cannot help

An exact binomial bound on `n` points cannot fall below `alpha` until
`n ≥ log(delta) / log(1 - alpha)`, however flawless the record. Certifying 5% at 95%
confidence needs 59 labelled points, 2% needs 149, 1% needs 299.

```python
>>> from taut import min_calibration_size
>>> min_calibration_size(0.01, 0.05)
299
```

Below that floor the gate does not return a threshold it cannot stand behind. It abstains,
sets `certifiable: False`, and reports a `shortfall` block naming the floor it missed and
how many rows short it is. Refusing to answer is a feature — it is the whole difference
between a guarantee and a number.

### Read `family_alpha`, not `alpha`, for a whole record

Gating five questions at 2% each does **not** make the record 2%. By the union bound it is
up to 10%, and the record-level block says so:

```python
>>> decision["gate"]["family_alpha"]
0.1
>>> decision["gate"]["family_guarantee"]
'at most 0.1 of accepted records are wrong on at least one of the 5 gated questions,
 with confidence 0.75 (union bound)'
```

Silently inheriting the per-question number here is the easiest way to misuse risk
control, so the module refuses to let you.

### The gate ships without the model

A fitted gate is JSON: thresholds and diagnostics, no weights. `taut.conformal` is pure
NumPy and imports neither torch nor transformers, so a gate can be fitted on a GPU box,
committed to your repo, reviewed as a diff, and applied in an edge process that never
loads a model.

```bash
$ python -c "import taut, sys; taut.ConformalGate.load('triage_gate.json'); print('torch' in sys.modules)"
False
```

### Serve it, or run it from the shell

Point `TAUT_GATE` at a saved gate and `taut-serve` stops being a scorer and becomes a
decision service. The risk budget is the operator's, fixed at deploy time and readable
from `/health` — deliberately not a request field, because a client that can name its own
`alpha` can claim any guarantee it likes.

```bash
TAUT_GATE=triage_gate.json taut-serve
```

```jsonc
// GET /health
{"status": "ok", "gate": {"alpha": 0.02, "delta": 0.05, "questions": ["intent"],
                          "modes": {"intent": "selective"},
                          "family_alpha": 0.02, "family_guarantee": "..."}}

// POST /v1/systemone -> every answer carries its gate block
{"answers": {"intent": {"choice": "billing", "gate": {"accepted": true, ...}}},
 "gate": {"accepted": true, "family_alpha": 0.02, ...}}
```

A gate that will not load is fatal rather than a warning: a server configured to certify
its answers and then quietly not doing so is the worst outcome available. Set
`TAUT_GATE_STRICT=1` to reject requests carrying questions the gate was not calibrated on,
instead of passing them through wearing no guarantee.

From the shell:

```bash
taut --gate triage_gate.json --report              # what does this gate certify?
taut "billed twice" --predict --gate triage_gate.json
```

```
intent      : ACCEPT     at most 0.02 of all decisions are accepted and wrong, with confidence 0.95

record      : at most 0.02 of accepted records are wrong on at least one of the 1 gated questions, with confidence 0.95 (union bound)
```

### A gate with no expiry date is a guarantee someone will keep quoting

The bound is distribution-free but not distribution-*proof*. It needs exchangeability:
calibration and live traffic drawn from the same distribution. A gate fitted on last
quarter's tickets carries no guarantee on this quarter's if the mix moved — and nothing in
the gate itself notices.

`GateMonitor` does, without labels:

```python
from taut import ConformalGate, GateMonitor

gate = ConformalGate.load("triage_gate.json")
monitor = GateMonitor(gate)

for ticket in stream:
    monitor.observe(gate.apply(agent.predict(ticket, questions)))

if monitor.check()["status"] == "expired":
    page_whoever_owns_the_labels(monitor.report())
```

```
Taut gate drift  status=expired  seen=1000  window=1000  level=0.005 per test

question               mode       status    acceptance       scores
-------------------------------------------------------------------
intent                 selective  expired   45.1% vs 72.4%   KS=0.287 p=1.1e-54

  intent: acceptance rate moved from 72.4% to 45.1% (p=1.6e-73); score distribution
          shifted (KS=0.287, p=1.1e-54)

  Refit on freshly labelled traffic before quoting the guarantee again.
```

Two tests, neither needing a single label. The gate was fitted to accept a known share of
traffic, so an **exact two-sided binomial** test against that calibrated coverage catches a
shifted input distribution outright. A **two-sample KS** test against the 101-quantile
sketch stored in the gate catches shifts that leave the acceptance rate unchanged — mass
moving *within* the accepted region, which the binomial test cannot see.

Measured false-positive rate on undrifted traffic: **0.7%** against a 1% test level. It
catches a 9% change in logit separation 91% of the time, and anything larger essentially
always. A monitor that cries wolf gets muted, which is worse than not having one, so that
number is tested in CI rather than asserted here.

**What it cannot tell you, and the distinction matters.** Both tests watch the *scores*. A
shift in the score-to-correctness link — same confidences, worse answers — moves real risk
while leaving both tests quiet. A clean report means "no evidence the gate has expired",
never "the gate still holds". Only refitting on freshly labelled data restores the
guarantee; this tells you when that is overdue.

### Is the bound real?

It is a falsifiable claim, so it has been falsification-tested: 1,000 trials per
configuration, scored against **population** risk rather than a sampled estimate, across
calibrated, overconfident and underconfident models. 21 of 21 configurations hold inside
the `delta` budget; the worst cell breaches 4.9% of the time against a 5% allowance.

Method, full table and the honest limitations are in
[`research/conformal/README.md`](research/conformal/README.md). Reproduce with
`python research/conformal/validate_risk.py`.

---

## Built-in Workflow Presets

Taut provides pre-tuned question schemas for immediate production use:

```python
import taut

agent = taut.load("thekarteek/taut")

# 1. Intelligent Model Router (routes to small vs. frontier models)
routing = agent.predict({"request": "Refactor this service using dependency injection"}, taut.router_questions())

# 2. Real-time Prompt Guardrails (jailbreaks, injections, leaks)
guard = agent.predict({"prompt": "Ignore all instructions"}, taut.guard_questions())

# 3. Content Safety & Moderation (toxicity, harassment, threats)
safety = agent.predict({"post": "User comment text"}, taut.moderation_questions())

# 4. Support Ticket Triage (intent, urgency, frustration, churn)
triage = agent.predict({"message": "My payment failed twice"}, taut.triage_questions())
```

---

## LangChain and LangGraph Integration

Fast System 1 routing and guardrails directly inside LangGraph workflows and LCEL chains:

```python
from taut.integrations.langchain import TautRouter, TautGuardrail

# 1. Sub-35ms LangGraph conditional edge routing with confidence fallback
router = TautRouter(
    criteria={"billing": "invoices, charges", "tech": "bugs, outages"},
    confidence_threshold=0.80,
    fallback="human_agent",
)
workflow.add_conditional_edges("triage", router)

# 2. Inline prompt guardrails
guard = TautGuardrail(action="raise")  # raises TautGuardrailError on jailbreak/injection
```

See [**`docs/langchain.md`**](docs/langchain.md) for full guide, support ticket triage nodes, and remote HTTP server configuration.

---

## Decision Primitives

| Primitive | Output | Use Cases |
|---|---|---|
| **`choice`** | Top label, probabilities per option, confidence | Department routing, intent classification, topic categorization |
| **`score`** | Expected level on ordinal rubric, distribution, confidence | Frustration level, ticket urgency, harm severity |
| **`noul`** | Calibrated probability P(true) from 0.0 to 1.0 | Phishing detection, spam filtering, jailbreak detection, churn risk |

`noul` always scores two semantic slots in `[false, true]` order and returns the probability of
the second slot. For compatibility, those slots are shown to the model as `false` and `true` by
default. The optional `labels` mapping overrides only that model-facing text without changing the
returned meaning:

```python
question = {
    "type": "noul",
    "instructions": "Is this review positive?",
    "criteria": {
        "false": "the review is negative",
        "true": "the review is positive",
    },
    "labels": {
        "false": "B",
        "true": "A",
    },
}
```

The `labels` mapping is optional. It must contain exactly the string keys `false` and `true`,
whose values must be distinct non-empty strings. Mapping order does not matter, and the returned
`noul` value is still P(true). Label sensitivity varies by checkpoint and state, so validate any
override on your own data rather than treating `A`/`B` as a universal fix.

---

## MCP Server (Optional)

Taut can be exposed as an [MCP](https://modelcontextprotocol.io) stdio server, so any MCP
client (OpenClaw, Claude Desktop, Cursor, ...) can call typed decisions as tools
(`taut_predict`, `taut_route`, `taut_preset`, `taut_status`) without writing glue code.
This is an **optional extra**: the core package has no `mcp` dependency.

```bash
pip install "taut[mcp]"
taut-mcp-server          # or: python -m taut.mcp.server
```

Example MCP client configuration (stdio transport):

```json
{
  "mcpServers": {
    "taut": {
      "command": "taut-mcp-server",
      "env": { "TAUT_DEVICE": "cpu" }
    }
  }
}
```

The environment variables follow the contract documented at the top of
[`taut/serve.py`](taut/serve.py), so the same variable has one meaning across the
package:

| Variable | Default | Meaning |
|---|---|---|
| `TAUT_DEVICE` | (auto) | Same as `taut.serve`: the value is passed straight to torch |
| `TAUT_PRELOAD` | `1` | Same as `taut.serve`: build the checkpoints at startup, not lazily |
| `TAUT_MODELS` | `english,multilingual` | Comma list to preload (serve contract). MCP difference: an empty value preloads `english,multilingual` so `typed-decisions` stays lazy; in `taut.serve` empty means every checkpoint |
| `TAUT_THREADS` | (torch default) | Same as `taut.serve`: cap torch intra-op threads for CPU inference; keep it at or below the physical core count |

The tools return structured JSON (answers with probabilities, routing metadata, device,
`latency_ms`). As with the SDK, use it for structured decisions only; not for open Q&A or
text generation. Tests: `tests/test_mcp.py` (CI, no weights) and
`tests/test_mcp_local_e2e.py` (local, real weights and a real stdio handshake).

---

## Benchmarks

Community diagnostic: [Chinese workplace decisions (Feishu-style)](research/benchmarks/feishu_zh/README.md) · [中文说明](research/benchmarks/feishu_zh/README.zh-CN.md). Includes frozen synthetic cases, archived paired Taut/Jev responses, and an offline audit; separate from the benchmark suites below.

**Full report: [`BENCHMARKS.md`](BENCHMARKS.md)** — every run consolidated, languages and themes, with per-language detail for all 51 languages.

<p align="center">
  <img src="https://raw.githubusercontent.com/thedataengineer/taut/main/assets/taut_benchmark.png" alt="Per-language accuracy for both checkpoints across 51 languages" width="100%" />
</p>

All Taut numbers below are measured. Every model answered byte-identical questions
(fixed seed) in the same run. Reproduce with
[`research/scripts/taut_benchmark_colab.ipynb`](research/scripts/taut_benchmark_colab.ipynb) on a T4.

### Speed (Tesla T4, measured)

| questions per call | `taut` | `taut-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

Batched throughput reaches 103-332 questions/sec on a single T4. For reference, TypeSafe Jev
has been independently measured at 236-276 ms p50
([AbdelStark](https://github.com/AbdelStark/jev-benchmarks),
[nibzard](https://github.com/nibzard/decision-model-benchmark)) -- Taut answers a single
question roughly **6-7x faster**.

### Taut (with routing) vs Jev

Every Taut figure is what `Router().predict(...)` actually returns — the checkpoint the router
selects for that input, not a hand-picked best of three. Jev figures are **third-party
published, never measured here** (no TypeSafe API access), so sample sizes and prompts differ.

| | Jev 1.13.0 | Taut (routed) | |
|---|---|---|---|
| typed-decisions, 2,000 decisions | 0.727 | **0.766** | +0.039 |
| AG News, 4 labels | 0.910 | **0.950** | +0.040 |
| DAIR Emotion, 6 labels | 0.480 | **0.595** | +0.115 |
| Banking77 (72 vs 77 labels) | **0.870** | 0.425 | Jev leads on >20 options |
| ECE *(lower better)* | 0.246 | **0.081** | 3× better (post-temperature) |
| p50 latency, 1 question | 236–276 ms | **32.8 ms** | 7.8× faster |
| Languages usable | *no published benchmark* | **45 of 51** | — |
| Weights | closed API | **Apache 2.0** | — |
| Cost | $0.042 / 1M tokens | **$0 self-hosted** | — |

On DAIR Emotion, Jev assigned **zero probability to the true label on 16% of examples** — a hard
failure for anything branching on confidence.

#### Where Jev leads

* **High-cardinality label spaces (>20 options at default settings):** On Banking77, Jev scores 0.870 (on 72 labels) while Taut scores 0.425 (on 77 labels at default 256-token head budget). This is an architectural token-budget constraint: options share a fixed `head_max_len` budget (192 tokens on English, 256 on multilingual), so 77 options receive only ~3 to 4 tokens per label, causing text to become indistinguishable. Jev supports up to 255 options out-of-the-box. While `taut-multilingual` supports 1,024 context (and up to 8,192 in the encoder) and you can raise `agent.cfg["head_max_len"] = 512` at runtime, Jev is currently better suited for 50+ options in a single prompt without tuning. `predict_shortlist` (see [Honest limits](#honest-limits)) keeps the top `k` labels with a caller-supplied embedding, then runs one forward pass on that shortlist.
* **Soft distribution matching:** On typed-decisions, while Taut achieves higher argmax accuracy (0.766 vs 0.727), Jev achieves higher soft accuracy (0.580 vs 0.471) against the teacher's full probability distributions.
* **Out-of-the-box raw calibration:** Before temperature scaling, the base checkpoint has higher raw ECE (0.213 vs 0.144). Taut achieves its 0.081 ECE after domain temperature fitting.

Full detail, including every workflow and all 51 languages: **[`BENCHMARKS.md`](BENCHMARKS.md)**.

### typed-decisions, measured on all three checkpoints

400 cases, 2,000 decisions, four workflows.

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| **`taut-typed-decisions`** | **0.766** | 0.471 | **0.062** | 0.213 | **0.242** |
| `taut` | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| `taut-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *teacher self-agreement ceiling* | *0.735* | | | | |
| *per-question majority class* | *0.461* | | | | |
| *random guess* | *0.318* | | | | |

The fine-tuned checkpoint beats Jev by 3.9 points and clears the teacher ceiling, with 2.4x
better Brier and 1.6x better score MAE. It wins on all four workflows: invoice processing
0.804, security incidents 0.766, customer service 0.764, agent-trace observability 0.730.
By primitive: `noul` 0.857, `choice` 0.733, `score` 0.723.

Two places it still trails Jev: **soft accuracy** (0.471 vs 0.580 — its argmax is better but
its distributions match the teacher less well) and **ECE** (0.213 vs 0.144), which temperature
fitting addresses.

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.342 against 0.461).
All of the capability on this benchmark comes from fine-tuning.

### Multilingual (51 languages, MASSIVE intent, 20 options, random = 0.050)

| | `taut` | `taut-multilingual` |
|---|---|---|
| English | **0.783** | 0.657 |
| 13 other languages | 0.306 | **0.451** |
| XNLI, English | **0.860** | 0.843 |
| XNLI, 14 other languages | 0.521 | **0.731** |

Across all 51 languages the English checkpoint macro-averages **0.227** with macro ECE
**0.733**, and only 23 of 51 languages clear 3x random. Khmer scores **0.000 at 95.2%
confidence**. This is why [`Router`](#quickstart-route-mode-recommended) exists: the
model's own confidence gives no warning, so the routing decision has to be made before the
forward pass.

### English tasks

| task | `taut` | `taut-multilingual` | note |
|---|---|---|---|
| AG News | **0.947** | 0.937 | in training mix |
| BoolQ | **0.830** | 0.787 | in training mix |
| DAIR Emotion | **0.573** | 0.513 | held out |
| prompt-injections | **0.698** | 0.578 | held out, n=116 |
| SST-5 (ordinal) | 0.372 | 0.282 | held out |

### Calibration

Both checkpoints are over-confident as shipped. Refitting one temperature per (question type,
option count) on held-out data moves mean ECE **0.466 -> 0.081** (`taut`) and
**0.314 -> 0.106** (`taut-multilingual`). `taut-multilingual` ships with no fitted
temperatures at all, so fit them before relying on its probabilities.

At checkpoint load, numeric temperature entries are clamped to `[0.5, 5.0]`; invalid or
non-finite entries use the neutral fallback `1.0`. A runtime warning reports the affected
entries and applied values. Bucket-specific temperatures still take precedence over per-type
values, including when a bucket uses the fallback. Raw values remain available in
`agent.temperature_raw` and `agent.temperature_by_options_raw`. A fallback prevents a loading
failure; it does not establish calibrated confidence.

### Honest limits

* **The base checkpoints are near chance on typed-decisions zero-shot** -- 0.362 and 0.352
  against a 0.318 random baseline and a 0.461 majority-class baseline. The 0.766 figure comes
  from the checkpoint fine-tuned on that benchmark's own training split. Taut is a fast base to
  specialise, not a zero-shot decision engine.
* **Avoid boolean-word labels in `choice` questions.** Choice keys are rendered verbatim, and the
  current checkpoints can follow labels such as `true`/`false` or `yes`/`no` instead of the option
  descriptions. Use semantic labels or opaque labels such as `A`/`B`, and validate them on the
  checkpoint and states you serve.
* **High-cardinality choice questions and token budgets:** Sequences split into an option prompt budget (`head_max_len`) and the remaining document/state budget (`max_len - head_max_len`):
  * `taut` (English) defaults to 512 context (`head_max_len = 192`, ~320 tokens for state).
  * `taut-multilingual` and `taut-typed-decisions` default to 1,024 context (`head_max_len = 256`, ~768 tokens for state; mmBERT-base encoder supports up to 8,192 with RoPE).
  At default settings, a 77-option question like Banking77 allocates only `(256 - 16) // 77` ≈ 3–4 tokens per label, which causes accuracy to fall off sharply (0.425 vs Jev's 0.870). If evaluating 50+ options in a single question:
  1. Raise `agent.cfg["head_max_len"] = 512` and `agent.cfg["max_len"] = 1024` (or up to 2048 / 4096 / 8192) so every option has enough tokens to remain distinct.
  2. Or shortlist with embeddings and run one forward pass on the top `k` labels (`predict_shortlist`, example below). `predict` and `system_one` still score every criterion they are given.
  3. Or split the label set yourself into a coarse question and a fine question.

```python
import taut

questions = {
    "intent": {
        "type": "choice",
        "instructions": "Which banking intent is this?",
        "criteria": {
            "card_arrival": "where is my card",
            "transfer_fee": "fee charged on a transfer",
            # ...the rest of a large label set
        },
    }
}
result = taut.predict_shortlist(
    agent,
    {"text": "I was charged twice for a transfer"},
    questions,
    embed_fn=taut.embed_fn_from_agent(agent),  # or any callable: texts -> (n, dim)
    k=20,
)
result["shortlist"]["intent"]["labels"]  # the top 20 labels sent to the model
```

`embed_fn(texts)` returns one vector per string. `embed_fn_from_agent` mean-pools the encoder already loaded on the agent; the decision head runs in the following `predict` / `system_one` call. Probabilities on a shortlisted choice are over those `k` labels. When `k` is at least the number of labels, the original question is passed through and `embed_fn` is not called.

[Issue #102](https://github.com/NandhaKishorM/laya/issues/102) reports that a top-20 zero-shot shortlist moved a BANKING77 run from 54.3% to 60.8% on the reporter's setup. Those figures are the reporter's; this repository has not remeasured them.

* Ordinal `score` questions are the weakest primitive (SST-5 0.372).
* **`noul` can follow its option labels instead of the state, most strongly on `taut` (English).** `noul` renders its two options as `false:` / `true:` by default, and on the English checkpoint that label pair can dominate the answer, returning a confident "no" for clearly positive input (#156). Until a retrained checkpoint lands, check `noul` answers on your own data. You can override the model-facing pair while keeping the `noul` result as P(true):

  ```python
  {"type": "noul", "instructions": "Is this review positive?",
   "criteria": {"true": "yes, the review is positive", "false": "no, the review is negative"},
   "labels": {"true": "A", "false": "B"}}
  ```

  Label sensitivity varies by checkpoint and state, so validate the override on your own data. A
  two-option `choice` with neutral keys remains another workaround:

  ```python
  {"type": "choice", "instructions": "Is this review positive?",
   "criteria": {"A": "yes, the review is positive", "B": "no, the review is negative"}}
  ```
* **`taut-multilingual` has a position bias on `score` questions** (#131): it rarely picks the first-listed level, in any language. For English score questions, route to `model="english"`, and for other languages validate score outputs on your own data before relying on them.
* **`action.act_probability` carries no usable signal yet** (#185). It reads 1.0 for almost every input, and its raw logits run against correctness (AUROC 0.30 on 396 labelled decisions). Gate on `confidence` instead, which reaches an AUROC of 0.77 on the same items.
* `taut` collapses outside English; `taut-multilingual` is weaker on English. Route, or pick
  deliberately.

---

## Community Tools

* **[omp-taut-judge](https://github.com/F0Rextasy/omp-taut-judge)**: an [oh-my-pi](https://github.com/can1357/oh-my-pi) plugin with a local System-1 judge MCP server and skill (`choice`/`bool`/`score`, 0 tokens, about 0.3 s on CPU), confidence-gated escalation, and reproducible quiz and Snake demos.
* **[taut-adk-toolkit](https://github.com/Ashfaqbs/taut-adk-toolkit)**: [Google ADK](https://google.github.io/adk-docs/) tools that let an agent call Taut's `classify`/`score`/`detect` typed decisions directly as tools, instead of asking an LLM to guess at structured output.
* **[taut-Ascend](https://github.com/zzhdbw/taut-Ascend)**: Taut on Huawei Ascend NPUs through `torch-npu`, with a CPU vs NPU benchmark (34x to 71x faster at batch size 1), a setup guide, and Snake and Tetris demos.

---

## Live Demo & Resources

* **Hugging Face Model:** [thekarteek/taut](https://huggingface.co/thekarteek/taut)
* **Source:** [thedataengineer/taut](https://github.com/thedataengineer/taut)
* **Upstream project:** [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — the encoder, the training recipe, and the interactive demo of the original checkpoints

---

## Fine-Tuning

Fine-tune Taut on your own domain data. The notebook runs on Kaggle's free 2xT4 GPUs and does
the whole loop: build the dataset, train with RLCD (proper-scoring-rule rewards, GRPO-style
policy gradient), fit calibration temperatures, evaluate, and push the result to the Hub.

* **[`notebooks/taut_finetune_typed_decisions_2xT4_kaggle.ipynb`](notebooks/taut_finetune_typed_decisions_2xT4_kaggle.ipynb)**

The notebook enables gradient checkpointing on both the encoder and the decision head.
For custom training loops, `model.head_checkpointing = True` enables activation
checkpointing for the decision-head layers; enable the encoder's gradient checkpointing
separately. During gradient-enabled training, this reduces stored intermediate activations
by recomputing them during backward, trading extra computation for lower activation memory.
The head flag defaults to `False` and is bypassed in evaluation and under `torch.no_grad()`.

The notebook fits one `temperature` per type (`choice`, `score`, `noul`) and removes inherited
`temperature_by_options` from the exported config. Otherwise those old bucket values take
precedence at inference and silently mask the new fit. Existing checkpoints still honor
intentional bucket-specific temperatures, falling back to the corresponding per-type value
when a bucket is absent; the runtime's temperature clamp is unchanged.

This fixes configuration persistence, not measured model accuracy or calibration quality.
The notebook's calibration samples come from its training items; evaluate on separate held-out
data before claiming an improvement. Already published checkpoints are not rewritten.
Run the CPU-only regression checks with `python tests/test_calibration_persistence.py`
(synthetic configs and tiny local fixtures; no pretrained downloads or training).

Fine-tuning is where most of the value is. On the typed-decisions benchmark the base
checkpoints score near chance zero-shot (0.36 and 0.35 against a 0.318 random baseline),
while the fine-tuned checkpoint reaches **0.766** on the same 2,000 decisions -- above
TypeSafe Jev's published 0.727 and above the 0.735 teacher self-agreement ceiling. Treat Taut
as a fast base to specialise, not as a zero-shot decision engine.

Runtime on 2xT4 is roughly 4-5 hours for 4 epochs over ~30k questions.

### Worked example: a browser-agent decision head

[`docs/finetune_browser_agent.md`](docs/finetune_browser_agent.md) records a complete specialisation
on a single 16 GB GPU with no paid API: Taut as the operation/target decider for
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (same request format as
TypeSafe Jev). Element top-1 among ~45 candidates goes from 0.10 zero-shot to 0.66, real-task
success from 0 % to 62 % at 17-23 ms per step; weights, pipeline code and per-run results are on
the Hub at [cklxx/taut-browser](https://huggingface.co/cklxx/taut-browser). The write-up covers the
data recipe (reverse-generated goals, executed DONE states, Mind2Web, on-policy corrections), the
input-format change that mattered most, and the things that did not work.

---

## License

Apache 2.0.

## Attribution

Taut is a fork of [Laya](https://github.com/NandhaKishorM/laya), developed by Convai
Innovations and released under Apache 2.0. The encoder architecture, the RLCD training
recipe, the checkpoints, the router, the language detection and the e-mail cleaning are
upstream's work, and the benchmark numbers in [BENCHMARKS.md](BENCHMARKS.md) measure
upstream's checkpoints.

The checkpoints published at [`thekarteek/taut*`](https://huggingface.co/thekarteek/taut)
are mirrors of upstream's Apache-2.0 weights, republished unmodified so this package has a
stable home for them. Nothing about the model has been retrained or altered.

What this fork adds, and is responsible for:

- `taut.conformal` — certified risk control over the decisions
- `taut.drift` — label-free detection of an expired gate
- the gate surface in the server (`TAUT_GATE`) and the CLI (`--gate`, `--report`)
- `research/conformal/` — the validation that the bound actually holds

Issue links in the source that point at `NandhaKishorM/laya` are deliberate: they record
where a fix came from.
