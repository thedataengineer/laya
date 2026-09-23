# Docker quickstart

Run the SDK without installing Python or PyTorch on your host. For the CPU
quickstart, allow 8 GB of RAM and 10 GB of free disk, with Docker Engine or
Docker Desktop and Compose v2 or newer.

From the repository root:

```bash
docker compose run --build --rm taut
```

This builds the checkout, runs the [sample request](../examples/docker/request.json)
on CPU and prints JSON covering `choice`, `score` and `noul`. The first request
downloads the selected public Hugging Face checkpoint; no account is needed.
Allow several minutes for its first download.
Weights stay in a named volume. Subsequent runs use `docker compose run --rm taut`.

Predictions and confidence still need evaluation on your workload. See the
[benchmark limits](../BENCHMARKS.md).

## NVIDIA GPU / CUDA

Install a compatible NVIDIA driver and configure Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The GPU image uses PyTorch CUDA 12.8 wheels. Check your GPU's compute capability
and driver against [PyTorch's supported builds](https://pytorch.org/get-started/locally/);
older cards may require a different build. Allow additional disk space for CUDA
layers. VRAM needs depend on the checkpoint, batch size and input length.

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --build --rm taut
```

The override selects GPU `0` and defaults to `TAUT_DEVICE=cuda`. Set
`TAUT_GPU_ID` to another host index or UUID. That GPU appears as device `0`
inside the container. Check access without downloading weights:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --rm taut python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").cpu())'
```

The sample rejects unavailable CUDA before loading a checkpoint. Taut can still
fall back to CPU after a memory or inference error, so inspect its warnings.
Rebuild when switching between CPU and CUDA configurations.

This uses [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).
Windows requires Docker Desktop's supported WSL2 GPU setup. Apple MPS,
AMD/ROCm and Intel GPU containers are outside this quickstart; use CPU unless
you configure and validate another backend.

## Configuration

Set Compose variables in your shell, a local `.env` file, or the service's
`environment` block. Don't commit secrets in `.env`. Runtime variables also
work with `docker run -e`; Compose-only settings are identified below.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TAUT_DEVICE` | `cpu` / `cuda` | Device selected by the base / GPU configuration |
| `TAUT_MODEL` | `auto` | Router alias: `auto`, `english`, `multilingual`, `typed-decisions` |
| `TAUT_MODEL_PATH` | unset | Compatible checkpoint path inside the container |
| `TAUT_REQUEST_FILE` | bundled request | JSON request path inside the container |
| `OMP_NUM_THREADS` | `4` | CPU threads; keep within available cores |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset | Optional Hugging Face credential |
| `TAUT_API_KEY` / `TAUT_API_KEY_FILE` | unset | **`taut-serve` only:** require `Authorization: Bearer <key>` |
| `TAUT_PORT` | `8000` | **`taut-serve` only:** container port, and the host port published for it |
| `HF_HUB_OFFLINE` | `0` | `1` uses only cached checkpoints |
| `HF_HOME` | `/home/taut/.cache/huggingface` | Cache path; see mount requirement below |
| `TAUT_CACHE_VOLUME` | project model cache | **Compose only:** named cache volume |
| `TAUT_GPU_ID` | `0` | **Compose only:** NVIDIA device index or UUID |
| `TAUT_TORCH_INDEX` | `cpu` / `cu128` | **Compose build:** PyTorch wheel index |

Compose forwards the runtime variables except `HF_HOME`, which stays aligned
with its fixed cache mount. If overriding `HF_HOME` in `docker run` or your own
Compose file, provide a matching mount writable by UID 10001. Direct Docker
builds select PyTorch with `--build-arg TORCH_INDEX=cu128`; runtime `-e` cannot
change the installed wheel.

```bash
TAUT_MODEL=english OMP_NUM_THREADS=2 docker compose run --build --rm taut

docker build -t taut:local .
docker run --rm -e TAUT_MODEL=english -e OMP_NUM_THREADS=2 \
  -v taut-model-cache:/home/taut/.cache/huggingface taut:local
```

For your own request:

```bash
docker compose run --rm --volume "$PWD/request.json:/inputs/request.json:ro" \
  --env TAUT_REQUEST_FILE=/inputs/request.json taut
```

For a commented configuration with request, checkpoint and secret-file mounts,
see [`compose.example.yml`](../compose.example.yml):

```bash
docker compose -f compose.yaml -f compose.example.yml run --build --rm taut
```

Add `-f compose.cuda.yaml` before `run` for NVIDIA GPUs. The example is an
override of `compose.yaml`, so cache and image settings stay in one place.

## Secret files

`HF_TOKEN_FILE` reads a mounted UTF-8 file at startup, trims surrounding
whitespace and takes precedence over `HF_TOKEN`. Unreadable, empty or invalid
files stop startup without printing their contents. The file must be readable
by UID 10001. `_FILE` applies only to supported secrets, not every setting.

With `HF_TOKEN_PATH` pointing to an existing host file outside the checkout:

```bash
docker compose run --rm --volume "$HF_TOKEN_PATH:/run/secrets/hf_token:ro" \
  --env HF_TOKEN_FILE=/run/secrets/hf_token taut
```

Docker secrets or Kubernetes Secret volumes can supply the same file. Values
are loaded into the process environment at startup; restart after changing a
file. Never use tokens as build arguments or bake them into images. Public
checkpoints need no token.

## Fine-tuned checkpoints

This image runs inference. Training scripts are tracked in
[#4](https://github.com/NandhaKishorM/laya/issues/4) and
[#26](https://github.com/NandhaKishorM/laya/issues/26); training commands can follow
once that interface is available.

Point `TAUT_CHECKPOINT_PATH` to an absolute host directory containing
`rl_agent_config.json`, `model.safetensors` and matching tokenizer files:

```bash
docker compose run --rm --volume "$TAUT_CHECKPOINT_PATH:/models/custom" \
  --env TAUT_MODEL_PATH=/models/custom taut
```

Use a working copy writable by UID 10001 because the loader may update tokenizer
configuration. A LoRA adapter alone is not a complete checkpoint. Leave
`TAUT_MODEL=auto` when setting `TAUT_MODEL_PATH`; an explicit alias and local path
are mutually exclusive. The local-path response comes from the Agent and has no
Router `routing` metadata. These settings also work with the CUDA override.
Evaluate fine-tuned checkpoints on held-out examples before relying on them.

## Development and cleanup

Open a Python prompt with `docker compose run --rm taut python`. To run the
existing routing/criteria checks and secret-file tests against your checkout
without downloading weights:

```bash
docker compose run --rm --volume "$PWD:/workspace:ro" --workdir /workspace taut \
  sh -ec 'python tests/test_router.py; python tests/test_criteria.py; python tests/test_docker_entrypoint.py'
```

Rebuild with `--build` after changing source or the bundled example. The image
runs as UID/GID 10001. New named volumes inherit the image cache directory's
ownership; host directories must be writable by that UID. Keep model caches
writable for tokenizer compatibility updates.

`--rm` removes completed containers. `docker compose down` retains the cache.
To **delete downloaded weights**, run `docker compose down --volumes` using the
same Compose files and `TAUT_CACHE_VOLUME` setting. The next request downloads
them again; don't remove a cache shared with another project.

## HTTP serving

The image ships `taut-serve`, so the same build that runs the one-shot quickstart can
serve the Jev-compatible API. `compose.http.yaml` adds it as a second service and leaves
`taut` alone:

```bash
docker compose -f compose.yaml -f compose.http.yaml up --build taut-serve
curl -s localhost:8000/health
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \
  --data @examples/docker/request.json
```

For NVIDIA, add the CUDA override. It repeats the build args and the device reservation
for `taut-serve`, because a Compose override replaces a service's `build` block whole
rather than merging it:

```bash
docker compose -f compose.yaml -f compose.http.yaml -f compose.cuda.yaml up --build taut-serve
```

`up` keeps the service running in the foreground; `-d` detaches. Weights go to the same
named `model-cache` volume as the quickstart, so serving after a quickstart run starts
with the checkpoints already on disk. Stop with `docker compose ... down`, using the same
Compose files.

### Server configuration

These apply to the `taut-serve` service only.

| variable | default | effect |
|---|---|---|
| `TAUT_HOST` | `0.0.0.0` | bind address inside the container |
| `TAUT_PORT` | `8000` | container port, and the host port published for it |
| `TAUT_PRELOAD` | `0` | `1` builds every checkpoint at startup instead of on first request |
| `TAUT_MODELS` | (all) | comma list to preload: `english,multilingual,typed-decisions` |
| `TAUT_THREADS` | `OMP_NUM_THREADS` | caps torch intra-op threads; keep at or below physical cores |
| `TAUT_AUTO_TASK` | `0` | `1` lets the router reach `typed-decisions` automatically |
| `TAUT_LOG_LEVEL` | `info` | uvicorn log level |
| `TAUT_API_KEY` | (none) | when set, requires `Authorization: Bearer <key>` |

`TAUT_PRELOAD` defaults to `0` here rather than the package default of `1`, because
preloading makes the first boot download all three checkpoints. Set it to `1` for a
long-running deployment so the first request does not pay for the build.

`TAUT_PORT` sets both the published host port and the port the server binds, so the two
cannot drift. Change one place to move the service:

```bash
TAUT_PORT=9000 docker compose -f compose.yaml -f compose.http.yaml up --build taut-serve
```

### Bearer token from a file

`TAUT_API_KEY_FILE` is read once at startup, moved into `TAUT_API_KEY`, and the `_FILE`
variable is removed before the server execs. Prefer this to putting the key in the
environment:

```bash
docker compose -f compose.yaml -f compose.http.yaml run --rm \
  --volume "$PWD/taut_api_key:/run/secrets/taut_api_key:ro" \
  -e TAUT_API_KEY_FILE=/run/secrets/taut_api_key \
  --service-ports taut-serve
```

