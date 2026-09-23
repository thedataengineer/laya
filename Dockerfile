FROM python:3.11-slim-bookworm AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU by default; the CUDA Compose override selects cu128.
ARG TORCH_INDEX=cpu
RUN pip install torch --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

WORKDIR /src
COPY pyproject.toml setup.py README.md LICENSE ./
COPY taut/ ./taut/
# The `serve` extra puts `taut-serve` (POST /v1/systemone, GET /health) in the image, so
# the same image can run a one-shot request or serve the Jev-compatible API. It adds
# fastapi and uvicorn only; torch was installed above.
RUN pip install ".[serve]" && pip check

FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Taut Docker quickstart" \
      org.opencontainers.image.source="https://github.com/NandhaKishorM/laya" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    TAUT_DEVICE=cpu \
    HF_HOME=/home/taut/.cache/huggingface

RUN groupadd --gid 10001 taut \
    && useradd --uid 10001 --gid taut --create-home taut \
    && mkdir -p /home/taut/.cache/huggingface \
    && chown -R taut:taut /home/taut/.cache

COPY --from=build /opt/venv /opt/venv
COPY LICENSE /usr/share/doc/taut/LICENSE
COPY examples/docker/ /opt/taut/examples/
COPY docker/entrypoint.py /opt/taut/entrypoint.py
USER taut
WORKDIR /home/taut

ENTRYPOINT ["python", "/opt/taut/entrypoint.py"]
CMD ["python", "/opt/taut/examples/quickstart.py"]
