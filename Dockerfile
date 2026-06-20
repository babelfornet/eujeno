# Eujeno node image (linux/arm64) — runs `eujeno serve` for a slice of layers.
# The model weights are NOT included: they are mounted at runtime from the host's
# HF cache (-v ~/.cache/huggingface:/root/.cache/huggingface), so all nodes
# share a single download.
FROM python:3.11-slim

WORKDIR /app

# Metadata + package only: minimal build context (see .dockerignore).
COPY pyproject.toml ./
COPY eujeno ./eujeno

# torch CPU-only: there is no GPU in the Docker VM (Mac). Installed from PyTorch's
# CPU index BEFORE the package, so `pip install -e .` does not pull the ~5GB CUDA wheels.
RUN pip install --no-cache-dir torch>=2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e .

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Entrypoint = eujeno CLI; the arguments (serve --coordinator ... --stages ...)
# are passed via `docker run ... eujeno-node serve ...`.
ENTRYPOINT ["eujeno"]
