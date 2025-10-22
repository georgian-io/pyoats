# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Set environment variables for GPU support
ENV TF_FORCE_GPU_ALLOW_GROWTH=true \
    CUDA_HOME=/usr/local/cuda/ \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install Python 3.12 and build dependencies in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and README (needed by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies with uv (frozen lockfile for reproducibility)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy application code
COPY oats ./oats

# Install package in editable mode
RUN uv pip install --no-cache -e .

# Set default command
CMD ["uv", "run", "python"]