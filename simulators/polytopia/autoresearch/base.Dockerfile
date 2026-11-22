# The agent's box (and the grader's training image): CUDA 12.4 devel, Python 3.12 venv, the Rust engine wheel, the
# CUDA engine built for sm_89 (L40S), torch + CUDA JAX for the GPU-resident trainer, the harness under /ar.
# Heavy layers only (CUDA, Python, torch, JAX, both engines). build_context.py writes task/environment/Dockerfile =
# this file + the COPY tail below, and modal_app.py builds this file alone with Python-level layers on top.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl pkg-config ca-certificates git \
        tmux less vim-tiny procps jq && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12 && uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/bin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
RUN uv pip install 'numpy>=2.1' 'maturin>=1.7,<2' zstandard pytest
RUN uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install 'jax[cuda12]==0.11.2'
COPY repo/engine-rs /build/engine-rs
RUN cd /build/engine-rs && maturin build --release -o /build/wheels && uv pip install /build/wheels/*.whl
COPY repo/engine-cu /workspace/engine-cu
RUN make -C /workspace/engine-cu -j8 cpu-variants \
    && make -C /workspace/engine-cu -j8 cuda-variants \
        ARCHFLAGS="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89" \
    && mkdir -p /ar && cp -r /workspace/engine-cu/build /ar/engine-build && rm -rf /workspace/engine-cu/build \
    && ls -la /ar/engine-build
