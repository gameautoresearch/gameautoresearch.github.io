# The researcher's box: one image for every game and task variant.
# CPU-only and offline at run time. CPU PyTorch and NumPy are installed so the
# researcher can import and dry-run their own modules; the game engines and GPU
# are not here, and code trains for real only on the workers the host broker starts.
FROM python:3.12.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux asciinema ripgrep git less jq nano vim-tiny procps tree \
    && rm -rf /var/lib/apt/lists/*
# The session's task builder sets this to the torch version of the game's training workers.
ARG TORCH=2.15.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple "torch==${TORCH}" numpy
RUN useradd --create-home --uid 10001 --shell /bin/bash researcher
# AGENT_INSTALL
COPY workspace /workspace
COPY ar /ar
# Rules, simulator source and instructions are root-owned and read-only. The
# researcher owns training/, configs/ and NOTES.md; only the declared files in
# them ever leave the box, and the host validates those on every request.
RUN chown -R root:root /workspace /ar \
    && find /workspace /ar -type d -exec chmod 755 {} + \
    && find /workspace /ar -type f -exec chmod 444 {} + \
    && chown -R researcher:researcher /workspace/training /workspace/configs /workspace/NOTES.md \
    && find /workspace/training /workspace/configs /workspace/NOTES.md -type f -exec chmod 644 {} + \
    && mkdir -p /ar/requests /ar/replies \
    && chown researcher:researcher /ar/requests \
    && printf '#!/bin/sh\nexec python3 -I /ar/ar.py "$@"\n' > /usr/local/bin/ar && chmod 755 /usr/local/bin/ar \
    && printf 'set -o ignoreeof\n' >> /etc/bash.bashrc
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
WORKDIR /workspace
