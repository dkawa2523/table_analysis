FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.clearml/uv-cache

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      git \
      libgomp1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      procps \
      tini && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "clearml>=1.15.0" \
    "clearml-agent>=1.9.3" \
    "uv>=0.5.0"

WORKDIR /root
VOLUME ["/root/.clearml"]
ENTRYPOINT ["tini", "--"]
CMD ["clearml-agent", "daemon", "--queue", "default"]
