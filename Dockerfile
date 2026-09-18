# ITCS355 Lab 1 — training image
#
# Base image pinned by digest, not tag. `python:3.11-slim` moves; the digest below is
# the exact manifest resolved on 2026-09-13 with:
#   docker pull python:3.11-slim && docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim
FROM python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first so this layer caches independently of your source.
COPY requirements.txt ./
# --require-hashes turns a silently-substituted package into a build failure.
RUN pip install --require-hashes --prefix=/install -r requirements.txt


FROM python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS runtime

# Non-root. A training container has no reason to run as root, and graders check.
RUN useradd --create-home --uid 10001 runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local
WORKDIR /app
# WORKDIR creates /app as root. mlflow writes a local mlruns/mlartifacts dir relative to
# the CWD when no artifact root is set, so /app itself — not just what gets COPYed into
# it — has to be writable by the non-root user, or that write fails at runtime instead
# of at build time.
RUN chown runner:runner /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner cloudlayer/ ./cloudlayer/
COPY --chown=runner:runner scripts/ ./scripts/
# Pointer + remote config only — never the data itself, and never .dvc/cache.
# entrypoint.sh uses these to `dvc pull` the real bytes at container start, on the one
# path (managed training) that has no local data and no bind mount to read it from.
COPY --chown=runner:runner .dvc/config ./.dvc/config
COPY --chown=runner:runner data/raw.dvc ./data/raw.dvc
COPY --chown=runner:runner --chmod=755 entrypoint.sh ./entrypoint.sh

USER runner

# Credentials NEVER enter an image layer. They arrive at runtime from SECRET_STORE_PATH
# or from the platform's identity. If you find yourself adding an ARG for a key, stop.
ENTRYPOINT ["./entrypoint.sh"]
CMD ["--n-estimators", "200", "--max-depth", "8"]
