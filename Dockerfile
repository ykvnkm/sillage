# The prediction service, CPU only.
#
#   docker build -t sillage .
#   docker run --rm -p 8000:8000 sillage
#
# Three stages: the model, downloaded at the revision pinned in deploy/hub.json and
# checked file by file against its sha256; the dependencies, installed by uv from
# the lock file; and a clean runtime that receives only those two results. No
# compiler, no uv, no source tree, no dev dependencies in the final image.
#
# The same file builds locally and in CI, and no build depends on whatever happens
# to sit in a developer's models/ directory.

FROM python:3.11-slim-bookworm AS model

WORKDIR /fetch
COPY deploy/hub.json scripts/fetch_model.py ./
RUN python fetch_model.py --manifest hub.json --out /model


FROM python:3.11-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

# Bytecode compiled at build time, so the first request does not pay for it.
# Copy rather than hardlink: the cache mount and the venv are different filesystems.
# Use the image's interpreter: the runtime stage has the same one and no other.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, project second: the dependency layer is rebuilt only when the
# lock file changes, not on every edit to the source.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.11-slim-bookworm

# System libraries RDKit's wheels link against (found with `ldd` over every RDKit
# .so in the image). The drawing module needs the X ones even when it only writes
# SVG; without them the service fails at import, before serving a single request.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libxrender1 libxext6 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 sillage

COPY --from=build --chown=sillage:sillage /app/.venv /app/.venv
COPY --from=model --chown=sillage:sillage /model /app/model

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SILLAGE_MODEL_DIR=/app/model \
    PORT=8000

USER sillage
WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"]

# PORT is read at start, not baked in, so a hosting platform can choose it.
CMD ["sh", "-c", "exec uvicorn sillage.interfaces.api.app:app --host 0.0.0.0 --port \"$PORT\""]
