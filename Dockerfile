FROM python:3.11-slim

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies — own layer so it's cached until pyproject.toml/uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

# Copy application code and model artifact
COPY src/ src/
COPY api/ api/
COPY config.yaml .
COPY models/ models/

# Cloud Run injects PORT at runtime; default to 8080 for local runs
ENV PORT=8080

# sh -c required for $PORT expansion (exec form doesn't expand env vars)
CMD ["sh", "-c", "uv run uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]