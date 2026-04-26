FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

FROM python:3.12-slim-bookworm

WORKDIR /app

# Add user without root access
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    TORCH_HOME=/app/models

USER appuser

RUN python -c "from torchvision.models import get_model; get_model('resnet18', weights='DEFAULT')"

COPY --chown=appuser:appuser . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4123"]