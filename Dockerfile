# ==============================================================================
# Competitive Intelligence Monitoring Platform - Production Dockerfile
# ==============================================================================

FROM python:3.11-slim

# System environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install minimal OS dependencies required for postgres and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy package metadata and source code required for installation
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[api,knowledge,agentic,acquisition]"

# Copy database migrations, configuration, and entrypoint
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini
COPY scripts/ /app/scripts/

# Set permissions
RUN chmod +x /app/scripts/docker-entrypoint.sh && \
    chown -R appuser:appgroup /app

# Switch to non-privileged user
USER appuser

# Expose default HTTP port
EXPOSE 8000

# Liveness probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

CMD ["uvicorn", "competitive_intelligence.application.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
