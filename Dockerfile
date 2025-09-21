# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Application configuration defaults (can be overridden at runtime)
ENV DATABASE_USER=postgres \
    DATABASE_PASSWORD=IHatePostgres \
    DATABASE_PORT=5432 \
    PORT=8000 \
    KEYS_MODE=true \
    RATE_LIMIT_REQUESTS=100 \
    RATE_LIMIT_WINDOW=60 \
    KEY_LIFE_SPAN=90 \
    DEBUG=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install -e .

# Copy application code
COPY src/ ./src/
COPY run.py ./

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "views_challenge.main:app", "--host", "0.0.0.0", "--port", "8000"]