FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# XGBoost requires the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN python -m pip install --no-cache-dir -r requirements-api.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser artifacts/model.joblib ./artifacts/model.joblib
COPY --chown=appuser:appuser artifacts/model_metadata.json ./artifacts/model_metadata.json

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]