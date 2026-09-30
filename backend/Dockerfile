FROM python:3.11-slim

# nbainjuries -> tabula-py needs a JRE to read the NBA's PDF injury reports.
# lightgbm's compiled extension needs libgomp (OpenMP) at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY config /app/config
COPY tests /app/tests

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["python", "-m", "backend.ingest.runner"]
