# STAGE - 1
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest pytest-mock

COPY . .

RUN pytest tests/ -v

# STAGE 2

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "scripts.api.app:app", "--host", "0.0.0.0", "--port", "8000"]