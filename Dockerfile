FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
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
# RUN pytest -V

COPY . .

RUN pytest tests/ -v

EXPOSE 8000

CMD ["uvicorn", "scripts.api.app:app", "--host", "0.0.0.0", "--port", "8000"]