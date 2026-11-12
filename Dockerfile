# Stage 1: Build zero-allocation Go 1.24 real-time feature scorer
FROM golang:1.24-bookworm AS go-builder
WORKDIR /build
COPY services/realtime-scorer-go/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /outguard-scorer ./cmd/outguard-scorer

# Stage 2: Python 3.12 ML training, verification & FastAPI control plane
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY --from=go-builder /outguard-scorer /usr/local/bin/outguard-scorer

RUN pip install --no-cache-dir -e .

EXPOSE 8080 8090
CMD ["python3", "-m", "outguard.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
