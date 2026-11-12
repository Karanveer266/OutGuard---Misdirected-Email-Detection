.PHONY: install run verify test test-go bench-go triage serve-py serve-go all

install:
	pip install -e ".[dev]"

run:
	python3 -m outguard.cli run-all

verify:
	python3 -m outguard.cli verify

test:
	pytest -v

test-go:
	cd services/realtime-scorer-go && go test -v ./...

bench-go:
	cd services/realtime-scorer-go && go test -v -bench=. -benchmem ./...

triage:
	python3 -m outguard.cli triage-incident

serve-py:
	python3 -m outguard.cli serve --host 0.0.0.0 --port 8080

serve-go:
	cd services/realtime-scorer-go && go run ./cmd/outguard-scorer --bundle ../../artifacts/model_bundle.json --addr :8090

all: run verify test test-go
