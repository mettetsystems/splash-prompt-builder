PYTHON ?= python3
VENV_PYTHON := .venv/bin/python
ARGS ?=

.PHONY: setup start stop build run dev-backend dev-frontend test latency audit package-macos

setup:
	$(PYTHON) scripts/setup.py --dev $(ARGS)

start:
	$(PYTHON) scripts/start.py $(ARGS)

stop:
	$(PYTHON) scripts/stop.py $(ARGS)

build:
	npm --prefix frontend run build

run: build
	$(VENV_PYTHON) -m backend.app

dev-backend:
	$(VENV_PYTHON) -m uvicorn backend.app:app --host 127.0.0.1 --port 43827 --reload --ws-max-size 100000

dev-frontend:
	npm --prefix frontend run dev

test:
	$(VENV_PYTHON) -m unittest discover -s tests -v
	npm --prefix frontend test

latency:
	$(VENV_PYTHON) tests/test_latency.py

audit:
	npm --prefix frontend audit

package-macos:
	$(PYTHON) scripts/package_macos.py
