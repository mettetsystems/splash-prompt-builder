PYTHON ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup build run dev-backend dev-frontend test latency audit

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install -r backend/requirements-dev.txt
	npm --prefix frontend ci

build:
	npm --prefix frontend run build

run: build
	$(VENV_PYTHON) -m backend.app

dev-backend:
	$(VENV_PYTHON) -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload --ws-max-size 100000

dev-frontend:
	npm --prefix frontend run dev

test:
	$(VENV_PYTHON) -m unittest discover -s tests -v
	npm --prefix frontend test

latency:
	$(VENV_PYTHON) tests/test_latency.py

audit:
	npm --prefix frontend audit
