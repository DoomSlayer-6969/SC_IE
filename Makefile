# Makefile
# Project: Secure Communication and Image Encryption
# Legendre-based Brain Emotional Learning Chaos Synchronization

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV_PYTHON) -m pip

.PHONY: help venv install install-dev dirs run test lint format check clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-14s %s\n", $$1, $$2}'

venv: ## Create the project virtual environment
	$(PYTHON) -m venv $(VENV)


install: venv ## Install the runtime dependencies from requirements.txt
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: venv ## Install optional development tools
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

dirs: ## Create data and results directories
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/input_images', 'data/output_images', 'results/graphs', 'results/metrics', 'results/logs']]"

run: dirs ## Run the main project entry point
	$(VENV_PYTHON) main.py

test: ## Run tests (if a tests/ directory exists)
	$(VENV_PYTHON) -m pytest -q

lint: ## Check Python code with Ruff
	$(VENV_PYTHON) -m ruff check .

format: ## Format Python code with Ruff
	$(VENV_PYTHON) -m ruff format .

check: lint test ## Run linting and tests

clean: ## Remove generated outputs and Python cache files
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find results/graphs results/metrics results/logs -type f -delete 2>/dev/null || true
