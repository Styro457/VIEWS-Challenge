# VIEWS Challenge Makefile
# Assumes virtual environment is already activated

# Variables
PYTHON := python3
PIP := pip
UVICORN := uvicorn
RUFF := ruff
PYTEST := pytest
DOCKER := docker

# Application settings
APP_MODULE := views_challenge.main:app
HOST := 127.0.0.1
PORT := 8000

# Default target
.PHONY: help
help:
	@echo "VIEWS Challenge - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "🚀 Development:"
	@echo "  make install     - Install project dependencies"
	@echo "  make run         - Start development server"
	@echo "  make dev         - Start development server with reload"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code with ruff"
	@echo "  make check       - Run lint + tests"
	@echo ""

# Installation
.PHONY: install
install:
	@echo "📦 Installing dependencies..."
	$(PIP) install -e .[dev]
	@echo "✅ Installation complete!"

# Development server
.PHONY: run
run:
	@echo "🚀 Starting VIEWS Challenge server..."
	$(UVICORN) $(APP_MODULE) --host $(HOST) --port $(PORT)

.PHONY: dev
dev:
	@echo "🔄 Starting development server with auto-reload..."
	$(UVICORN) $(APP_MODULE) --host $(HOST) --port $(PORT) --reload

# Testing and code quality
.PHONY: test
test:
	@echo "🧪 Running tests..."
	$(PYTEST) tests/ -v

.PHONY: lint
lint:
	@echo "🔍 Running code linting..."
	$(RUFF) check src/views_challenge tests

.PHONY: format
format:
	@echo "🎨 Formatting code..."
	$(RUFF) format src/views_challenge tests
	$(RUFF) check --fix src/views_challenge tests

.PHONY: check
check: lint test
	@echo "✅ All checks passed!"
