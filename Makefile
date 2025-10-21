.PHONY: help install install-dev test lint format clean build docs

help:
	@echo "Available commands:"
	@echo "  make install      - Install package dependencies"
	@echo "  make install-dev  - Install package with dev dependencies"
	@echo "  make test         - Run tests with pytest"
	@echo "  make lint         - Run ruff linter on source code"
	@echo "  make format       - Format code with black and fix with ruff"
	@echo "  make clean        - Remove build artifacts and cache files"
	@echo "  make build        - Build distribution packages"
	@echo "  make docs         - Build documentation with Sphinx"

install:
	uv sync

install-dev:
	uv sync --all-extras

test:
	uv run pytest -v --tb=short

lint:
	uv run ruff check oats tests

lint-fix:
	uv run ruff check --fix oats tests

format:
	uv run black oats tests
	uv run ruff check --fix oats tests

format-check:
	uv run black --check oats tests
	uv run ruff check oats tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete

build: clean
	uv build

docs:
	cd docs && uv run make html

all: format lint test
