.PHONY: install test lint format clean

install:
	pip install -r requirements.txt
	pip install black ruff pytest

test:
	python3 -m pytest tests/

lint:
	python3 -m ruff check .

format:
	python3 -m black .
	python3 -m ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete