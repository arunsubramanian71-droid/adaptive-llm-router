.PHONY: install test lint verify-dry verify-live

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts
	mypy src

# Free — exercises the full pipeline with a fake completion, no API key needed.
verify-dry:
	python scripts/verify_stage0.py

# Spends a small amount of real money — makes a couple of real Anthropic calls.
verify-live:
	python scripts/verify_stage0.py --live --model claude-haiku-4-5 --num-prompts 2
