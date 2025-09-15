.PHONY: format lint

# Format code with black and isort
format:
	black .
	isort .

# Check formatting without modifying files
lint:
	black --check .
	isort --check-only .
