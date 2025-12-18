.PHONY: help setup setup-gpu clean test test-coverage format lint typecheck train eval docker-build docker-run

help:
	@echo "RLFactorSynth - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install dependencies (CPU)"
	@echo "  make setup-gpu      - Install dependencies (GPU)"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format code with black and isort"
	@echo "  make lint           - Lint code with flake8"
	@echo "  make typecheck      - Type check with mypy"
	@echo "  make test           - Run tests"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo ""
	@echo "Training:"
	@echo "  make train-distill  - Train distillation model"
	@echo "  make train-ppo      - Train PPO model"
	@echo "  make eval           - Run benchmark evaluation"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Remove build artifacts"

setup:
	pip install -e .
	pre-commit install

setup-gpu:
	pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
	pre-commit install

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/ --max-line-length=100 --extend-ignore=E203,W503

typecheck:
	mypy src/rlfactorsynth

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ -v --cov=src/rlfactorsynth --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

train-distill:
	python scripts/train_distill.py --config configs/train/distill.yaml

train-ppo:
	python scripts/train_ppo.py --config configs/train/ppo.yaml

eval:
	python scripts/eval_benchmarks.py --config configs/eval/benchmarks.yaml

profile:
	python scripts/profile_batching.py --batch_sizes 1,16,64,256

docker-build:
	docker build -t rlfactorsynth:latest .

docker-run:
	docker run --gpus all -it -v $$(pwd):/workspace rlfactorsynth:latest

docker-run-cpu:
	docker run -it -v $$(pwd):/workspace rlfactorsynth:latest
