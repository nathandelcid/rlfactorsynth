# RLFactorSynth Quick Start Guide

This guide will get you up and running with RLFactorSynth in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- (Optional) CUDA-capable GPU for training

## Installation

### Option 1: Local Setup (Recommended for Development)

```bash
# Extract the repository
tar -xzf rlfactorsynth.tar.gz
cd rlfactorsynth

# Install dependencies (CPU-only)
pip install -e .

# Or with GPU support
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import rlfactorsynth; print('Success!')"
```

### Option 2: Docker Setup (Recommended for Reproducibility)

```bash
# Build Docker image
docker build -t rlfactorsynth:latest .

# Run container with GPU
docker run --gpus all -it -v $(pwd):/workspace rlfactorsynth:latest

# Or CPU-only
docker run -it -v $(pwd):/workspace rlfactorsynth:latest
```

## Basic Usage

### 1. Test the Installation

Run the unit tests to verify everything works:

```bash
pytest tests/test_unitary_distance.py -v
```

### 2. Explore with Jupyter

Open the example notebook:

```bash
jupyter notebook notebooks/01_unitary_basics.ipynb
```

### 3. Train a Model (Quick Test)

Train for a small number of steps to test the pipeline:

```bash
python scripts/train_ppo.py \
  train.total_steps=1000 \
  env.max_steps=50 \
  logging.log_interval=10
```

This will:
- Create a synthesis environment
- Initialize a Transformer policy
- Train using PPO for 1000 steps
- Save checkpoints to `outputs/`

### 4. Synthesize Unitaries

Once you have a trained model (or use a random initialization for testing):

```bash
python scripts/synthesize.py \
  --checkpoint outputs/rlfactorsynth_default/*/checkpoints/final.pt \
  --n_qubits 1 \
  --n_random 5
```

This will synthesize 5 random single-qubit unitaries and report:
- Success rate
- Average T-count
- Synthesis time

## Configuration

All experiments are configured via YAML files in `configs/`. Override any setting from the command line:

```bash
# Change learning rate
python scripts/train_ppo.py train.learning_rate=1e-4

# Use different model
python scripts/train_ppo.py model=ffnn

# Adjust environment
python scripts/train_ppo.py env.max_steps=200 env.n_qubits=2
```

## Common Commands

```bash
# Format code
make format

# Run tests
make test

# Train with default settings
make train-ppo

# Clean build artifacts
make clean
```

## Project Structure

```
rlfactorsynth/
├── configs/          # Configuration files
├── src/              # Source code
│   └── rlfactorsynth/
│       ├── quantum/  # Quantum primitives
│       ├── envs/     # RL environments
│       ├── models/   # Neural networks
│       ├── rl/       # Training algorithms
│       └── utils/    # Utilities
├── scripts/          # Executable scripts
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks
└── README.md         # Full documentation
```

## Next Steps

1. **Read the full README**: `README.md` contains detailed documentation
2. **Review PROJECT_SUMMARY**: `PROJECT_SUMMARY.md` explains what's implemented
3. **Explore the code**: Start with `src/rlfactorsynth/envs/synthesis_env.py`
4. **Customize**: Modify configs in `configs/` for your use case
5. **Extend**: Add meta-actions, expert policies, or new architectures

## Troubleshooting

### Import Error
```bash
# Make sure you're in the right directory
cd rlfactorsynth
pip install -e .
```

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train_ppo.py train.batch_size=32
```

### Slow Training
```bash
# Enable mixed precision
python scripts/train_ppo.py train.use_amp=true
```

## Getting Help

- Check `README.md` for detailed documentation
- Review `PROJECT_SUMMARY.md` for implementation details
- Look at example notebooks in `notebooks/`
- Examine test files in `tests/` for usage examples

## Performance Expectations

On a modern GPU (A100/3090):
- Training: ~1000 steps/minute
- Inference: ~100 unitaries/second (batched)
- Memory: ~4GB for default configuration

On CPU:
- Training: ~100 steps/minute
- Inference: ~10 unitaries/second
- Memory: ~2GB

## What's Next?

After getting familiar with the basics:

1. **Implement meta-actions**: Add precomputed gate sequences
2. **Integrate expert policy**: Connect to Trasyn or gridsynth
3. **Add benchmarks**: Load real quantum circuits
4. **Optimize performance**: Enable batching and GPU acceleration
5. **Run ablations**: Compare architectures and encoding methods

Happy synthesizing! 🚀
