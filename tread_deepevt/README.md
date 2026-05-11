# TREAD Phase 2: DeepEVT

This directory (`tread_deepevt`) contains the implementation for the second phase of the TREAD project: **DeepEVT (Deep Extreme Value Theory) for Conditional Tail Risk Modeling**. 

It builds upon the driving events extracted in Phase 1 (`tread_highd`) to model the heavy-tailed distribution of safety-critical risks conditioned on contextual driving factors (e.g., prefix trajectories, semantic features). The models are trained specifically per event type (e.g., car-following, cut-in).

## Core Functionalities

DeepEVT aims to learn the contextual distribution of tail risks (like severe TTC, THW, DRAC aggregated as a `risk_score`) and predicts threshold exceedance probabilities, expected shortfalls (ES), and conditional tail quantiles (e.g., $q_{90}$, $q_{95}$, $q_{99}$).

- **Data Reconstruction**: Rebuilds fixed-length physical windows from `events.csv` to construct prefix trajectories and context features without information leakage from risk labels.
- **DeepEVT Architecture**: Employs a sequence encoder (e.g., GRU) for trajectory encoding, fused with semantic context features, predicting Generalized Pareto Distribution (GPD) parameters $(\xi, \beta)$ and tail exceedance probabilities.
- **Baselines**: Includes baseline implementations like Global POT-GPD and Quantile-Only Neural Baseline for comparative evaluation.

## Directory Structure

### `src/` - Core Library
- `window_rebuild.py`: Reconstructs fixed-length trajectory states from HighD raw data.
- `features.py`: Context feature extraction for different events ensuring no risk leakage.
- `data.py`: Utilities for building dataset tensors (`dataset.npz`), handling splits and normalization.
- `model.py`: DeepEVT PyTorch architecture.
- `losses.py`: Objective functions including Pinball loss, Exceedance Binary Cross-Entropy, and GPD Negative Log-Likelihood.
- `train.py` & `evaluate.py`: Training loop and evaluation framework for test set calibration and visualization.
- `inference.py` & `metrics.py`: Inference pipeline and extreme value metrics (e.g., ECE, Tail Quantile Error).
- `baselines.py`: Implementation of global baselines and quantile regression baselines.

### `scripts/` - Execution Pipeline
- `configs/`: YAML configurations for training and evaluation.
- `01_build_deepevt_dataset.py`: Builds the DeepEVT dataset from `events.csv`.
- `02_train_deepevt.py`: Trains the DeepEVT model.
- `03_evaluate_deepevt.py`: Generates evaluation metrics and calibration plots.
- `04_export_tail_conditions.py`: Exports predictions to act as targets for downstream diffusion generation.

## Usage

All scripts should be executed from the root of the `TREAD` directory or inside `tread_deepevt/scripts`, using the provided YAML configuration files:

```bash
# 1. Build the DeepEVT Dataset
python tread_deepevt/scripts/01_build_deepevt_dataset.py --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 2. Train the DeepEVT model
python tread_deepevt/scripts/02_train_deepevt.py --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 3. Evaluate the DeepEVT model
python tread_deepevt/scripts/03_evaluate_deepevt.py --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 4. Export tail risk conditions for diffusion models
python tread_deepevt/scripts/04_export_tail_conditions.py --config tread_deepevt/scripts/configs/deepevt_following.yaml
```
