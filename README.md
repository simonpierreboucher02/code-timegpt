<div align="center">

# TimeGPT — Transformer Foundation Model for Time Series Forecasting

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Model-27M%20Params-6366F1?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Architecture-Transformer-F59E0B?style=for-the-badge"/>

<br/>

<img src="https://img.shields.io/badge/Task-Time%20Series%20Forecasting-0EA5E9?style=flat-square"/>
<img src="https://img.shields.io/badge/Inference-Zero--Shot-10B981?style=flat-square"/>
<img src="https://img.shields.io/badge/Output-Multi--Quantile-8B5CF6?style=flat-square"/>
<img src="https://img.shields.io/badge/Data-S%26P%20500-EF4444?style=flat-square"/>
<img src="https://img.shields.io/badge/Hardware-MPS%20%7C%20CUDA%20%7C%20CPU-64748B?style=flat-square"/>

<br/><br/>

*A PyTorch implementation of TimeGPT — a Transformer-based foundation model for **zero-shot**, **multi-horizon**, **probabilistic** time series forecasting with randomized horizon conditioning.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Metrics](#key-metrics)
- [Architecture](#architecture)
- [Key Innovations](#key-innovations)
- [Mathematical Foundation](#mathematical-foundation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Advanced Usage](#advanced-usage)
- [Performance & Hardware](#performance--hardware)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Overview

**TimeGPT** is a Transformer-based foundation model for time series forecasting that introduces three key concepts:

1. **Randomized Horizon Conditioning** — Instead of predicting fixed horizons, the model learns a continuous transition function `g(h_t, φ(h))`, enabling extrapolation beyond the maximum training horizon.
2. **Self-Supervised Masked Reconstruction** — BERT-style span masking on 15% of context tokens, providing robust pretraining signal and acting as regularization.
3. **Local Z-Score Normalization** — Per-sample normalization ensures scale invariance across heterogeneous domains.

The model is trained entirely on S&P 500 log return data (2010–2024) and can generalize to unseen series in a **zero-shot** setting without fine-tuning.

---

## Key Metrics

<div align="center">

### Model Architecture

| Metric | Value |
|--------|-------|
| ![](https://img.shields.io/badge/Total%20Parameters-27%2C121%2C670-6366F1?style=flat-square) | 27,121,670 |
| ![](https://img.shields.io/badge/Model%20Dimension-512-3B82F6?style=flat-square) | 512 |
| ![](https://img.shields.io/badge/Transformer%20Layers-8-8B5CF6?style=flat-square) | 8 |
| ![](https://img.shields.io/badge/Attention%20Heads-8-EC4899?style=flat-square) | 8 |
| ![](https://img.shields.io/badge/Head%20Dimension-64-F59E0B?style=flat-square) | 64 |
| ![](https://img.shields.io/badge/FFN%20Dimension-2048-10B981?style=flat-square) | 2048 |
| ![](https://img.shields.io/badge/Max%20Context%20Length-512-0EA5E9?style=flat-square) | 512 tokens |
| ![](https://img.shields.io/badge/Max%20Forecast%20Horizon-96-EF4444?style=flat-square) | 96 steps |

### Validation Results (1 Epoch — AAPL)

| Metric | Value | Badge |
|--------|-------|-------|
| Validation Loss | 0.2741 | ![](https://img.shields.io/badge/Val%20Loss-0.2741-22C55E?style=flat-square) |
| MAE | 0.00804 | ![](https://img.shields.io/badge/MAE-0.00804-3B82F6?style=flat-square) |
| RMSE | 0.01020 | ![](https://img.shields.io/badge/RMSE-0.01020-8B5CF6?style=flat-square) |
| Coverage q₀.₁₀ | 10.4% | ![](https://img.shields.io/badge/Coverage%20q0.10-10.4%25%20%E2%9C%93-22C55E?style=flat-square) |
| Coverage q₀.₉₀ | 93.8% | ![](https://img.shields.io/badge/Coverage%20q0.90-93.8%25%20%E2%9C%93-22C55E?style=flat-square) |
| Train Batches (1 epoch) | 3,073 | ![](https://img.shields.io/badge/Batches-3%2C073-F59E0B?style=flat-square) |

### Training Configuration

| Parameter | Value | Badge |
|-----------|-------|-------|
| Batch Size | 8 | ![](https://img.shields.io/badge/Batch%20Size-8-64748B?style=flat-square) |
| Learning Rate | 1e-4 | ![](https://img.shields.io/badge/LR-1e--4-0EA5E9?style=flat-square) |
| Optimizer | AdamW | ![](https://img.shields.io/badge/Optimizer-AdamW-EC4899?style=flat-square) |
| Scheduler | Cosine Annealing | ![](https://img.shields.io/badge/Scheduler-CosineAnnealingWarmRestarts-F59E0B?style=flat-square) |
| Gradient Clipping | 1.0 | ![](https://img.shields.io/badge/Grad%20Clip-1.0-10B981?style=flat-square) |
| Weight Decay | 1e-4 | ![](https://img.shields.io/badge/Weight%20Decay-1e--4-8B5CF6?style=flat-square) |
| Mask Ratio | 15% | ![](https://img.shields.io/badge/Mask%20Ratio-15%25-EF4444?style=flat-square) |
| λ (Reconstruction) | 0.5 | ![](https://img.shields.io/badge/%CE%BB%20Mask-0.5-6366F1?style=flat-square) |

### Data Pipeline

| Parameter | Value | Badge |
|-----------|-------|-------|
| Training Range | 2010–2024 | ![](https://img.shields.io/badge/Data-2010--2024-3B82F6?style=flat-square) |
| Train / Val Split | 80 / 20 | ![](https://img.shields.io/badge/Split-80%2F20-22C55E?style=flat-square) |
| Context Length (range) | 96–512 | ![](https://img.shields.io/badge/Context-96--512-8B5CF6?style=flat-square) |
| Forecast Horizon (range) | 24–96 | ![](https://img.shields.io/badge/Horizon-24--96-F59E0B?style=flat-square) |
| Time Features | 8 (calendar + Fourier) | ![](https://img.shields.io/badge/Time%20Features-8-0EA5E9?style=flat-square) |
| Quantile Levels | 5 | ![](https://img.shields.io/badge/Quantiles-5-EC4899?style=flat-square) |

</div>

---

## Architecture

```
TimeGPT (27,121,670 parameters)
│
├── TimeSeriesEmbedding
│   ├── Value Embedding      Linear(1 → 512)
│   ├── Time Embedding       Linear(8 → 512)       ← 8 calendar + Fourier features
│   ├── Positional Embedding Embedding(512, 512)    ← Learned, not sinusoidal
│   └── Series Embedding     Embedding(N, 512)      ← Optional, per-series identity
│
├── TransformerEncoder (×8 layers)
│   ├── CausalMultiHeadAttention   (8 heads × 64-dim each = 512)
│   │   ├── W_q, W_k, W_v          Linear(512, 512)
│   │   ├── W_o                    Linear(512, 512)
│   │   └── Causal mask            Lower-triangular (M_{ab} = -∞ if b > a)
│   ├── LayerNorm (pre & post attention)
│   └── FeedForwardNetwork         Linear(512→2048) → GELU → Linear(2048→512)
│
├── MultiQuantileForecastHead
│   ├── Horizon Embedding    Embedding(96, 512)
│   ├── MLP                  Linear(512→1024) → GELU → Linear(1024→1024) → GELU → Linear(1024→5)
│   └── Monotonicity         Cumulative softplus over quantile dimension
│
└── MaskedReconstructionHead
    └── Linear(512 → 1)      ← Only used during masked pretraining batches
```

### Token Embedding (Eq. 7–8)

Each time step `ℓ` is encoded as:

```
z_ℓ = W_y · ỹ_ℓ + W_t · φ(t_ℓ) + p_ℓ + s_i

where:
  ỹ_ℓ      : normalized value at step ℓ
  φ(t_ℓ)   : 8-dimensional time features (calendar + Fourier)
  p_ℓ      : learned positional embedding
  s_i      : optional series identity embedding
```

### Causal Self-Attention (Eq. 9–11)

```
Attention(Q, K, V) = softmax((QKᵀ / √d_h) ⊙ M) · V

M_{ab} = { 0    if b ≤ a
         { -∞   if b > a    ← causal mask (no future leakage)
```

---

## Key Innovations

### 1. Randomized Horizon Conditioning

**Problem**: Classic models learn a lookup table — one set of weights per horizon index. This prevents extrapolation beyond the training horizon range.

**Solution**: During 50% of training batches, horizons are randomly subsampled from `[0, H_max]`. This forces the model to learn a continuous **transition function**:

```
ŷ_{t+h} = g(h_t, φ(h))

where:
  h_t   : context summary from the last Transformer position
  φ(h)  : learned horizon embedding (Embedding layer, not sinusoidal)
  g     : 3-layer MLP with GELU activations
```

**Four sampling strategies:**

| Mode | Description | Best for |
|------|-------------|----------|
| `all` | All horizons [0 → H] (baseline) | Evaluation |
| `random_subset` | Same subset per batch | Batch efficiency |
| `random_individual` | Different horizons per sample | Maximum diversity |
| `mixed` | Alternates randomly **(recommended)** | Training |

### 2. Self-Supervised Masked Reconstruction

Inspired by BERT's masked language modeling:

- **Mask ratio**: 15% of context tokens
- **Masking strategy**: Contiguous spans (max 10 steps) — respects temporal locality
- **Applied to**: 30% of training batches
- **Objective**: MSE at masked positions only

```
L_mask = (1/|M|) Σ_{ℓ∈M} (ỹ_{i,ℓ} - ŷ_{i,ℓ})²
```

**Benefit**: Learns robust temporal representations, reduces overfitting, acts as implicit regularization.

### 3. Local Z-Score Normalization

Per-sample, per-context normalization:

```
μ = mean(y_ctx)
σ = std(y_ctx) + ε      ← ε = 1e-8 for numerical stability

ỹ = (y - μ) / σ        ← normalize input
ŷ_denorm = ŷ · σ + μ   ← denormalize output
```

**Benefit**: Handles heterogeneous series with different scales and magnitudes without domain-specific preprocessing.

---

## Mathematical Foundation

### Total Loss (Eq. 19)

```
L_total(θ) = E[L_forecast(θ) + λ · L_mask(θ)]

where λ = 0.5
```

### Quantile (Pinball) Loss (Eq. 16–17)

```
L_forecast = (1 / H·Q) Σ_h Σ_τ ρ_τ(y_{t+h} - q̂^(τ)(h))

ρ_τ(u) = { τ · u        if u ≥ 0
          { (τ - 1) · u  if u < 0
```

### Monotonicity Constraint (Eq. 15)

Ensures no quantile crossing via cumulative softplus:

```
q̂^(τ₁) = α                                (unconstrained)
q̂^(τₖ) = q̂^(τₖ₋₁) + softplus(δₖ)       (k > 1)

Guarantee: q̂^(τ₁) ≤ q̂^(τ₂) ≤ ... ≤ q̂^(τ_Q)
```

### Horizon Embedding (Eq. 14)

```
q̂^(τ)(h) = g_θ(h_t + e^(h))

where:
  h_t    : last hidden state of Transformer encoder
  e^(h)  : learned horizon embedding vector
  g_θ    : MLP(512 → 1024 → 1024 → Q)
```

---

## Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/simonpierreboucher02/code-timegpt.git
cd code-timegpt

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt** installs:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Core deep learning framework |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `yfinance` | ≥ 0.2.28 | Yahoo Finance data download |
| `tqdm` | ≥ 4.65.0 | Progress bars |
| `matplotlib` | ≥ 3.7.0 | Visualization |

---

## Training

### Quick Test (1 epoch, 10 tickers — ~11 min on Apple M-series)

```bash
python train.py --num_tickers 10 --num_epochs 1 --batch_size 8
```

### Full Training (100 epochs, 50 tickers)

```bash
python train.py --num_tickers 50 --num_epochs 100 --batch_size 8
```

### Horizon Sampling Modes

```bash
# Mixed mode — recommended (alternates strategies per batch)
python train.py --horizon_sampling_mode mixed

# Random subset — same subset for all samples in batch (efficient)
python train.py --horizon_sampling_mode random_subset

# Random individual — different horizons per sample (maximum diversity)
python train.py --horizon_sampling_mode random_individual

# Disabled — always predict all horizons (baseline)
python train.py --no_horizon_sampling
```

### Resume Training

```bash
python train.py --resume checkpoints/latest_checkpoint.pt
```

### Custom Run

```bash
python train.py \
  --num_tickers 50 \
  --num_epochs 100 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --checkpoint_dir my_experiment \
  --horizon_sampling_mode mixed
```

### Console Output During Training

```
============================================================
STARTING TIMEGPT TRAINING
============================================================
  Configuration:
   - Epochs:           1
   - Batch Size:       8
   - Learning Rate:    0.0001
   - Device:           mps
   - Model Params:     27,121,670
   - Context Length:   (96, 512)
   - Forecast Horizon: (24, 96)
   - Quantiles:        [0.1, 0.25, 0.5, 0.75, 0.9]

  Horizon Sampling (Transition Function Learning):
   - Mode:             mixed
   - Probability:      50.0%
   - Min Horizons:     8
============================================================

Epoch 0: 100%|████████| 3073/3073 [10:45<00:00, 5.26it/s, loss=0.27]

============================================================
EPOCH 1/1 COMPLETED
============================================================
  Training Metrics:
   - Train Loss:           0.3845
   - Forecast Loss:        0.2736
   - Reconstruction Loss:  0.2219

  Validation Metrics:
   - Val Loss:             0.2741
   - Val Forecast Loss:    0.2741
   - Val Quantile Loss:    0.1716

  NEW BEST MODEL!
============================================================
```

---

## Inference

### Run on AAPL (requires trained checkpoint)

```bash
python test_inference.py
```

**What it does:**
1. Loads `checkpoints/best_checkpoint.pt`
2. Downloads AAPL data from Yahoo Finance (2020–2024)
3. Generates 48-step ahead forecast with 5 quantile levels
4. Prints summary statistics (MAE, RMSE, coverage per quantile)
5. Saves dual-panel plot to `forecast_AAPL.png`

### Inference on Any Ticker

Edit `test_inference.py`:

```python
ticker = 'TSLA'           # Change to any valid Yahoo Finance symbol
context_length = 256      # Number of historical days as context
forecast_horizon = 48     # Number of days to forecast
```

Then run:

```bash
python test_inference.py
```

### Programmatic Inference

```python
import torch
from model import TimeGPT

# Load model
checkpoint = torch.load('checkpoints/best_checkpoint.pt', map_location='cpu')
config = checkpoint['config']

model = TimeGPT(
    d_model=config['d_model'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    d_ff=config['d_ff'],
    max_seq_len=config['max_seq_len'],
    max_horizon=config['max_horizon'],
    time_feature_dim=8,
    num_series=50,
    quantiles=config['quantiles'],
    dropout=config['dropout'],
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    outputs = model(
        y_ctx=y_ctx_tensor,          # (1, context_length)
        time_features_ctx=time_feat, # (1, context_length, 8)
        horizons=torch.arange(48).unsqueeze(0),
        series_ids=torch.LongTensor([0])
    )

quantile_forecasts = outputs['quantile_forecasts']  # (1, 48, 5)
```

---

## Configuration

All hyperparameters live in `train.py`. Full reference:

```python
config = {
    # ── Data Parameters ───────────────────────────────────────────
    'num_tickers': 10,                    # S&P 500 stocks to use
    'start_date': '2010-01-01',
    'end_date': '2024-12-01',
    'context_length_range': (96, 512),    # Randomized per batch
    'forecast_horizon_range': (24, 96),   # Randomized per batch
    'train_split': 0.8,

    # ── Model Architecture ─────────────────────────────────────────
    'd_model': 512,
    'num_layers': 8,
    'num_heads': 8,
    'd_ff': 2048,
    'max_seq_len': 512,
    'max_horizon': 96,
    'dropout': 0.1,
    'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],

    # ── Training Hyperparameters ───────────────────────────────────
    'num_epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,

    # ── Loss Function ──────────────────────────────────────────────
    'lambda_mask': 0.5,                   # Reconstruction loss weight λ
    'use_masked_loss': True,
    'mask_probability': 0.3,              # % of batches with masking
    'mask_ratio': 0.15,                   # % of tokens masked per sequence
    'max_mask_span': 10,                  # Max contiguous span length

    # ── Horizon Sampling ───────────────────────────────────────────
    'use_horizon_sampling': True,
    'horizon_sampling_prob': 0.5,         # 50% of batches use random horizons
    'horizon_sampling_mode': 'mixed',     # 'all' | 'random_subset' | 'random_individual' | 'mixed'
    'min_horizons_sampled': 8,

    # ── LR Scheduler ──────────────────────────────────────────────
    'scheduler_t0': 10,
    'scheduler_tmult': 2,

    # ── Logging & Checkpointing ────────────────────────────────────
    'validate_every': 1,
    'save_every': 5,
    'checkpoint_dir': 'checkpoints',
}
```

---

## Evaluation Metrics

### Forecast Accuracy

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | `mean(|y_true - ŷ_median|)` | Mean Absolute Error on median forecast |
| **RMSE** | `sqrt(mean((y_true - ŷ_median)²))` | Root Mean Squared Error |

### Quantile Calibration

| Metric | Target | Description |
|--------|--------|-------------|
| **Coverage(q_τ)** | ≈ τ | Fraction of true values below predicted quantile |
| **50% Interval Width** | — | `q₀.₇₅ - q₀.₂₅` (average) |
| **80% Interval Width** | — | `q₀.₉₀ - q₀.₁₀` (average) |

A **well-calibrated** model achieves:
- `Coverage(q₀.₁₀) ≈ 10%`
- `Coverage(q₀.₅₀) ≈ 50%`
- `Coverage(q₀.₉₀) ≈ 90%`

---

## Advanced Usage

### Custom Dataset

To use your own time series data, subclass `SP500StockDataset` or create a new `Dataset`:

```python
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, series_dict, context_length_range, forecast_horizon_range):
        """
        series_dict: {series_id: np.array of values}
        """
        ...

    def __getitem__(self, idx):
        # Must return same keys as SP500StockDataset:
        return {
            'y_ctx': torch.FloatTensor(...),          # (L,)
            'y_fut': torch.FloatTensor(...),          # (H,)
            'time_features_ctx': torch.FloatTensor(...),  # (L, 8)
            'time_features_fut': torch.FloatTensor(...),  # (H, 8)
            'series_id': int,
            'mu': torch.FloatTensor([mu]),
            'sigma': torch.FloatTensor([sigma]),
            'L': int,
            'H': int
        }
```

### Multi-Domain Training

Combine datasets from different domains for broader generalization:

```python
from torch.utils.data import ConcatDataset

dataset_stocks = SP500StockDataset(...)
dataset_energy = EnergyDataset(...)      # Custom dataset
dataset_weather = WeatherDataset(...)    # Custom dataset

combined = ConcatDataset([dataset_stocks, dataset_energy, dataset_weather])
```

### Export to ONNX (Production Deployment)

```python
import torch.onnx

model.eval()
dummy_y_ctx = torch.randn(1, 256)
dummy_time = torch.randn(1, 256, 8)
dummy_horizons = torch.arange(48).unsqueeze(0)
dummy_series = torch.tensor([0])

torch.onnx.export(
    model,
    (dummy_y_ctx, dummy_time, dummy_horizons, dummy_series),
    "timegpt.onnx",
    input_names=['y_ctx', 'time_features', 'horizons', 'series_ids'],
    output_names=['quantile_forecasts'],
    dynamic_axes={
        'y_ctx': {1: 'context_length'},
        'horizons': {1: 'num_horizons'}
    }
)
```

### Mixed Precision Training (PyTorch 2.0+)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(y_ctx, time_features, horizons, series_ids)
    loss = criterion(outputs, y_fut)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation (Effective Batch Size = 32 with `--batch_size 8`)

```python
# Accumulate over 4 micro-batches:
accumulation_steps = 4
for step, batch in enumerate(train_loader):
    outputs = model(...)
    loss = criterion(...) / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Performance & Hardware

### Training Time Estimates

| Hardware | Tickers | Epochs | Time |
|----------|---------|--------|------|
| Apple M-series (MPS) | 10 | 1 | ~11 min |
| Apple M-series (MPS) | 10 | 100 | ~18 hours |
| Apple M-series (MPS) | 50 | 100 | ~90 hours |
| NVIDIA RTX 3090 (CUDA) | 10 | 1 | ~5 min |
| NVIDIA RTX 3090 (CUDA) | 50 | 100 | ~40 hours |

### Memory Requirements

| Batch Size | Approximate VRAM / RAM |
|-----------|------------------------|
| 4 | ~4 GB |
| 8 | ~7 GB |
| 16 | ~13 GB |
| 32 | ~24 GB |

### Device Priority (Auto-Detected)

```python
if torch.backends.mps.is_available():    # Apple Silicon GPU
    device = torch.device('mps')
elif torch.cuda.is_available():          # NVIDIA GPU
    device = torch.device('cuda')
else:                                    # CPU fallback
    device = torch.device('cpu')
```

---

## File Structure

```
code-timegpt/
│
├── model.py              # TimeGPT architecture (493 lines, 27M parameters)
│   ├── TimeSeriesEmbedding
│   ├── CausalMultiHeadAttention
│   ├── TransformerEncoderLayer
│   ├── MultiQuantileForecastHead
│   ├── MaskedReconstructionHead
│   └── TimeGPT (main class)
│
├── losses.py             # Loss functions (307 lines)
│   ├── PinballLoss           ← Multi-quantile pinball loss
│   ├── MaskedReconstructionLoss  ← MSE at masked positions
│   ├── TimeGPTLoss           ← Joint objective L_total = L_forecast + λ·L_mask
│   ├── create_contiguous_mask
│   └── quantile_score
│
├── data_loader.py        # Data pipeline (306 lines)
│   ├── SP500StockDataset     ← Yahoo Finance download + windowing
│   ├── get_sp500_tickers
│   └── collate_variable_length  ← Padding for variable-length batches
│
├── train.py              # Training loop (661 lines)
│   ├── create_random_horizons   ← 4 horizon sampling strategies
│   ├── TimeGPTTrainer
│   │   ├── train_epoch()
│   │   ├── validate()
│   │   ├── save_checkpoint()
│   │   └── load_checkpoint()
│   └── main()                   ← CLI entry point
│
├── test_inference.py     # Inference + visualization (364 lines)
│   ├── load_model_from_checkpoint
│   ├── forecast_single_series
│   ├── plot_forecast
│   └── print_forecast_summary
│
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

### Checkpoint Structure

```
checkpoints/
├── best_checkpoint.pt          # Best model (lowest val loss)
├── latest_checkpoint.pt        # Most recent epoch
├── checkpoint_epoch_5.pt       # Periodic saves (every 5 epochs)
├── checkpoint_epoch_10.pt
└── training_log.json           # Full metrics history (JSON)
```

Each `.pt` file contains:

```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'metrics': {
        'train_loss': float,
        'val_loss': float,
        'val_coverage_0.1': float,
        ...
    },
    'config': dict          # Full training config for reproducibility
}
```

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train.py --batch_size 4

# Reduce context length in train.py
'context_length_range': (96, 256)   # was (96, 512)
```

### Slow Training

```bash
# Fewer tickers
python train.py --num_tickers 10

# Smaller model (edit train.py)
'd_model': 256, 'num_layers': 4     # was 512, 8
```

### NaN Loss

- Check for infinite values in raw data (stock splits, delisting)
- Lower the learning rate: `--learning_rate 5e-5`
- Tighten gradient clipping (edit `train.py`): `'grad_clip': 0.5`

### pin_memory Warning on MPS

This is handled automatically — `pin_memory` is disabled when using MPS (Apple Silicon), as it is not yet supported by PyTorch.

### Load Checkpoint Manually

```python
import torch

ckpt = torch.load('checkpoints/best_checkpoint.pt')
print(f"Epoch: {ckpt['epoch']}")
print(f"Val Loss: {ckpt['metrics']['val_loss']:.4f}")
print(f"Config: {ckpt['config']}")
```

### Monitor Training Live

```bash
# Watch JSON log (appended after each epoch)
tail -f checkpoints/training_log.json
```

---

## Dependencies

```
torch>=2.0.0        PyTorch deep learning framework
numpy>=1.24.0       Numerical computation
pandas>=2.0.0       DataFrame and time series manipulation
yfinance>=0.2.28    Yahoo Finance market data API
tqdm>=4.65.0        Progress bars
matplotlib>=3.7.0   Visualization and plotting
```

**Python**: 3.8+

---

## References

- **TimeGPT** — Garza & Mergenthaler-Canseco (2023). *TimeGPT-1*
- **Attention Is All You Need** — Vaswani et al. (2017). *Transformers*
- **BERT** — Devlin et al. (2019). *Masked Language Modeling*
- **DeepAR** — Salinas et al. (2020). *Probabilistic Forecasting with Autoregressive RNNs*
- **Temporal Fusion Transformer** — Lim et al. (2021)
- **GPT-2/3** — Radford et al. (2018/2019/2020)

---

## Citation

If you use this code in your research or project, please cite:

```bibtex
@software{boucher2024timegpt,
  title     = {TimeGPT: A Transformer Foundation Model for Zero-Shot Multi-Horizon Probabilistic Time Series Forecasting},
  author    = {Boucher, Simon-Pierre},
  year      = {2024},
  url       = {https://github.com/simonpierreboucher02/code-timegpt},
  note      = {PyTorch implementation}
}
```

---

## Authors

<table>
<tr>
<td align="center">
<b>Simon-Pierre Boucher</b><br/>
Lead Author & Architect<br/>
<a href="mailto:spbou4@protonmail.com">spbou4@protonmail.com</a><br/>
<a href="https://www.spboucher.ai">www.spboucher.ai</a><br/>
<a href="https://github.com/simonpierreboucher02">@simonpierreboucher02</a>
</td>
<td align="center">
<b>Claude (Anthropic)</b><br/>
AI Pair Programmer<br/>
<a href="https://claude.ai">claude.ai</a><br/>
Model: Claude Sonnet 4.6<br/>
<a href="https://anthropic.com">anthropic.com</a>
</td>
</tr>
</table>

---

## License

```
MIT License

Copyright (c) 2024 Simon-Pierre Boucher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

**Built with Claude Code** &nbsp;|&nbsp; **Author: Simon-Pierre Boucher** &nbsp;|&nbsp; **2024** &nbsp;|&nbsp; **Version 1.0**

<img src="https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch"/>
<img src="https://img.shields.io/badge/Built%20with-Claude%20Code-6366F1?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>

</div>
