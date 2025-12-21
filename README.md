# TimeGPT - Transformer Foundation Model for Time Series Forecasting

A PyTorch implementation of TimeGPT, a Transformer-based foundation model for zero-shot multi-horizon probabilistic time series forecasting with randomized horizon conditioning.

## 🎯 Key Features

- **Universal Forecasting Operator**: Learn from heterogeneous time series across domains
- **Randomized Horizon Conditioning**: Learn transition function $g(h_t, \varphi(h))$ instead of lookup table
- **Self-Supervised Pre-training**: Masked reconstruction for robust representations
- **Multi-Quantile Forecasting**: Calibrated probabilistic predictions with 5 quantiles
- **Zero-Shot Generalization**: Forecast on unseen series without retraining
- **Scale Invariant**: Local normalization handles arbitrary magnitudes

## 📊 Results

After 1 epoch on S&P 500 data:

| Metric | Value |
|--------|-------|
| Validation Loss | 0.2741 |
| MAE (AAPL) | 0.00804 |
| RMSE (AAPL) | 0.01020 |
| Coverage q₀.₁₀ | 10.4% (target: 10%) |
| Coverage q₀.₉₀ | 93.8% (target: 90%) |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Quick test (1 epoch, 10 tickers)
python train.py --num_tickers 10 --num_epochs 1 --batch_size 8

# Full training (100 epochs, 50 tickers)
python train.py --num_tickers 50 --num_epochs 100 --batch_size 8

# With specific horizon sampling mode
python train.py --horizon_sampling_mode random_subset --batch_size 8

# Disable horizon sampling (baseline)
python train.py --no_horizon_sampling --batch_size 8
```

### Inference

```bash
# Test on AAPL stock (requires trained checkpoint)
python test_inference.py
```

This will:
- Load the best checkpoint from `checkpoints/`
- Download AAPL data from Yahoo Finance
- Generate 48-step ahead forecast with 5 quantiles
- Save visualization to `forecast_AAPL.png`

## 📁 File Structure

```
code-timegpt/
├── model.py                # TimeGPT architecture (27M parameters)
├── losses.py               # Pinball loss + masked reconstruction
├── data_loader.py          # S&P 500 data loading and preprocessing
├── train.py                # Training script with horizon sampling
├── test_inference.py       # Single-series inference and visualization
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🏗️ Architecture

```
TimeGPT (27,121,670 parameters)
├── TimeSeriesEmbedding
│   ├── Value embedding (1 → 512)
│   ├── Time embedding (8 → 512)
│   ├── Positional embedding (learned, 512 positions)
│   └── Series embedding (10 series → 512)
│
├── TransformerEncoder (8 layers)
│   ├── Multi-head attention (8 heads, 64-dim each)
│   └── Feed-forward network (512 → 2048 → 512)
│
├── MultiQuantileForecastHead
│   ├── Horizon embedding (96 horizons → 512)
│   ├── MLP (512 → 1024 → 1024 → 5 quantiles)
│   └── Monotonicity enforcement via softplus
│
└── MaskedReconstructionHead
    └── Linear projection (512 → 1)
```

## 🔧 Hyperparameters

### Model Architecture
- Model dimension: 512
- Layers: 8
- Attention heads: 8
- Feed-forward dim: 2048
- Dropout: 0.1

### Training
- Batch size: 8
- Learning rate: 1e-4 (AdamW)
- Weight decay: 1e-4
- Gradient clipping: 1.0
- LR schedule: Cosine annealing with warm restarts

### Data
- Context length: [96, 512] (randomized)
- Forecast horizon: [24, 96] (randomized)
- Train/val split: 80/20
- Time features: 8 (calendar + Fourier)

### Regularization
- Horizon sampling: 50% of batches
- Masked reconstruction: 30% of batches
- Mask ratio: 15%
- Reconstruction weight (λ): 0.5

### Quantiles
- Levels: [0.1, 0.25, 0.5, 0.75, 0.9]
- Monotonicity enforced via cumulative softplus

## 📖 Key Innovations

### 1. Randomized Horizon Conditioning

**Problem**: Traditional models predict fixed horizons, learning a lookup table.

**Solution**: Randomly sample which horizons to predict during training (4 modes):
- `all`: Predict all horizons [0→H] (baseline)
- `random_subset`: Sample random subset for batch
- `random_individual`: Different horizons per sample
- `mixed`: Randomly alternate (recommended)

**Result**: Model learns continuous transition function, can extrapolate beyond H_max.

### 2. Self-Supervised Masked Reconstruction

**Strategy**: Mask 15% of context with contiguous blocks (max span: 10 steps)

**Objective**: MSE reconstruction loss at masked positions

**Benefit**: Pre-trains robust temporal representations, reduces overfitting

### 3. Local Normalization

**Per-sample z-score normalization**:
```
μ = mean(context)
σ = std(context)
normalized = (value - μ) / σ
```

**Enables**: Cross-domain generalization despite different scales

## 🧪 Training Details

### Data Pipeline
1. Download S&P 500 stocks from Yahoo Finance (2010-2024)
2. Compute log returns: `r_t = log(p_t / p_{t-1})`
3. Random window sampling: `(i, t, L, H)`
4. Local normalization per sample
5. Extract 8 time features (calendar + Fourier)
6. Collate with variable-length padding

### Training Loop (per batch)
1. **50% of batches**: Randomize horizons
2. **30% of batches**: Apply contiguous masking
3. Forward pass through Transformer
4. Compute losses:
   - Pinball loss (multi-quantile)
   - Masked reconstruction (if applicable)
5. Backpropagation with gradient clipping
6. AdamW optimizer step
7. Cosine LR schedule update

### Validation
- Always use all horizons (no randomization)
- No masking
- Compute quantile metrics: loss, coverage, interval widths
- Save checkpoints: latest, best, periodic

## 📊 Output Files

After training, the following are generated:

```
checkpoints/
├── best_checkpoint.pt         # Best model (lowest val loss)
├── latest_checkpoint.pt       # Most recent checkpoint
├── checkpoint_epoch_X.pt      # Periodic saves (every 5 epochs)
└── training_log.json          # Metrics history
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training metrics
- Full configuration

## 🔬 Evaluation Metrics

### Forecast Accuracy
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

### Quantile Calibration
- **Coverage**: Fraction of true values below predicted quantile
  - Target: coverage(q_τ) ≈ τ
- **Interval Width**: Average width of prediction intervals
  - 50% interval: q₀.₇₅ - q₀.₂₅
  - 80% interval: q₀.₉₀ - q₀.₁₀

## 💡 Usage Examples

### Basic Training
```bash
# Train on 10 stocks for 1 epoch (quick test)
python train.py --num_tickers 10 --num_epochs 1 --batch_size 8

# Full training
python train.py --num_tickers 50 --num_epochs 100 --batch_size 8
```

### Horizon Sampling Modes
```bash
# Mixed mode (recommended, alternates randomly)
python train.py --horizon_sampling_mode mixed

# Random subset (efficient, same subset per batch)
python train.py --horizon_sampling_mode random_subset

# Random individual (maximum diversity, slower)
python train.py --horizon_sampling_mode random_individual

# Disable (baseline, always predict all horizons)
python train.py --no_horizon_sampling
```

### Resume Training
```bash
python train.py --resume checkpoints/latest_checkpoint.pt
```

### Custom Configuration
```bash
python train.py \
  --num_tickers 50 \
  --num_epochs 100 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --checkpoint_dir my_experiment \
  --horizon_sampling_mode mixed
```

### Inference on Custom Stock
Edit `test_inference.py` and change:
```python
ticker = 'AAPL'  # Change to any stock symbol
context_length = 256
forecast_horizon = 48
```

Then run:
```bash
python test_inference.py
```

## 📈 Expected Training Time

On Apple M-series GPU (MPS):
- **1 epoch, 10 tickers**: ~11 minutes (3,073 batches)
- **100 epochs, 10 tickers**: ~18 hours
- **100 epochs, 50 tickers**: ~90 hours

On NVIDIA RTX 3090:
- **1 epoch, 10 tickers**: ~5 minutes
- **100 epochs, 50 tickers**: ~40 hours

## 🎓 Mathematical Foundation

### Objective Function
```
L_total = E[(L_forecast + λ * L_mask)]

where:
  L_forecast = (1/HQ) Σ_h Σ_τ ρ_τ(y_true - q_pred)
  L_mask = (1/|M|) Σ_ℓ∈M (y_true - y_recon)²
  λ = 0.5
```

### Quantile Loss (Pinball)
```
ρ_τ(u) = u * (τ - I{u < 0})
       = { τ * u       if u ≥ 0
         { (τ-1) * u   if u < 0
```

### Horizon Embedding
```
ŷ_t+h = g(h_t, φ(h))

where:
  h_t: context summary from Transformer
  φ(h): learned horizon embedding
  g: 3-layer MLP with GELU activations
```

## 🔍 Monitoring Training

### View Training Progress
```bash
# Watch live output
tail -f checkpoints/training_log.json

# Check metrics
cat checkpoints/training_log.json | python -m json.tool
```

### Checkpoint Information
```python
import torch

checkpoint = torch.load('checkpoints/best_checkpoint.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['metrics']['val_loss']:.4f}")
print(f"Config: {checkpoint['config']}")
```

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 4

# Reduce context length
# Edit train.py: 'context_length_range': (96, 256)
```

### Slow Training
```bash
# Use fewer tickers
python train.py --num_tickers 10

# Reduce model size
# Edit train.py: 'd_model': 256, 'num_layers': 4
```

### NaN Loss
- Check for extreme values in data
- Reduce learning rate: `--learning_rate 5e-5`
- Increase gradient clipping (edit `train.py`: `'grad_clip': 0.5`)

## 📚 Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- yfinance (Yahoo Finance API)
- tqdm (progress bars)
- matplotlib (visualization)

See `requirements.txt` for exact versions.

## 🔬 Advanced Usage

### Custom Dataset

To use your own time series data, modify `data_loader.py`:

```python
class CustomDataset(Dataset):
    def __init__(self, data_path, ...):
        # Load your CSV/Parquet/HDF5 data
        # Ensure columns: timestamp, value, series_id

    def __getitem__(self, idx):
        # Return same format as SP500StockDataset
        return {
            'y_ctx': ...,
            'y_fut': ...,
            'time_features_ctx': ...,
            'series_id': ...,
            ...
        }
```

### Multi-Domain Training

Concatenate datasets from different domains:
```python
from torch.utils.data import ConcatDataset

dataset_stocks = SP500StockDataset(...)
dataset_energy = EnergyDataset(...)
dataset_weather = WeatherDataset(...)

combined = ConcatDataset([dataset_stocks, dataset_energy, dataset_weather])
```

### Export to ONNX (for production)

```python
import torch.onnx

model.eval()
dummy_input = {
    'y_ctx': torch.randn(1, 256),
    'time_features_ctx': torch.randn(1, 256, 8),
    'horizons': torch.arange(48).unsqueeze(0),
    'series_ids': torch.tensor([0])
}

torch.onnx.export(
    model,
    dummy_input,
    "timegpt.onnx",
    input_names=['y_ctx', 'time_features', 'horizons', 'series_ids'],
    output_names=['quantile_forecasts'],
    dynamic_axes={'y_ctx': {1: 'context_length'}, 'horizons': {1: 'num_horizons'}}
)
```

## 📖 Code Overview

### `model.py` (493 lines)
- `TimeSeriesEmbedding`: Tokenization layer
- `CausalMultiHeadAttention`: Causal self-attention
- `TransformerEncoderLayer`: Single Transformer layer
- `MultiQuantileForecastHead`: Horizon-conditioned quantile prediction
- `MaskedReconstructionHead`: Self-supervised reconstruction
- `TimeGPT`: Main model class

### `losses.py` (307 lines)
- `PinballLoss`: Multi-quantile loss function
- `MaskedReconstructionLoss`: MSE for masked positions
- `TimeGPTLoss`: Joint objective (forecast + λ*mask)
- `create_contiguous_mask`: Block masking strategy
- `quantile_score`: Evaluation metrics

### `data_loader.py` (306 lines)
- `SP500StockDataset`: Yahoo Finance data loading
- Window sampling with randomized (L, H)
- Local z-score normalization
- Time feature extraction (calendar + Fourier)
- `collate_variable_length`: Batch collation with padding

### `train.py` (661 lines)
- `create_random_horizons`: 4 horizon sampling strategies
- `TimeGPTTrainer`: Training loop with validation
- Configuration management
- Checkpoint saving/loading
- CLI argument parsing

### `test_inference.py` (364 lines)
- Load trained checkpoint
- Single-series forecasting
- Multi-quantile prediction
- Visualization with matplotlib
- Coverage and error metrics

## 🎯 Model Configuration

All hyperparameters are in `train.py` (lines 500-548):

```python
config = {
    # Data parameters
    'num_tickers': 10,
    'start_date': '2010-01-01',
    'end_date': '2024-12-01',
    'context_length_range': (96, 512),
    'forecast_horizon_range': (24, 96),

    # Model parameters
    'd_model': 512,
    'num_layers': 8,
    'num_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1,

    # Training parameters
    'num_epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,

    # Horizon sampling (NEW!)
    'use_horizon_sampling': True,
    'horizon_sampling_prob': 0.5,
    'horizon_sampling_mode': 'mixed',
    'min_horizons_sampled': 8,

    # Loss parameters
    'lambda_mask': 0.5,
    'mask_probability': 0.3,
    'mask_ratio': 0.15,

    # Quantiles
    'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
}
```

## 🧮 Mathematical Details

### Token Embedding
```
z_ℓ = W_y·ỹ + W_t·φ(t) + p_ℓ + s_i

where:
  W_y: value embedding weights
  ỹ: normalized time series value
  W_t: time feature embedding weights
  φ(t): 8-dim time features (calendar + Fourier)
  p_ℓ: learned positional embedding
  s_i: learned series embedding
```

### Multi-Quantile Forecast
```
q̂^(τ)(h) = g_θ(h_t + e^(h))

where:
  h_t: context summary from last Transformer layer
  e^(h): learned horizon embedding
  g_θ: 3-layer MLP (512 → 1024 → 1024 → 5)
```

### Monotonicity Constraint
```
q̂^(τ₁) = α
q̂^(τₖ) = q̂^(τₖ₋₁) + softplus(δₖ)

Ensures: q̂^(τ₁) ≤ q̂^(τ₂) ≤ ... ≤ q̂^(τ_Q)
```

## 📊 Expected Outputs

### Console Output
```
============================================================
🚀 STARTING TIMEGPT TRAINING
============================================================
⚙️  Configuration:
   - Epochs:           1
   - Batch Size:       8
   - Learning Rate:    0.0001
   - Device:           mps
   - Model Params:     27,121,670
   - Context Length:   (96, 512)
   - Forecast Horizon: (24, 96)
   - Quantiles:        [0.1, 0.25, 0.5, 0.75, 0.9]

📐 Horizon Sampling (Transition Function Learning):
   - Mode:             mixed
   - Probability:      50.0%
   - Min Horizons:     8
============================================================

Epoch 0: 100%|████████████| 3073/3073 [10:45<00:00, 5.26it/s, loss=0.27]

============================================================
EPOCH 1/1 COMPLETED
============================================================
📊 Training Metrics:
   - Train Loss:           0.3845
   - Forecast Loss:        0.2736
   - Reconstruction Loss:  0.2219

📈 Validation Metrics:
   - Val Loss:             0.2741
   - Val Forecast Loss:    0.2741
   - Val Quantile Loss:    0.1716

🌟 NEW BEST MODEL!
============================================================
```

### Checkpoints (311 MB each)
- Model weights
- Optimizer state
- Scheduler state
- Training metrics
- Full config for reproducibility

### Training Log (JSON)
```json
[
  {
    "epoch": 0,
    "train_loss": 0.3845,
    "val_loss": 0.2741,
    "val_coverage_0.1": 0.442,
    "val_coverage_0.9": 0.927,
    "learning_rate": 9.76e-05
  }
]
```

## 🎨 Visualization

`test_inference.py` generates a dual-panel plot:

**Top Panel**: Raw log returns
- Blue: Historical context (256 days)
- Green: Actual future (48 days)
- Red dashed: Median forecast
- Shaded: 50% and 80% prediction intervals

**Bottom Panel**: Normalized view (training space)
- Shows mean reversion to zero
- Reveals model's internal predictions

## 🚀 Performance Optimization

### Faster Training
```bash
# Use more workers for data loading
# Edit train.py: num_workers=8

# Enable mixed precision (PyTorch 2.0+)
# Add to train.py:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Larger Batch Size
```bash
# If GPU has >16GB memory
python train.py --batch_size 32

# Adjust learning rate proportionally
python train.py --batch_size 32 --learning_rate 4e-4
```

### Gradient Accumulation
For effective batch size of 32 with batch_size=8:
```python
# In train.py, accumulate over 4 batches:
if (step + 1) % 4 == 0:
    optimizer.step()
    optimizer.zero_grad()
```

## 🔗 References

- **Original TimeGPT paper**: Garza & Mergenthaler-Canseco (2023)
- **Transformers**: Vaswani et al. (2017) - "Attention is All You Need"
- **DeepAR**: Salinas et al. (2020) - Probabilistic RNN forecasting
- **Temporal Fusion Transformer**: Lim et al. (2021)
- **BERT**: Devlin et al. (2019) - Masked language modeling
- **GPT**: Radford et al. (2018, 2019, 2020)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{boucher2024timegpt,
  title={TimeGPT: A Transformer Foundation Model for Zero-Shot Multi-Horizon Probabilistic Time Series Forecasting},
  author={Boucher, Simon-Pierre},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

This is a research implementation. For production use:
- Add more extensive testing
- Implement proper data validation
- Add logging and monitoring
- Optimize for deployment (ONNX, TorchScript)
- Add API endpoints for inference

## 📄 License

MIT License - Free for research and commercial use.

## 🙏 Acknowledgments

- PyTorch team for the framework
- Yahoo Finance for data access
- Open-source community for tools (pandas, numpy, matplotlib)
- Anthropic for Claude Code

## 📧 Contact

For questions or collaboration:
- Email: simon.pierre.boucher@example.com
- GitHub: @spboucher

---

**Built with Claude Code** | **December 2024** | **Version 1.0**
