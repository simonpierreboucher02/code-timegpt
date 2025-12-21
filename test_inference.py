"""
Script to test TimeGPT model inference on a single series.
Loads trained checkpoint and generates multi-horizon quantile forecasts.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

from model import TimeGPT
from data_loader import SP500StockDataset


def extract_time_features(dates):
    """Extract time features from dates (same as data_loader.py)"""
    features = []
    for date in dates:
        day_of_week = date.dayofweek / 6.0
        day_of_month = date.day / 31.0
        month = date.month / 12.0
        day_of_year = date.dayofyear / 365.0

        freq_week = 2 * np.pi * date.dayofweek / 7.0
        sin_week = np.sin(freq_week)
        cos_week = np.cos(freq_week)

        freq_year = 2 * np.pi * date.dayofyear / 365.0
        sin_year = np.sin(freq_year)
        cos_year = np.cos(freq_year)

        feat = [
            day_of_week, day_of_month, month, day_of_year,
            sin_week, cos_week, sin_year, cos_year
        ]
        features.append(feat)

    return np.array(features, dtype=np.float32)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    epoch = checkpoint['epoch']

    print(f"\nCheckpoint Info:")
    print(f"  - Epoch: {epoch}")
    print(f"  - Global Step: {checkpoint['global_step']}")
    print(f"  - Val Loss: {checkpoint['metrics']['val_loss']:.4f}")

    # Create model with same config as training
    # Extract num_series from checkpoint (model was trained with num_tickers series)
    series_embedding_size = checkpoint['model_state_dict']['embedding.series_embedding.weight'].shape[0]

    model = TimeGPT(
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        max_horizon=config['max_horizon'],
        time_feature_dim=8,
        num_series=series_embedding_size,  # Must match training
        quantiles=config['quantiles'],
        dropout=config['dropout'],
        use_series_embedding=True,  # Must match training
        use_masked_reconstruction=config.get('use_masked_loss', True)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    return model, config


def forecast_single_series(
    model,
    ticker='AAPL',
    context_length=256,
    forecast_horizon=48,
    device='cpu'
):
    """
    Generate forecast for a single stock ticker.

    Args:
        model: Trained TimeGPT model
        ticker: Stock ticker symbol
        context_length: Length of context window
        forecast_horizon: Number of steps to forecast
        device: Device to run on

    Returns:
        Dictionary with forecasts and metadata
    """
    print(f"📈 Forecasting {ticker}")
    print(f"   - Context Length: {context_length}")
    print(f"   - Forecast Horizon: {forecast_horizon}")
    print(f"   - Device: {device}\n")

    # Download data
    print("Downloading data from Yahoo Finance...")
    stock = yf.Ticker(ticker)
    df = stock.history(start='2020-01-01', end='2024-12-21')

    if len(df) < context_length + forecast_horizon:
        raise ValueError(f"Insufficient data for {ticker}")

    # Compute log returns
    prices = df['Close'].values
    returns = np.diff(np.log(prices))
    dates = df.index[1:]  # Align with returns

    print(f"Downloaded {len(returns)} days of data\n")

    # Take last context_length + forecast_horizon points
    total_len = context_length + forecast_horizon
    y_all = returns[-total_len:]
    dates_all = dates[-total_len:]

    # Split into context and future
    y_ctx = y_all[:context_length]
    y_fut = y_all[context_length:]
    dates_ctx = dates_all[:context_length]
    dates_fut = dates_all[context_length:]

    # Local normalization (same as training)
    mu = np.mean(y_ctx)
    sigma = np.std(y_ctx) + 1e-8
    y_ctx_norm = (y_ctx - mu) / sigma
    y_fut_norm = (y_fut - mu) / sigma

    # Extract time features
    time_features_ctx = extract_time_features(dates_ctx)

    # Convert to tensors
    y_ctx_tensor = torch.FloatTensor(y_ctx_norm).unsqueeze(0).to(device)
    time_features_tensor = torch.FloatTensor(time_features_ctx).unsqueeze(0).to(device)
    horizons = torch.arange(forecast_horizon, device=device).unsqueeze(0)

    # Generate forecast
    print("Generating forecasts...")
    with torch.no_grad():
        # Use series_id=0 for single series inference
        series_ids = torch.LongTensor([0]).to(device)

        outputs = model(
            y_ctx=y_ctx_tensor,
            time_features_ctx=time_features_tensor,
            horizons=horizons,
            series_ids=series_ids,
            ctx_mask=None
        )

    # Extract predictions
    quantile_forecasts = outputs['quantile_forecasts'].cpu().numpy()[0]  # (H, Q)

    # Denormalize predictions
    quantile_forecasts_denorm = quantile_forecasts * sigma + mu

    print("✅ Forecasts generated!\n")

    return {
        'ticker': ticker,
        'dates_ctx': dates_ctx,
        'dates_fut': dates_fut,
        'y_ctx': y_ctx,
        'y_fut': y_fut,
        'y_ctx_norm': y_ctx_norm,
        'y_fut_norm': y_fut_norm,
        'quantile_forecasts_norm': quantile_forecasts,
        'quantile_forecasts': quantile_forecasts_denorm,
        'quantiles': model.quantiles,
        'mu': mu,
        'sigma': sigma
    }


def plot_forecast(result, save_path='forecast_plot.png'):
    """Plot forecast with quantile intervals"""
    dates_ctx = result['dates_ctx']
    dates_fut = result['dates_fut']
    y_ctx = result['y_ctx']
    y_fut = result['y_fut']
    quantile_forecasts = result['quantile_forecasts']
    quantiles = result['quantiles']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Full view with context and forecast
    ax1.plot(dates_ctx, y_ctx, 'b-', label='Context (observed)', linewidth=1.5)
    ax1.plot(dates_fut, y_fut, 'g-', label='Future (actual)', linewidth=1.5, alpha=0.7)

    # Plot median forecast
    median_idx = quantiles.index(0.5)
    ax1.plot(dates_fut, quantile_forecasts[:, median_idx], 'r--',
             label='Forecast (median)', linewidth=2)

    # Plot quantile intervals
    colors = ['lightcoral', 'coral', 'orange']
    intervals = [(0, 4), (1, 3), (2, 2)]  # (q0.1, q0.9), (q0.25, q0.75), (q0.5, q0.5)
    labels = ['80% interval', '50% interval', 'median']

    for i, (lower_idx, upper_idx) in enumerate(intervals):
        if lower_idx != upper_idx:
            ax1.fill_between(
                dates_fut,
                quantile_forecasts[:, lower_idx],
                quantile_forecasts[:, upper_idx],
                alpha=0.3,
                color=colors[i],
                label=labels[i]
            )

    ax1.axvline(dates_ctx[-1], color='black', linestyle=':', alpha=0.5,
                label='Forecast start')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log Returns')
    ax1.set_title(f'{result["ticker"]} - Multi-Quantile Forecast', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Normalized view
    ax2.plot(dates_ctx, result['y_ctx_norm'], 'b-', label='Context (normalized)', linewidth=1.5)
    ax2.plot(dates_fut, result['y_fut_norm'], 'g-', label='Future (actual)', linewidth=1.5, alpha=0.7)
    ax2.plot(dates_fut, result['quantile_forecasts_norm'][:, median_idx], 'r--',
             label='Forecast (median)', linewidth=2)

    # Plot normalized intervals
    for i, (lower_idx, upper_idx) in enumerate(intervals):
        if lower_idx != upper_idx:
            ax2.fill_between(
                dates_fut,
                result['quantile_forecasts_norm'][:, lower_idx],
                result['quantile_forecasts_norm'][:, upper_idx],
                alpha=0.3,
                color=colors[i],
                label=labels[i]
            )

    ax2.axvline(dates_ctx[-1], color='black', linestyle=':', alpha=0.5)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Log Returns')
    ax2.set_title('Normalized View (training space)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {save_path}")

    return fig


def print_forecast_summary(result):
    """Print summary statistics of forecast"""
    y_fut = result['y_fut']
    quantile_forecasts = result['quantile_forecasts']
    quantiles = result['quantiles']

    print("\n" + "="*60)
    print("📊 FORECAST SUMMARY")
    print("="*60)

    # Compute errors
    median_idx = quantiles.index(0.5)
    median_forecast = quantile_forecasts[:, median_idx]

    mae = np.mean(np.abs(y_fut - median_forecast))
    mse = np.mean((y_fut - median_forecast) ** 2)
    rmse = np.sqrt(mse)

    print(f"\nMedian Forecast Errors:")
    print(f"  - MAE:  {mae:.6f}")
    print(f"  - RMSE: {rmse:.6f}")

    # Check coverage
    print(f"\nQuantile Coverage (actual vs target):")
    for i, q in enumerate(quantiles):
        below = (y_fut < quantile_forecasts[:, i]).sum()
        coverage = below / len(y_fut)
        print(f"  - q_{q:0.2f}: {coverage:6.1%} (target: {q*100:5.1f}%)")

    # Interval widths
    print(f"\nPrediction Interval Widths:")
    q10_idx = quantiles.index(0.1)
    q90_idx = quantiles.index(0.9)
    q25_idx = quantiles.index(0.25)
    q75_idx = quantiles.index(0.75)

    width_80 = np.mean(quantile_forecasts[:, q90_idx] - quantile_forecasts[:, q10_idx])
    width_50 = np.mean(quantile_forecasts[:, q75_idx] - quantile_forecasts[:, q25_idx])

    print(f"  - 80% interval: {width_80:.6f}")
    print(f"  - 50% interval: {width_50:.6f}")

    # First 5 forecasts
    print(f"\nFirst 5 Forecasts (median ± 80% interval):")
    for h in range(min(5, len(y_fut))):
        med = quantile_forecasts[h, median_idx]
        lower = quantile_forecasts[h, q10_idx]
        upper = quantile_forecasts[h, q90_idx]
        actual = y_fut[h]
        print(f"  h={h+1}: {med:8.5f} [{lower:8.5f}, {upper:8.5f}] | actual: {actual:8.5f}")

    print("="*60 + "\n")


def main():
    # Configuration
    checkpoint_path = 'checkpoints/best_checkpoint.pt'
    ticker = 'AAPL'
    context_length = 256
    forecast_horizon = 48

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("\n" + "="*60)
    print("🧪 TIMEGPT INFERENCE TEST")
    print("="*60)
    print(f"Device: {device}")
    print(f"Ticker: {ticker}")
    print("="*60 + "\n")

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)

    # Generate forecast
    result = forecast_single_series(
        model=model,
        ticker=ticker,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        device=device
    )

    # Print summary
    print_forecast_summary(result)

    # Plot results
    plot_forecast(result, save_path=f'forecast_{ticker}.png')

    print("\n" + "="*60)
    print("✅ INFERENCE TEST COMPLETE!")
    print("="*60)
    print(f"📊 Forecast plot: forecast_{ticker}.png")
    print(f"📈 Ticker: {ticker}")
    print(f"🎯 Horizons: 1 to {forecast_horizon}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
