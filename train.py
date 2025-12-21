import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import json
import argparse

from model import TimeGPT
from losses import TimeGPTLoss, create_contiguous_mask, quantile_score
from data_loader import (
    SP500StockDataset,
    get_sp500_tickers,
    collate_variable_length
)


def create_random_horizons(
    batch_size: int,
    max_horizon: int,
    actual_horizons: torch.Tensor,
    mode: str = "mixed",
    min_horizons: int = 8,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create randomized horizon indices for training to force the model
    to learn a true transition function g(h_t, φ(h)) rather than a lookup table.

    Strategies:
        - "all": Return all horizons [0, 1, ..., H-1] (original behavior)
        - "random_subset": Sample random subset of horizons (e.g., [2, 5, 10, 24, 48])
        - "random_individual": Sample individual random horizons per sample
        - "mixed": Randomly choose between strategies

    Args:
        batch_size: Number of samples in batch
        max_horizon: Maximum horizon value (padded size)
        actual_horizons: (batch_size,) - actual (unpadded) horizon lengths
        mode: Sampling strategy
        min_horizons: Minimum number of horizons to sample
        device: Device to create tensor on

    Returns:
        Horizon indices: (batch_size, num_horizons)
    """
    if mode == "all":
        # Original behavior: predict all horizons
        horizons = torch.arange(max_horizon, device=device).unsqueeze(0).expand(batch_size, -1)
        return horizons

    elif mode == "random_subset":
        # Sample same subset for entire batch (good for batch efficiency)
        # Sample between min_horizons and max_horizon points
        num_to_sample = np.random.randint(min_horizons, max_horizon + 1)

        # Sample unique horizons
        sampled_h = torch.from_numpy(
            np.sort(np.random.choice(max_horizon, size=num_to_sample, replace=False))
        ).long()

        # Expand to batch
        horizons = sampled_h.unsqueeze(0).expand(batch_size, -1).to(device)
        return horizons

    elif mode == "random_individual":
        # Sample different horizons for each sample in batch
        horizons_list = []
        for i in range(batch_size):
            actual_H = min(actual_horizons[i].item(), max_horizon)
            num_to_sample = np.random.randint(min_horizons, actual_H + 1)

            sampled_h = torch.from_numpy(
                np.sort(np.random.choice(actual_H, size=num_to_sample, replace=False))
            ).long()

            # Pad to max length in this batch
            padded = torch.zeros(max_horizon, dtype=torch.long)
            padded[:len(sampled_h)] = sampled_h
            horizons_list.append(padded)

        horizons = torch.stack(horizons_list).to(device)
        return horizons

    elif mode == "mixed":
        # Randomly choose a strategy for this batch
        strategy = np.random.choice(["all", "random_subset", "random_individual"])
        return create_random_horizons(
            batch_size, max_horizon, actual_horizons, mode=strategy,
            min_horizons=min_horizons, device=device
        )

    else:
        raise ValueError(f"Unknown horizon sampling mode: {mode}")


class TimeGPTTrainer:
    """Trainer for TimeGPT model"""

    def __init__(
        self,
        model: TimeGPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = TimeGPTLoss(
            quantiles=config['quantiles'],
            lambda_mask=config['lambda_mask'],
            use_masked_loss=config['use_masked_loss']
        ).to(device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=config['scheduler_tmult'],
            eta_min=config['learning_rate'] * 0.01
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Logging
        self.log_file = os.path.join(self.checkpoint_dir, 'training_log.json')
        self.training_history = []

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_forecast_loss = 0
        total_reconstruction_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Prepare inputs
            y_ctx = batch['y_ctx']
            y_fut = batch['y_fut']
            time_features_ctx = batch['time_features_ctx']
            series_ids = batch['series_id']
            ctx_mask = batch['ctx_mask']
            fut_mask = batch['fut_mask']
            actual_H = batch['actual_H']

            # Create horizon indices for each sample
            # Use the actual padded size to avoid dimension mismatch
            batch_size, max_H_padded = y_fut.shape

            # Use randomized horizons during training to learn transition function
            # instead of lookup table (implements "GPT-style" horizon conditioning)
            use_random_horizons = (
                self.config.get('use_horizon_sampling', True) and
                np.random.random() < self.config.get('horizon_sampling_prob', 0.5)
            )

            if use_random_horizons:
                horizons = create_random_horizons(
                    batch_size=batch_size,
                    max_horizon=max_H_padded,
                    actual_horizons=actual_H,
                    mode=self.config.get('horizon_sampling_mode', 'mixed'),
                    min_horizons=self.config.get('min_horizons_sampled', 8),
                    device=self.device
                )
                # Extract y_fut values at sampled horizons for loss computation
                # horizons: (B, num_sampled_horizons)
                # y_fut: (B, max_H_padded)
                # We need y_fut_sampled: (B, num_sampled_horizons)
                batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand_as(horizons)
                y_fut_sampled = y_fut[batch_indices, horizons]
                fut_mask_sampled = fut_mask[batch_indices, horizons]
            else:
                # Standard: predict all horizons
                horizons = torch.arange(max_H_padded, device=self.device).unsqueeze(0).expand(batch_size, -1)
                y_fut_sampled = y_fut
                fut_mask_sampled = fut_mask

            # Forward pass with or without masking
            if self.config['use_masked_loss'] and np.random.random() < self.config['mask_probability']:
                # Create masked inputs for self-supervised learning
                mask_indices = create_contiguous_mask(
                    batch_size=batch_size,
                    seq_len=y_ctx.shape[1],
                    mask_ratio=self.config['mask_ratio'],
                    max_span_length=self.config['max_mask_span'],
                    device=self.device
                )

                outputs = self.model.forward_with_masking(
                    y_ctx=y_ctx,
                    time_features_ctx=time_features_ctx,
                    horizons=horizons,
                    mask_indices=mask_indices,
                    series_ids=series_ids,
                    ctx_mask=ctx_mask
                )

                # Compute loss
                loss_dict = self.criterion(
                    outputs=outputs,
                    y_fut=y_fut_sampled,
                    y_ctx=y_ctx,
                    fut_mask=fut_mask_sampled
                )
            else:
                # Standard forecasting forward pass
                outputs = self.model(
                    y_ctx=y_ctx,
                    time_features_ctx=time_features_ctx,
                    horizons=horizons,
                    series_ids=series_ids,
                    ctx_mask=ctx_mask
                )

                # Compute loss
                loss_dict = self.criterion(
                    outputs=outputs,
                    y_fut=y_fut_sampled,
                    fut_mask=fut_mask_sampled
                )

            loss = loss_dict['loss']

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step(self.current_epoch + num_batches / len(self.train_loader))

            # Accumulate metrics
            total_loss += loss.item()
            total_forecast_loss += loss_dict['forecast_loss'].item()
            if 'reconstruction_loss' in loss_dict:
                total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
            num_batches += 1

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        # Compute average metrics
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_forecast_loss': total_forecast_loss / num_batches,
            'train_reconstruction_loss': total_reconstruction_loss / num_batches if total_reconstruction_loss > 0 else 0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        total_forecast_loss = 0
        num_batches = 0

        all_y_true = []
        all_y_pred_quantiles = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            y_ctx = batch['y_ctx']
            y_fut = batch['y_fut']
            time_features_ctx = batch['time_features_ctx']
            series_ids = batch['series_id']
            ctx_mask = batch['ctx_mask']
            fut_mask = batch['fut_mask']
            actual_H = batch['actual_H']

            # Create horizon indices
            # Use the actual padded size to avoid dimension mismatch
            batch_size, max_H_padded = y_fut.shape
            horizons = torch.arange(max_H_padded, device=self.device).unsqueeze(0).expand(batch_size, -1)

            # Forward pass
            outputs = self.model(
                y_ctx=y_ctx,
                time_features_ctx=time_features_ctx,
                horizons=horizons,
                series_ids=series_ids,
                ctx_mask=ctx_mask
            )

            # Compute loss
            loss_dict = self.criterion(
                outputs=outputs,
                y_fut=y_fut,
                fut_mask=fut_mask
            )

            total_loss += loss_dict['loss'].item()
            total_forecast_loss += loss_dict['forecast_loss'].item()
            num_batches += 1

            # Collect predictions for metrics (only valid positions)
            # Extract valid positions using fut_mask for each sample
            for b in range(y_fut.size(0)):
                actual_h = fut_mask[b].sum().item()
                if actual_h > 0:
                    all_y_true.append(y_fut[b, :actual_h].cpu().unsqueeze(0))
                    all_y_pred_quantiles.append(outputs['quantile_forecasts'][b, :actual_h].cpu().unsqueeze(0))

        # Concatenate all predictions
        # Find max horizon in collected data
        max_h_collected = max(y.size(1) for y in all_y_true)

        # Pad all to same size before concatenating
        all_y_true_padded = []
        all_y_pred_quantiles_padded = []
        for y_true, y_pred in zip(all_y_true, all_y_pred_quantiles):
            h = y_true.size(1)
            if h < max_h_collected:
                # Pad to max size
                y_true_padded = torch.cat([y_true, torch.zeros(1, max_h_collected - h)], dim=1)
                y_pred_padded = torch.cat([
                    y_pred,
                    torch.zeros(1, max_h_collected - h, y_pred.size(2))
                ], dim=1)
            else:
                y_true_padded = y_true
                y_pred_padded = y_pred
            all_y_true_padded.append(y_true_padded)
            all_y_pred_quantiles_padded.append(y_pred_padded)

        all_y_true = torch.cat(all_y_true_padded, dim=0)
        all_y_pred_quantiles = torch.cat(all_y_pred_quantiles_padded, dim=0)

        # Compute quantile metrics
        quantile_metrics = quantile_score(
            all_y_true,
            all_y_pred_quantiles,
            self.config['quantiles']
        )

        metrics = {
            'val_loss': total_loss / num_batches,
            'val_forecast_loss': total_forecast_loss / num_batches,
            **{f'val_{k}': v for k, v in quantile_metrics.items()}
        }

        return metrics

    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_loss: {metrics['val_loss']:.4f}")

        # Save periodic checkpoint
        if self.current_epoch % self.config['save_every'] == 0:
            epoch_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{self.current_epoch}.pt'
            )
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("🚀 STARTING TIMEGPT TRAINING")
        print("="*60)
        print(f"⚙️  Configuration:")
        print(f"   - Epochs:           {self.config['num_epochs']}")
        print(f"   - Batch Size:       {self.config['batch_size']}")
        print(f"   - Learning Rate:    {self.config['learning_rate']}")
        print(f"   - Device:           {self.device}")
        print(f"   - Model Params:     {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - Context Length:   {self.config['context_length_range']}")
        print(f"   - Forecast Horizon: {self.config['forecast_horizon_range']}")
        print(f"   - Quantiles:        {self.config['quantiles']}")
        print(f"\n📐 Horizon Sampling (Transition Function Learning):")
        if self.config.get('use_horizon_sampling', False):
            print(f"   - Mode:             {self.config['horizon_sampling_mode']}")
            print(f"   - Probability:      {self.config['horizon_sampling_prob']:.1%}")
            print(f"   - Min Horizons:     {self.config['min_horizons_sampled']}")
        else:
            print(f"   - Disabled (always predict all horizons)")
        print("="*60 + "\n")

        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if (epoch + 1) % self.config['validate_every'] == 0:
                val_metrics = self.validate()

                # Combine metrics
                metrics = {**train_metrics, **val_metrics}

                # Check if best model
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']

                # Save checkpoint
                self.save_checkpoint(metrics, is_best)

                # Log metrics
                print("\n" + "="*60)
                print(f"EPOCH {epoch + 1}/{self.config['num_epochs']} COMPLETED")
                print("="*60)
                print(f"📊 Training Metrics:")
                print(f"   - Train Loss:           {train_metrics['train_loss']:.4f}")
                print(f"   - Forecast Loss:        {train_metrics['train_forecast_loss']:.4f}")
                if train_metrics['train_reconstruction_loss'] > 0:
                    print(f"   - Reconstruction Loss:  {train_metrics['train_reconstruction_loss']:.4f}")
                print(f"\n📈 Validation Metrics:")
                print(f"   - Val Loss:             {val_metrics['val_loss']:.4f}")
                print(f"   - Val Forecast Loss:    {val_metrics['val_forecast_loss']:.4f}")
                print(f"   - Val Quantile Loss:    {val_metrics['val_quantile_loss']:.4f}")
                if is_best:
                    print(f"\n🌟 NEW BEST MODEL! (Previous: {self.best_val_loss:.4f})")
                print(f"\n⏱️  Learning Rate:        {train_metrics['learning_rate']:.6f}")
                print("="*60)

                # Save to history
                metrics['epoch'] = epoch
                self.training_history.append(metrics)

                # Save training history
                with open(self.log_file, 'w') as f:
                    json.dump(self.training_history, f, indent=2)

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"🏆 Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"📁 Checkpoints saved in: {self.checkpoint_dir}")
        print(f"📊 Training log saved in: {self.log_file}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train TimeGPT model')
    parser.add_argument('--num_tickers', type=int, default=50,
                        help='Number of S&P 500 tickers to use')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--horizon_sampling_mode', type=str, default='mixed',
                        choices=['all', 'random_subset', 'random_individual', 'mixed'],
                        help='Horizon sampling strategy for training')
    parser.add_argument('--no_horizon_sampling', action='store_true',
                        help='Disable randomized horizon sampling')

    args = parser.parse_args()

    # Configuration
    config = {
        # Data parameters
        'num_tickers': args.num_tickers,
        'start_date': '2010-01-01',
        'end_date': '2024-12-01',
        'context_length_range': (96, 512),
        'forecast_horizon_range': (24, 96),
        'train_split': 0.8,

        # Model parameters
        'd_model': 512,
        'num_layers': 8,
        'num_heads': 8,
        'd_ff': 2048,
        'max_seq_len': 512,
        'max_horizon': 96,
        'dropout': 0.1,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],

        # Training parameters
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,

        # Loss parameters
        'lambda_mask': 0.5,
        'use_masked_loss': True,
        'mask_probability': 0.3,
        'mask_ratio': 0.15,
        'max_mask_span': 10,

        # Horizon sampling parameters (for learning transition function)
        'use_horizon_sampling': not args.no_horizon_sampling,  # Enable randomized horizon sampling
        'horizon_sampling_prob': 0.5,  # Probability of using random horizons (vs all horizons)
        'horizon_sampling_mode': args.horizon_sampling_mode,  # 'all', 'random_subset', 'random_individual', 'mixed'
        'min_horizons_sampled': 8,  # Minimum number of horizons to sample

        # Scheduler parameters
        'scheduler_t0': 10,
        'scheduler_tmult': 2,

        # Logging parameters
        'validate_every': 1,
        'save_every': 5,
        'checkpoint_dir': args.checkpoint_dir,
    }

    # Device (prioritize MPS for Apple Silicon, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Get tickers
    print("Loading S&P 500 tickers...")
    tickers = get_sp500_tickers(config['num_tickers'])
    print(f"Using {len(tickers)} tickers")

    # Create datasets
    print("Creating datasets...")
    full_dataset = SP500StockDataset(
        tickers=tickers,
        start_date=config['start_date'],
        end_date=config['end_date'],
        context_length_range=config['context_length_range'],
        forecast_horizon_range=config['forecast_horizon_range']
    )

    # Split into train/val
    train_size = int(config['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create data loaders
    # Disable pin_memory for MPS as it's not supported yet
    use_pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_variable_length,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_variable_length,
        pin_memory=use_pin_memory
    )

    # Create model
    print("Creating model...")
    model = TimeGPT(
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        max_horizon=config['max_horizon'],
        time_feature_dim=8,  # From data loader
        num_series=len(tickers),
        quantiles=config['quantiles'],
        dropout=config['dropout'],
        use_series_embedding=True,
        use_masked_reconstruction=config['use_masked_loss']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = TimeGPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
