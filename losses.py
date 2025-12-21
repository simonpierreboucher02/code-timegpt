import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """
    Multi-quantile pinball loss for probabilistic forecasting.
    Implements Eq. 16-17 from the paper.

    The pinball loss for quantile τ is:
        ρ_τ(u) = u * (τ - I{u < 0})

    where u = y_true - y_pred
    """

    def __init__(self, quantiles: list):
        """
        Args:
            quantiles: List of quantile levels τ ∈ (0, 1)
        """
        super().__init__()
        self.quantiles = torch.FloatTensor(quantiles)
        self.num_quantiles = len(quantiles)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute multi-quantile pinball loss.

        Args:
            y_true: (batch_size, H) - true future values (normalized)
            y_pred: (batch_size, H, num_quantiles) - predicted quantiles
            mask: (batch_size, H) - valid positions (optional)

        Returns:
            Scalar loss value
        """
        batch_size, H, num_quantiles = y_pred.shape

        # Move quantiles to same device as predictions
        quantiles = self.quantiles.to(y_pred.device)

        # Expand dimensions for broadcasting
        y_true_exp = y_true.unsqueeze(-1)  # (B, H, 1)
        quantiles_exp = quantiles.view(1, 1, -1)  # (1, 1, Q)

        # Compute errors: u = y_true - y_pred
        errors = y_true_exp - y_pred  # (B, H, Q)

        # Pinball loss: ρ_τ(u) = u * (τ - I{u < 0})
        # Equivalent to: max(τ * u, (τ - 1) * u)
        loss = torch.where(
            errors >= 0,
            quantiles_exp * errors,
            (quantiles_exp - 1) * errors
        )

        # Apply mask if provided
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)  # (B, H, 1)
            loss = loss * mask_exp
            # Normalize by number of valid elements
            loss = loss.sum() / (mask.sum() * num_quantiles)
        else:
            loss = loss.mean()

        return loss


class MaskedReconstructionLoss(nn.Module):
    """
    Masked reconstruction loss for self-supervised pretraining.
    Implements Eq. 18 from the paper.

    L_mask(θ) = Σ_{ℓ∈M} (ỹ_{i,t-L+ℓ} - ŷ_{i,t-L+ℓ})²
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss only at masked positions.

        Args:
            y_true: (batch_size, L) - true context values (normalized)
            y_pred: (batch_size, L) - reconstructed values
            mask_indices: (batch_size, L) - binary mask (1 = masked position)

        Returns:
            Scalar loss value
        """
        # Compute MSE for all positions
        mse_all = self.mse(y_pred, y_true)  # (B, L)

        # Only consider masked positions
        masked_mse = mse_all * mask_indices

        # Average over masked positions
        num_masked = mask_indices.sum()
        if num_masked > 0:
            loss = masked_mse.sum() / num_masked
        else:
            loss = torch.tensor(0.0, device=y_true.device)

        return loss


class TimeGPTLoss(nn.Module):
    """
    Joint objective for TimeGPT training.
    Implements Eq. 19 from the paper:

    L_total(θ) = E[(L_forecast(θ) + λ * L_mask(θ))]

    where:
        - L_forecast: Multi-quantile pinball loss for forecasting
        - L_mask: Masked reconstruction loss for self-supervised learning
        - λ: Weight balancing the two objectives
    """

    def __init__(
        self,
        quantiles: list,
        lambda_mask: float = 0.5,
        use_masked_loss: bool = True
    ):
        """
        Args:
            quantiles: List of quantile levels
            lambda_mask: Weight for masked reconstruction loss (λ in paper)
            use_masked_loss: Whether to include masked reconstruction loss
        """
        super().__init__()

        self.pinball_loss = PinballLoss(quantiles)
        self.reconstruction_loss = MaskedReconstructionLoss()

        self.lambda_mask = lambda_mask
        self.use_masked_loss = use_masked_loss

    def forward(
        self,
        outputs: dict,
        y_fut: torch.Tensor,
        y_ctx: torch.Tensor = None,
        fut_mask: torch.Tensor = None
    ) -> dict:
        """
        Compute joint loss.

        Args:
            outputs: Dictionary from model forward pass containing:
                - quantile_forecasts: (batch_size, H, num_quantiles)
                - reconstructions: (batch_size, L) [optional]
                - mask_indices: (batch_size, L) [optional]
            y_fut: (batch_size, H) - true future values
            y_ctx: (batch_size, L) - true context values [for reconstruction]
            fut_mask: (batch_size, H) - valid positions in future

        Returns:
            Dictionary containing:
                - loss: Total loss
                - forecast_loss: Forecasting component
                - reconstruction_loss: Reconstruction component (if applicable)
        """
        # Forecasting loss
        quantile_forecasts = outputs['quantile_forecasts']
        forecast_loss = self.pinball_loss(y_fut, quantile_forecasts, fut_mask)

        total_loss = forecast_loss
        result = {
            'loss': total_loss,
            'forecast_loss': forecast_loss
        }

        # Add masked reconstruction loss if applicable
        if self.use_masked_loss and 'reconstructions' in outputs:
            reconstructions = outputs['reconstructions']
            mask_indices = outputs['mask_indices']

            reconstruction_loss = self.reconstruction_loss(
                y_ctx, reconstructions, mask_indices
            )

            total_loss = forecast_loss + self.lambda_mask * reconstruction_loss

            result['loss'] = total_loss
            result['reconstruction_loss'] = reconstruction_loss

        return result


def create_contiguous_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.15,
    max_span_length: int = 10,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create contiguous block masks for self-supervised learning.
    This is a common strategy for time series masking.

    Args:
        batch_size: Number of sequences in batch
        seq_len: Length of each sequence
        mask_ratio: Approximate ratio of positions to mask
        max_span_length: Maximum length of contiguous masked spans
        device: Device to create tensor on

    Returns:
        Binary mask: (batch_size, seq_len) with 1 = masked position
    """
    mask = torch.zeros(batch_size, seq_len, device=device)

    for i in range(batch_size):
        num_to_mask = int(seq_len * mask_ratio)
        masked_count = 0

        while masked_count < num_to_mask:
            # Sample span length
            span_length = torch.randint(1, max_span_length + 1, (1,)).item()
            span_length = min(span_length, num_to_mask - masked_count)

            # Sample start position
            max_start = seq_len - span_length
            if max_start <= 0:
                break

            start_pos = torch.randint(0, max_start + 1, (1,)).item()

            # Apply mask
            mask[i, start_pos:start_pos + span_length] = 1.0
            masked_count += span_length

    return mask


def quantile_score(
    y_true: torch.Tensor,
    y_pred_quantiles: torch.Tensor,
    quantiles: list
) -> dict:
    """
    Compute quantile score metrics for evaluation.

    Args:
        y_true: (batch_size, H) - true values
        y_pred_quantiles: (batch_size, H, num_quantiles) - predicted quantiles
        quantiles: List of quantile levels

    Returns:
        Dictionary of metrics including:
            - quantile_loss: Average quantile loss
            - coverage: Coverage for each quantile level
            - interval_width: Width of prediction intervals
    """
    batch_size, H, num_quantiles = y_pred_quantiles.shape

    # Compute quantile loss
    pinball = PinballLoss(quantiles)
    loss = pinball(y_true, y_pred_quantiles)

    # Compute coverage (what % of true values fall below each quantile)
    coverage = {}
    for i, q in enumerate(quantiles):
        pred_q = y_pred_quantiles[:, :, i]
        below = (y_true <= pred_q).float().mean()
        coverage[f'coverage_{q}'] = below.item()

    # Compute prediction interval widths
    # E.g., 80% interval = q_0.9 - q_0.1
    interval_widths = {}
    quantile_to_idx = {q: i for i, q in enumerate(quantiles)}

    # Common intervals: 50%, 80%, 90%
    intervals = [(0.25, 0.75), (0.1, 0.9), (0.05, 0.95)]

    for lower_q, upper_q in intervals:
        if lower_q in quantile_to_idx and upper_q in quantile_to_idx:
            lower_idx = quantile_to_idx[lower_q]
            upper_idx = quantile_to_idx[upper_q]

            lower_pred = y_pred_quantiles[:, :, lower_idx]
            upper_pred = y_pred_quantiles[:, :, upper_idx]

            width = (upper_pred - lower_pred).mean()
            interval_name = f'interval_width_{int((upper_q - lower_q) * 100)}%'
            interval_widths[interval_name] = width.item()

    return {
        'quantile_loss': loss.item(),
        **coverage,
        **interval_widths
    }
