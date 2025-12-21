import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TimeSeriesEmbedding(nn.Module):
    """
    Implements the tokenization strategy from Section 2.4 of the paper.
    Combines value, time feature, and positional embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        time_feature_dim: int,
        num_series: int,
        use_series_embedding: bool = True
    ):
        """
        Args:
            d_model: Model dimension (d in paper)
            max_seq_len: Maximum sequence length for positional embeddings
            time_feature_dim: Dimension of time features (K in paper)
            num_series: Number of unique series for series embeddings
            use_series_embedding: Whether to use series-specific embeddings
        """
        super().__init__()

        self.d_model = d_model
        self.use_series_embedding = use_series_embedding

        # Value embedding: W_y * y + b_y (Eq. 7)
        self.value_embedding = nn.Linear(1, d_model)

        # Time feature embedding: W_t * φ(t) + b_t (Eq. 7)
        self.time_embedding = nn.Linear(time_feature_dim, d_model)

        # Learnable positional embeddings: e^(pos)_ℓ (Eq. 7)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        # Optional series ID embedding: e^(id)_i (Eq. 7)
        if use_series_embedding:
            self.series_embedding = nn.Embedding(num_series, d_model)

    def forward(
        self,
        values: torch.Tensor,
        time_features: torch.Tensor,
        series_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            values: (batch_size, seq_len) - normalized time series values
            time_features: (batch_size, seq_len, time_feature_dim)
            series_ids: (batch_size,) - series identifiers

        Returns:
            Token embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = values.shape

        # Value embedding
        value_emb = self.value_embedding(values.unsqueeze(-1))  # (B, L, d)

        # Time feature embedding
        time_emb = self.time_embedding(time_features)  # (B, L, d)

        # Positional embedding
        positions = torch.arange(seq_len, device=values.device).unsqueeze(0)  # (1, L)
        pos_emb = self.positional_embedding(positions)  # (1, L, d)

        # Combine embeddings (Eq. 8)
        token_emb = value_emb + time_emb + pos_emb

        # Add series embedding if enabled
        if self.use_series_embedding and series_ids is not None:
            series_emb = self.series_embedding(series_ids)  # (B, d)
            token_emb = token_emb + series_emb.unsqueeze(1)  # (B, L, d)

        return token_emb


class CausalMultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention (Section 2.5).
    Implements Eq. 9-11 with causal masking.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads  # Head dimension

        # Query, Key, Value projections for all heads (combined)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) - True for valid positions

        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, d_h)

        # Scaled dot-product attention (Eq. 10)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)
        # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply causal mask (Eq. 9: M_{ab} = -∞ if b > a)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Apply padding mask if provided
        if mask is not None:
            # mask shape: (batch_size, seq_len)
            # Expand for attention: (batch_size, 1, 1, seq_len)
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len, d_h)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.W_o(out)

        return out


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network (FFN)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with causal self-attention.
    Implements Eq. 12-13 in the paper.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = CausalMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len)

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Self-attention with residual and layer norm (Eq. 12)
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # FFN with residual and layer norm (Eq. 13)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class MultiQuantileForecastHead(nn.Module):
    """
    Multi-horizon, multi-quantile forecast head (Section 2.6).
    Implements Eq. 14-15.
    """

    def __init__(
        self,
        d_model: int,
        max_horizon: int,
        quantiles: list,
        hidden_dim: int = 256,
        enforce_monotonicity: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            max_horizon: Maximum forecast horizon
            quantiles: List of quantile levels τ (e.g., [0.1, 0.5, 0.9])
            hidden_dim: Hidden dimension of the MLP
            enforce_monotonicity: If True, enforce no quantile crossing
        """
        super().__init__()

        self.d_model = d_model
        self.max_horizon = max_horizon
        self.quantiles = sorted(quantiles)
        self.num_quantiles = len(quantiles)
        self.enforce_monotonicity = enforce_monotonicity

        # Horizon embedding: e^(h)_h = W_h * φ(h) + b_h (Eq. 14)
        self.horizon_embedding = nn.Embedding(max_horizon, d_model)

        # MLP for quantile prediction: g_θ (Eq. 14)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_quantiles)
        )

    def forward(
        self,
        h_ctx: torch.Tensor,
        horizons: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_ctx: Context summary (batch_size, d_model) - last token from encoder
            horizons: (batch_size, H) - horizon indices [0, 1, ..., H-1]

        Returns:
            Predicted quantiles: (batch_size, H, num_quantiles)
        """
        batch_size, H = horizons.shape

        # Get horizon embeddings
        horizon_emb = self.horizon_embedding(horizons)  # (B, H, d)

        # Expand context summary
        h_ctx_exp = h_ctx.unsqueeze(1).expand(-1, H, -1)  # (B, H, d)

        # Combine context and horizon embeddings
        combined = h_ctx_exp + horizon_emb  # (B, H, d)

        # Predict quantiles via MLP
        quantiles_pred = self.mlp(combined)  # (B, H, Q)

        # Enforce monotonicity if requested (Eq. 15 variant)
        if self.enforce_monotonicity:
            quantiles_pred = self._enforce_monotonicity(quantiles_pred)

        return quantiles_pred

    def _enforce_monotonicity(self, raw_quantiles: torch.Tensor) -> torch.Tensor:
        """
        Enforce quantile monotonicity: q_τ1 ≤ q_τ2 ≤ ... ≤ q_τQ
        Using cumulative sum of softplus (Eq. 15 in paper)
        """
        batch_size, H, Q = raw_quantiles.shape

        # First quantile is unconstrained
        monotonic = torch.zeros_like(raw_quantiles)
        monotonic[:, :, 0] = raw_quantiles[:, :, 0]

        # Subsequent quantiles are cumulative softplus
        for i in range(1, Q):
            monotonic[:, :, i] = monotonic[:, :, i - 1] + F.softplus(raw_quantiles[:, :, i])

        return monotonic


class MaskedReconstructionHead(nn.Module):
    """Head for masked time series reconstruction (Section 2.7)"""

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)

        Returns:
            Reconstructed values: (batch_size, seq_len)
        """
        return self.projection(hidden_states).squeeze(-1)


class TimeGPT(nn.Module):
    """
    TimeGPT: Transformer Foundation Model for Time Series Forecasting
    Implements the complete architecture from the paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        max_horizon: int = 96,
        time_feature_dim: int = 8,
        num_series: int = 100,
        quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9],
        dropout: float = 0.1,
        use_series_embedding: bool = True,
        use_masked_reconstruction: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_layers: Number of Transformer layers (L_tr in paper)
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            max_seq_len: Maximum sequence length for positional embeddings
            max_horizon: Maximum forecast horizon
            time_feature_dim: Dimension of time features
            num_series: Number of unique series
            quantiles: List of quantile levels for forecasting
            dropout: Dropout rate
            use_series_embedding: Whether to use series-specific embeddings
            use_masked_reconstruction: Whether to include masked reconstruction head
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_masked_reconstruction = use_masked_reconstruction

        # Tokenization / Embedding layer
        self.embedding = TimeSeriesEmbedding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            time_feature_dim=time_feature_dim,
            num_series=num_series,
            use_series_embedding=use_series_embedding
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Forecast head
        self.forecast_head = MultiQuantileForecastHead(
            d_model=d_model,
            max_horizon=max_horizon,
            quantiles=quantiles,
            hidden_dim=d_ff // 2
        )

        # Optional masked reconstruction head
        if use_masked_reconstruction:
            self.reconstruction_head = MaskedReconstructionHead(d_model)

        self.quantiles = quantiles

    def forward(
        self,
        y_ctx: torch.Tensor,
        time_features_ctx: torch.Tensor,
        horizons: torch.Tensor,
        series_ids: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> dict:
        """
        Forward pass for forecasting.

        Args:
            y_ctx: (batch_size, L) - normalized context values
            time_features_ctx: (batch_size, L, time_feature_dim)
            horizons: (batch_size, H) - horizon indices to predict
            series_ids: (batch_size,) - series identifiers
            ctx_mask: (batch_size, L) - mask for valid positions
            return_hidden: If True, return hidden states for reconstruction

        Returns:
            Dictionary containing:
                - quantile_forecasts: (batch_size, H, num_quantiles)
                - hidden_states: (batch_size, L, d_model) if return_hidden=True
        """
        # Tokenize time series (Section 2.4)
        token_embeddings = self.embedding(y_ctx, time_features_ctx, series_ids)
        # Shape: (batch_size, L, d_model)

        # Pass through Transformer encoder (Section 2.5)
        hidden = token_embeddings
        for layer in self.layers:
            hidden = layer(hidden, ctx_mask)
        # Shape: (batch_size, L, d_model)

        # Extract context summary from last token (Eq. 14)
        h_ctx = hidden[:, -1, :]  # (batch_size, d_model)

        # Generate forecasts (Section 2.6)
        quantile_forecasts = self.forecast_head(h_ctx, horizons)
        # Shape: (batch_size, H, num_quantiles)

        output = {'quantile_forecasts': quantile_forecasts}

        if return_hidden:
            output['hidden_states'] = hidden

        return output

    def forward_with_masking(
        self,
        y_ctx: torch.Tensor,
        time_features_ctx: torch.Tensor,
        horizons: torch.Tensor,
        mask_indices: torch.Tensor,
        series_ids: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass with masked reconstruction for self-supervised learning.

        Args:
            y_ctx: (batch_size, L) - normalized context values
            time_features_ctx: (batch_size, L, time_feature_dim)
            horizons: (batch_size, H) - horizon indices
            mask_indices: (batch_size, L) - binary mask (1 = masked position)
            series_ids: (batch_size,)
            ctx_mask: (batch_size, L)

        Returns:
            Dictionary containing:
                - quantile_forecasts: (batch_size, H, num_quantiles)
                - reconstructions: (batch_size, L) - reconstructed values at masked positions
        """
        # Apply masking to input
        y_ctx_masked = y_ctx.clone()
        y_ctx_masked[mask_indices.bool()] = 0  # Replace masked values with 0 (or learnable mask token)

        # Forward pass with reconstruction
        output = self.forward(
            y_ctx_masked,
            time_features_ctx,
            horizons,
            series_ids,
            ctx_mask,
            return_hidden=True
        )

        # Reconstruct masked positions
        if self.use_masked_reconstruction:
            reconstructions = self.reconstruction_head(output['hidden_states'])
            output['reconstructions'] = reconstructions
            output['mask_indices'] = mask_indices

        return output
