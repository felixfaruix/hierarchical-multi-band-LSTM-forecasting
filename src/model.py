"""
Hierarchical Forecasting Model using Attention and RNNs with PyTorch.

This module implements a hierarchical forecasting model that predicts daily, weekly, 
and monthly values from time series data. It uses a combination of attention mechanisms 
and recurrent neural networks to process sequences of features and generate forecasts.

The model is divided into several components:
1. **FeatureAttention**: Soft-selects features within a single day using a linear layer
2. **ChunkAttentionPool**: Applies multi-head attention pooling over a 7-step chunk to produce a chunk token
3. **DailyEncoder**: Encodes a 14-day sequence into weekly tokens and a next-day forecast
4. **WeeklyEncoder**: Encodes weekly tokens to produce a next-week forecast and a weekly token
5. **MonthlyEncoder**: Encodes monthly tokens to produce a month token
6. **MonthDecoder**: Autoregressively decodes 30 daily values from the month token
7. **HierForecastNet**: The main model that combines all encoders and decoders

Between the components, there are utility layers like `PreNormRes` for residual connections 
with layer normalization, and `GLUffn` for feed-forward networks with gated linear units.

Example Usage:
    >>> import torch
    >>> from model import HierForecastNet
    >>> 
    >>> # Initialize model
    >>> model = HierForecastNet(in_f=10, hid=128, p=0.1)
    >>> 
    >>> # Create sample inputs
    >>> batch_size = 32
    >>> x14 = torch.randn(batch_size, 14, 10)  # 14 days of features
    >>> week_fifo = torch.randn(batch_size, 7, 128)  # 7 weekly tokens
    >>> month_fifo = torch.randn(batch_size, 12, 128)  # 12 monthly tokens
    >>> 
    >>> # Forward pass
    >>> day_pred, week_pred, month_pred_seq, wk0_tok, week_tok = model(x14, week_fifo, month_fifo)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Constants for better maintainability
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_CHUNK_SIZE = 7
DEFAULT_DAILY_WINDOW = 14
DEFAULT_MONTH_STEPS = 30


class FeatureAttention(nn.Module):
    """
    Feature attention mechanism that applies soft attention over features within a single day.
    
    This module computes attention weights for input features and returns a weighted sum,
    allowing the model to focus on the most relevant features for the current time step.
    
    Args:
        in_dim (int): Input dimension (number of features).
        
    Example:
        >>> attention = FeatureAttention(in_dim=10)
        >>> x = torch.randn(32, 5, 10)  # (batch, time_steps, features)
        >>> output = attention(x)  # (32, 5) - weighted features per time step
    """
    
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {in_dim}")
        self.score = nn.Linear(in_dim, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feature attention to input tensor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim)
            
        Returns:
            Tensor: Attention-weighted features of shape (batch_size, in_dim)
        """
        attention_weights = torch.softmax(self.score(x), dim=1)
        return (attention_weights * x).sum(dim=1)

class ChunkAttentionPool(nn.Module):
    """
    Chunk attention pooling layer that applies multi-head attention over a sequence chunk.
    
    This module uses a learnable query to pool information from a sequence chunk using
    multi-head attention, producing a single representation token for the chunk.
    
    Args:
        hidden_dim (int): Hidden dimension size.
        num_heads (int, optional): Number of attention heads. Defaults to DEFAULT_ATTENTION_HEADS.
        dropout_rate (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT_RATE.
        
    Example:
        >>> pool = ChunkAttentionPool(hidden_dim=128, num_heads=4)
        >>> chunk = torch.randn(32, 7, 128)  # (batch, seq_len, hidden_dim)
        >>> pooled = pool(chunk)  # (32, 128) - single token per chunk
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = DEFAULT_ATTENTION_HEADS, 
                 dropout_rate: float = DEFAULT_DROPOUT_RATE) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"Number of heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension {hidden_dim} must be divisible by num_heads {num_heads}")
            
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.multi_head_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        
    def forward(self, chunk: Tensor) -> Tensor:
        """
        Pool information from input chunk using attention.
        
        Args:
            chunk (Tensor): Input chunk of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tensor: Pooled representation of shape (batch_size, hidden_dim)
        """
        batch_size = chunk.size(0)
        query = self.query_token.expand(batch_size, -1, -1)
        context, _ = self.multi_head_attention(query, chunk, chunk)
        return context.squeeze(1)

class PreNormRes(nn.Module):
    """
    Pre-normalization residual connection wrapper.
    
    This module applies layer normalization before the sub-module, then adds a residual
    connection with dropout. This is a common pattern in transformer architectures.
    
    Args:
        dim (int): Dimension for layer normalization.
        sub_module (nn.Module): The sub-module to wrap.
        dropout_rate (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT_RATE.
        
    Example:
        >>> sub_layer = nn.Linear(128, 128)
        >>> pre_norm_layer = PreNormRes(128, sub_layer, dropout_rate=0.1)
        >>> x = torch.randn(32, 128)
        >>> output = pre_norm_layer(x)  # Applies norm -> sub_layer -> dropout -> residual
    """
    
    def __init__(self, dim: int, sub_module: nn.Module, 
                 dropout_rate: float = DEFAULT_DROPOUT_RATE) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self.layer_norm = nn.LayerNorm(dim)
        self.sub_module = sub_module
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Apply pre-normalization and residual connection.
        
        Args:
            x (Tensor): Input tensor.
            *args: Additional positional arguments for sub_module.
            **kwargs: Additional keyword arguments for sub_module.
            
        Returns:
            Tensor: Output with residual connection applied.
        """
        return x + self.dropout(self.sub_module(self.layer_norm(x), *args, **kwargs))

class GLUffn(nn.Module):
    """
    Gated Linear Unit Feed-Forward Network.
    
    This module implements a feed-forward network with Gated Linear Units (GLU),
    which helps with gradient flow and allows the model to control information flow.
    
    Args:
        dim (int): Input and output dimension.
        dropout_rate (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT_RATE.
        
    Example:
        >>> ffn = GLUffn(dim=128, dropout_rate=0.1)
        >>> x = torch.randn(32, 10, 128)
        >>> output = ffn(x)  # Same shape as input: (32, 10, 128)
    """
    
    def __init__(self, dim: int, dropout_rate: float = DEFAULT_DROPOUT_RATE) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply GLU feed-forward network.
        
        Args:
            x (Tensor): Input tensor of any shape (..., dim).
            
        Returns:
            Tensor: Output tensor of same shape as input.
        """
        return self.feed_forward(x)

class DailyEncoder(nn.Module):
    """
    Daily encoder that processes a 14-day sequence to produce daily predictions and weekly tokens.
    
    This encoder takes a 14-day window of features, applies feature attention for each day,
    processes the sequence with an LSTM, and produces:
    1. A prediction for the next day
    2. Two weekly tokens (chunk pooling over 7-day periods)
    
    Args:
        input_features (int): Number of input features per day.
        hidden_dim (int): Hidden dimension size for LSTM and projections.
        dropout_rate (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT_RATE.
        
    Example:
        >>> encoder = DailyEncoder(input_features=10, hidden_dim=128)
        >>> x14 = torch.randn(32, 14, 10)  # 14 days of features
        >>> day_pred, week1_token, week0_token = encoder(x14)
        >>> print(f"Day prediction shape: {day_pred.shape}")  # (32,)
        >>> print(f"Weekly token shape: {week1_token.shape}")  # (32, 128)
    """
    
    def __init__(self, input_features: int, hidden_dim: int, 
                 dropout_rate: float = DEFAULT_DROPOUT_RATE) -> None:
        super().__init__()
        if input_features <= 0:
            raise ValueError(f"Input features must be positive, got {input_features}")
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}")
            
        self.feature_projection = nn.Linear(input_features, hidden_dim)
        self.daily_feature_attention = FeatureAttention(input_features)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.chunk_pooler = ChunkAttentionPool(hidden_dim)
        self.daily_prediction_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x14: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Process 14-day sequence to produce daily prediction and weekly tokens.
        
        Args:
            x14 (Tensor): Input tensor of shape (batch_size, 14, input_features)
            
        Returns:
            tuple: (day_prediction, week1_token, week0_token)
                - day_prediction (Tensor): Shape (batch_size,) - prediction for next day
                - week1_token (Tensor): Shape (batch_size, hidden_dim) - first week token
                - week0_token (Tensor): Shape (batch_size, hidden_dim) - second week token
        """
        batch_size, seq_len, num_features = x14.shape
        if seq_len != DEFAULT_DAILY_WINDOW:
            raise ValueError(f"Expected sequence length {DEFAULT_DAILY_WINDOW}, got {seq_len}")
        
        # Apply feature attention for each day
        daily_features = x14.view(batch_size * DEFAULT_DAILY_WINDOW, num_features)
        attention_weights = self.daily_feature_attention(daily_features).view(batch_size, DEFAULT_DAILY_WINDOW, 1)
        
        # Project features and apply attention weights
        projected_features = self.feature_projection(x14) * attention_weights
        
        # Process with LSTM
        lstm_output, _ = self.lstm(projected_features)
        
        # Generate daily prediction from last timestep
        daily_prediction = self.daily_prediction_head(lstm_output[:, -1]).squeeze(1)
        
        # Generate weekly tokens using chunk attention pooling
        week1_token = self.chunk_pooler(lstm_output[:, 0:7])  # First 7 days
        week0_token = self.chunk_pooler(lstm_output[:, 7:14])  # Last 7 days
        
        return daily_prediction, week1_token, week0_token

class WeeklyEncoder(nn.Module):
    """7 weekly tokens → **7‑day vector** + new weekly token."""
    def __init__(self, hid: int, heads: int = 4, p: float = 0.1):
        super().__init__()
        self.cross_attn = PreNormRes(
            hid, nn.MultiheadAttention(hid, heads, dropout=p, batch_first=True), p)
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.ffn = PreNormRes(hid, GLUffn(hid, p), p)
        self.week_head = nn.Linear(hid, 7)  # <-- return 7‑vector
    def forward(self, wtokens: Tensor):
        q = wtokens[:, -1:, :]
        ctx, _ = self.cross_attn.sub(q, wtokens, wtokens)
        seq, _ = self.lstm(torch.cat([wtokens, ctx], dim=1))
        token = self.ffn(seq[:, -1])
        wk_pred = self.week_head(token)  # [B,7]
        return wk_pred, token

class MonthlyEncoder(nn.Module):
    def __init__(self, hid: int, heads: int = 4, p: float = 0.1):
        super().__init__()
        self.cross_attn = PreNormRes(
            hid, nn.MultiheadAttention(hid, heads, dropout=p, batch_first=True), p)
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.ffn = PreNormRes(hid, GLUffn(hid, p), p)
    def forward(self, mtokens: Tensor):
        q = mtokens[:, -1:, :]
        ctx, _ = self.cross_attn.sub(q, mtokens, mtokens)
        seq, _ = self.lstm(torch.cat([mtokens, ctx], dim=1))
        return self.ffn(seq[:, -1])

class MonthDecoder(nn.Module):
    def __init__(self, hid: int, p: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.out = nn.Linear(hid, 1)
        self.drop = nn.Dropout(p)
    def forward(self, token: Tensor, steps: int = 30):
        B, H = token.shape
        h = token.unsqueeze(0); c = torch.zeros_like(h)
        inp = torch.zeros(B, 1, H, device=token.device)
        outs = []
        for _ in range(steps):
            out, (h, c) = self.lstm(inp, (h, c))
            val = self.out(self.drop(out))
            outs.append(val.squeeze(2))
            inp = out
        return torch.cat(outs, dim=1)

# ---------- full model ----------
class HierForecastNet(nn.Module):
    def __init__(self, in_f: int, hid: int = 128, p: float = 0.1):
        super().__init__()
        self.daily_enc = DailyEncoder(in_f, hid, p)
        self.week_enc = WeeklyEncoder(hid, p=p)
        self.month_enc = MonthlyEncoder(hid, p=p)
        self.month_dec = MonthDecoder(hid, p)
    def forward(self, x14: Tensor, week_fifo: Tensor, month_fifo: Tensor):
        day_pred, wk1_tok, wk0_tok = self.daily_enc(x14)
        week_input = torch.cat([week_fifo[:, 1:], wk0_tok.unsqueeze(1)], dim=1)
        week_pred, week_tok = self.week_enc(week_input)
        month_input = torch.cat([month_fifo[:, 1:], week_tok.unsqueeze(1)], dim=1)
        month_token = self.month_enc(month_input)
        month_pred_seq = self.month_dec(month_token)
        return day_pred, week_pred, month_pred_seq, wk0_tok, week_tok
