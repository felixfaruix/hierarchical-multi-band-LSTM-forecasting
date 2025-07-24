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
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

hidden_size = 128
dropout_rate = 0.01
attention_heads = 4
chunk_size = 7
daily_window = 14
month_steps = 30


class FeatureAttention(nn.Module):
    """
    Feature attention mechanism that applies soft attention over features within a single day.

    This module computes attention weights for input features and returns a weighted sum,
    allowing the model to focus on the most relevant features for the current time step.

    Args:
        in_dim (int): Input dimension (number of features).
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, in_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feature attention to input tensor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size*seq_len, feature vector size)
            
        """
        features_logits = self.score(x)  # Shape: (batch_size*seq_len, in_dim)
        attention_weights = torch.softmax(features_logits, dim=1)
        
        return (attention_weights * x)

class ChunkAttentionPool(nn.Module):
    """
    Chunk attention pooling layer that applies multi-head attention over a sequence chunk.
    
    This module uses a learnable query to pool information from a sequence chunk using
    multi-head attention, producing a single representation token for the chunk.
    
    Args:
        hidden_dim (int): Hidden dimension size.
        num_heads (int, optional): Number of attention heads. Defaults to DEFAULT_ATTENTION_HEADS.
        dropout_rate (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT_RATE.
    """

    def __init__(self, hidden_dim: int, num_heads: int = attention_heads, dropout_rate: float = dropout_rate) -> None:
        super().__init__()
            
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.multi_head_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate, batch_first=True)
        
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
    """

    def __init__(self, dim: int, sub_module: nn.Module, dropout_rate: float = dropout_rate) -> None:
        super().__init__()

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
        dropout_rate (float, optional): Dropout rate.

    Returns: 
        Tensor: Output tensor of same shape as input.
    """

    def __init__(self, dim: int, dropout_rate: float = dropout_rate) -> None:
        super().__init__()

        self.feed_forward = nn.Sequential(nn.Linear(dim, dim * 2), nn.GLU(), nn.Dropout(dropout_rate), nn.Linear(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply GLU feed-forward network.
        I learns gates to control information flow. We feed the input through a linear layer [shape: (batch_size, dim)],
        and we apply GLU to produce the output [shape: (batch_size, dim * 2)]. 
        Then we apply another linear layer to reduce the output back to the original dimension.
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
    """
    def __init__(self, input_features: int, hidden_dim: int, dropout_rate: float = dropout_rate) -> None:
        super().__init__()
            
        self.feature_projection = nn.Linear(input_features, hidden_dim)
        self.daily_feature_attention = FeatureAttention(input_features)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.chunk_pooler = ChunkAttentionPool(hidden_dim)
        self.mu_daily_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())  # guarantees σ > 0

    def forward(self, x14: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        batch_size, _, num_features = x14.shape

        # Apply feature attention for each day
        # Here we are transforming the 3D tensor to a 2D one to feed it to the Feature Attention function
        # The latter is expecting a 2D tensor of shape (batch_size * daily_window, num_features)
        daily_features = x14.view(batch_size * daily_window, num_features)
        features_weighted = self.daily_feature_attention(daily_features).view(batch_size, daily_window, num_features) #back to a 3D tensor
        
        # Process with LSTM
        lstm_output, _ = self.lstm(features_weighted)
        last_state = lstm_output[:, -1, :]  # Last output for daily prediction (context vector)
        mu_daily = self.mu_daily_head(last_state).squeeze(1)
        sigma_daily = self.sigma_head(last_state).squeeze(1)  # Get σ for the last day
        # Generate weekly tokens using chunk attention pooling
        week1_token = self.chunk_pooler(lstm_output[:, 0:7])  # First 7 days
        week0_token = self.chunk_pooler(lstm_output[:, 7:14])  # Last 7 days

        return mu_daily, sigma_daily, week1_token, week0_token

class WeeklyEncoder(nn.Module):
    """
    Weekly encoder that processes weekly tokens to produce weekly predictions and new weekly tokens.
    This encoder takes a sequence of weeklyy tokens, applies cross-attention and LSTM processing,
    and produces:
    1. A 7-day prediction vector for the next week
    2. A new weekly token representing the processed week
    
    Args:
        hidden_dim (int): Hidden dimension size for all layers.
        num_heads (int, optional): Number of attention heads. Defaults to DEFAULT_ATTENTION_HEADS.
        dropout_rate (float, optional): Dropout rate. Defaults to default_dropout_rate.
    """
    def __init__(self, hidden_dim: int, num_heads: int = attention_heads, dropout_rate: float = dropout_rate) -> None:
        super().__init__()

        self.cross_attention = PreNormRes(hidden_dim, nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate, batch_first=True), 
                                          dropout_rate)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Feed-forward network to process the final LSTM output
        # This will apply a GLU-based feed-forward network to the last LSTM output
        # It allows the model to learn complex transformations of the input features
        # It ensures that the weekly token is a rich representation and we can use it for further processing
        self.feed_forward = PreNormRes(hidden_dim, GLUffn(hidden_dim, dropout_rate), dropout_rate)
        # Weekly prediction head to generate a 7-day vector from the processed weekly token
        self.mu_weekly_head = nn.Linear(hidden_dim, chunk_size) # 7-day prediction
        self.sigma_w_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, weekly_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Process weekly tokens to produce weekly prediction and new token.
        
        Args:
            weekly_tokens (Tensor): Input tensor of shape (batch_size, num_weeks, hidden_dim)
            
        Returns:
            tuple: (weekly_prediction, new_weekly_token)
                - weekly_prediction (Tensor): Shape (batch_size, 7) - prediction for next 7 days
                - new_weekly_token (Tensor): Shape (batch_size, hidden_dim) - processed weekly token
        """

        # Use last token as query for cross-attention
        query = weekly_tokens[:, -1:, :]  # Shape: (batch_size, 1, hidden_dim)
        
        # Apply cross-attention to infer context from weekly tokens
        # This allows the model to focus on relevant weekly information out of the sequence
        context, _ = self.cross_attention.sub_module(query, weekly_tokens, weekly_tokens)
        enhanced_q = query + context # residual fusion

        # Concatenating with original tokens and process with LSTM
        lstm_input = torch.cat([weekly_tokens, enhanced_q], dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        mu_weekly = self.mu_weekly_head(lstm_output[:, -1]).squeeze(1)
        sigma_weekly = self.sigma_w_head(lstm_output[:, -1]).squeeze(1)
        # Get the last token and apply feed-forward
        last_token = self.feed_forward(lstm_output[:, -1])

        # The new weekly token is the processed last token
        return mu_weekly, sigma_weekly, last_token

class MonthlyEncoder(nn.Module):
    """
    Monthly encoder that processes monthly tokens to produce a final monthly representation.   
    This encoder takes a sequence of monthly tokens, applies cross-attention and LSTM processing,
    and produces a final monthly token that can be used for monthly predictions.
    
    Args:
        hidden_dim (int): Hidden dimension size for all layers.
        num_heads (int, optional): Number of attention heads. Defaults to DEFAULT_ATTENTION_HEADS.
        dropout_rate (float, optional): Dropout rate. Defaults to default_dropout_rate.
    """
    def __init__(self, hidden_dim: int, num_heads: int = attention_heads, dropout_rate: float = dropout_rate) -> None:
        super().__init__()

        self.cross_attention = PreNormRes(hidden_dim, nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate, batch_first=True), 
                                          dropout_rate)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.feed_forward = PreNormRes(hidden_dim, GLUffn(hidden_dim, dropout_rate), dropout_rate)
        
    def forward(self, monthly_tokens: Tensor) -> Tensor:
        """
        Process monthly tokens to produce a final monthly representation.
        
        Args:
            monthly_tokens (Tensor): Input tensor of shape (batch_size, num_months, hidden_dim)
            
        Returns:
            Tensor: Final monthly token of shape (batch_size, hidden_dim)
        """
        
        # Use last token as query for cross-attention
        query_m = monthly_tokens[:, -1:, :]  # Shape: (batch_size, 1, hidden_dim)
        
        # Apply cross-attention to infer context from monthly tokens
        # This allows the model to focus on relevant monthly information out of the sequence
        # The context is a weighted sum of the monthly tokens based on the query
        context_m, _ = self.cross_attention.sub_module(query_m, monthly_tokens, monthly_tokens)
        enhanced_q_m = query_m + context_m # residual fusion

        # Concatenate with original tokens and process with LSTM
        lstm_m_input = torch.cat([monthly_tokens, enhanced_q_m], dim=1)
        lstm_m_output, _ = self.lstm(lstm_m_input)
        
        # Get the last token and apply feed-forward
        final_token = self.feed_forward(lstm_m_output[:, -1])

        return final_token

class MonthDecoder(nn.Module):
    """
    Monthly decoder that autoregressively generates daily values from a monthly token.
    This decoder takes a monthly token and generates a sequence of daily predictions
    using an LSTM in autoregressive mode. Each step uses the previous output as input.
    
    Args:
        hidden_dim (int): Hidden dimension size for the LSTM and linear layers.
        dropout_rate (float, optional): Dropout rate. Defaults to default_dropout_rate.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = dropout_rate) -> None:
        super().__init__()
            
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.mu_monthly = nn.Linear(hidden_dim, 1)
        self.sigma_monthly = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())  # guarantees σ > 0
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, monthly_token: Tensor, steps: int = month_steps) -> tuple[Tensor, Tensor]:
        """
        Generate daily sequence from monthly token using autoregressive decoding.
        
        Args:
            monthly_token (Tensor): Monthly token of shape (batch_size, hidden_dim)
            steps (int, optional): Number of daily steps to generate. Defaults to DEFAULT_MONTH_STEPS.
            
        Returns:
            Tensor: Generated daily sequence of shape (batch_size, steps)
        """

        batch_size, hidden_dim = monthly_token.shape
        
        # Initialize LSTM hidden states
        hidden_state = monthly_token.unsqueeze(0)  # (1, batch_size, hidden_dim)
        cell_state = torch.zeros_like(hidden_state)
        
        # Initialize input as zeros
        lstm_input = torch.zeros(batch_size, 1, hidden_dim, device=monthly_token.device)
        
        # Autoregressively generate daily values
        daily_outputs_mu = []
        daily_outputs_sigma = []
        for _ in range(steps):
            # LSTM forward pass
            lstm_output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
            
            # Generate daily value
            daily_value = self.mu_monthly(self.dropout(lstm_output))
            sigma_value = self.sigma_monthly(self.dropout(lstm_output))
            daily_outputs_mu.append(daily_value.squeeze(2))  # Remove feature dimension
            daily_outputs_sigma.append(sigma_value.squeeze(2))  # Remove feature dimension

            # Use output as next input
            lstm_input = lstm_output

        return torch.cat(daily_outputs_mu, dim=1), torch.cat(daily_outputs_sigma, dim=1)

class HierForecastNet(nn.Module):
    """
    Hierarchical Forecasting Network for multi-scale time series prediction.
    This is the main model that combines all encoders and decoders to perform hierarchical
    forecasting at daily, weekly, and monthly scales. The model processes a 14-day window
    of features and produces predictions at multiple time horizons.
    
    Architecture:
    1. DailyEncoder: Processes 14 days → daily prediction + weekly tokens
    2. WeeklyEncoder: Processes weekly tokens → weekly prediction + weekly token
    3. MonthlyEncoder: Processes monthly tokens → monthly token
    4. MonthDecoder: Monthly token → sequence of daily predictions

    Args:
        input_features (int): Number of input features per day.
        hidden_dim (int, optional): Hidden dimension size. Defaults to DEFAULT_HIDDEN_SIZE.
        dropout_rate (float, optional): Dropout rate. Defaults to default_dropout_rate.
    """  
    def __init__(self, input_features: int, hidden_dim: int = hidden_size, 
                 dropout_rate: float = dropout_rate) -> None:
        super().__init__()
            
        self.daily_encoder = DailyEncoder(input_features, hidden_dim, dropout_rate)
        self.weekly_encoder = WeeklyEncoder(hidden_dim, dropout_rate=dropout_rate)
        self.monthly_encoder = MonthlyEncoder(hidden_dim, dropout_rate=dropout_rate)
        self.monthly_decoder = MonthDecoder(hidden_dim, dropout_rate)
        
    def forward(self, x14: Tensor, week_fifo: Tensor, month_fifo: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform hierarchical forecasting across multiple time scales.
        
        Args:
            x14 (Tensor): 14-day feature window of shape (batch_size, 14, input_features)
            week_fifo (Tensor): Weekly token history of shape (batch_size, 7, hidden_dim)
            month_fifo (Tensor): Monthly token history of shape (batch_size, 12, hidden_dim)
            
        Returns:
            tuple: (day_prediction, week_prediction, month_prediction_sequence, week0_token, weekly_token)
                - day_prediction (Tensor): Next day prediction, shape (batch_size,)
                - week_prediction (Tensor): Next 7 days prediction, shape (batch_size, 7)
                - month_prediction_sequence (Tensor): Next 30 days prediction, shape (batch_size, 30)
                - week0_token (Tensor): Latest weekly token, shape (batch_size, hidden_dim)
                - weekly_token (Tensor): Processed weekly token, shape (batch_size, hidden_dim)
        """
        # Daily encoding: 14 days -> daily prediction + 2 weekly tokens
        mu_daily, sigma_daily, week1_token, week0_token = self.daily_encoder(x14)
        
        # Weekly encoding: update weekly FIFO and get weekly prediction
        weekly_input = torch.cat([week_fifo[:, 1:], week0_token.unsqueeze(1)], dim=1)
        mu_weekly, sigma_weekly, processed_weekly_token = self.weekly_encoder(weekly_input)

        # Monthly encoding: update monthly FIFO and get monthly token
        monthly_input = torch.cat([month_fifo[:, 1:], processed_weekly_token.unsqueeze(1)], dim=1)
        monthly_tokens = self.monthly_encoder(monthly_input)
        
        # Monthly decoding: monthly token -> 30-day sequence
        mu_monthly_sequence, sigma_monthly_sequence = self.monthly_decoder(monthly_tokens)
        # Ensure the output is in the correct shape
        return mu_daily, sigma_daily, mu_weekly, sigma_weekly, mu_monthly_sequence, sigma_monthly_sequence
