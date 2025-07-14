"""
Hierarchical Multi-Band LSTM Training Script

This module provides modular training functions for the HierForecastNet model,
designed for hierarchical time series forecasting at daily, weekly, and monthly scales.

The module contains separated, reusable components for:
- Hierarchical loss function (HierarchicalWRMSSE)
- Training configuration management (TrainingConfig)
- Modular trainer class (HierarchicalTrainer)
- Bootstrap functions for FIFO initialization
- Backward-compatible training function

Author: Refactored for modularity and maintainability
"""

from __future__ import annotations
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


sys.path.append(str(Path(__file__).parent))
from model import HierForecastNet, DEFAULT_HIDDEN_SIZE
from timeseries_datamodule import merge_calendars, build_loaders


@dataclass
class TrainingConfig:
    """Configuration class for hierarchical forecasting training."""
    
    # Data Configuration
    data_file: str = "processed_data/calendar_scaled.parquet"
    batch_size: int = 64
    
    # Model Configuration
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    
    # Training Configuration
    epochs: int = 40
    learning_rate: float = 2e-4
    gradient_clip_norm: float = 1.0
    
    # Loss Configuration
    daily_weight: float = 0.1
    weekly_weight: float = 0.3
    monthly_weight: float = 0.6
    
    # Reproducibility Configuration
    seed: int = 42
    
    # Checkpoint Configuration
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 5
    
    # Device Configuration
    device: Optional[str] = None
    
    # Logging Configuration
    verbose: bool = True


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for training."""
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class HierarchicalWRMSSE(nn.Module):
    """
    Hierarchical Weighted Root Mean Squared Scaled Error (WRMSSE) loss function.
    
    This loss function computes RMSSE for daily, weekly, and monthly predictions
    and combines them using weighted averages. The RMSSE normalizes prediction
    errors by the historical naive forecast error, making it scale-invariant.
    
    Args:
        daily_weight (float): Weight for daily prediction loss
        weekly_weight (float): Weight for weekly prediction loss  
        monthly_weight (float): Weight for monthly prediction loss
        epsilon (float): Small constant to avoid division by zero
        
    Example:
        >>> loss_fn = HierarchicalWRMSSE(daily_weight=0.1, weekly_weight=0.3, monthly_weight=0.6)
        >>> # During training:
        >>> loss = loss_fn(daily_pred, weekly_pred, monthly_pred,
        ...                daily_true, weekly_true, monthly_true,
        ...                daily_insample, weekly_insample, monthly_insample)
    """
    
    def __init__(self, daily_weight: float = 0.1, weekly_weight: float = 0.3, 
                 monthly_weight: float = 0.6, epsilon: float = 1e-8) -> None:
        super().__init__()
        
        if daily_weight < 0 or weekly_weight < 0 or monthly_weight < 0:
            raise ValueError("All weights must be non-negative")
        if daily_weight + weekly_weight + monthly_weight == 0:
            raise ValueError("At least one weight must be positive")
            
        self.daily_weight = daily_weight
        self.weekly_weight = weekly_weight  
        self.monthly_weight = monthly_weight
        self.epsilon = epsilon
        
        # Normalize weights
        total_weight = daily_weight + weekly_weight + monthly_weight
        self.daily_weight /= total_weight
        self.weekly_weight /= total_weight
        self.monthly_weight /= total_weight
        
    def _compute_rmsse(self, predictions: torch.Tensor, targets: torch.Tensor, 
                      insample_data: torch.Tensor) -> torch.Tensor:
        """
        Compute Root Mean Squared Scaled Error for a single scale.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): True values
            insample_data (torch.Tensor): Historical data for scaling
            
        Returns:
            torch.Tensor: RMSSE value
        """
        # Compute mean squared error of predictions
        prediction_mse = torch.mean((predictions - targets) ** 2, dim=-1)
        
        # Compute naive forecast error (using seasonal naive with lag=1)
        if insample_data.dim() > 1:
            naive_diff = insample_data[:, 1:] - insample_data[:, :-1]
        else:
            naive_diff = insample_data[1:] - insample_data[:-1]
            
        naive_mse = torch.mean(naive_diff ** 2, dim=-1)
        
        # Compute RMSSE with epsilon for numerical stability
        rmsse = torch.sqrt(prediction_mse / (naive_mse + self.epsilon))
        
        return rmsse
        
    def forward(self, daily_pred: torch.Tensor, weekly_pred: torch.Tensor, monthly_pred: torch.Tensor,
                daily_true: torch.Tensor, weekly_true: torch.Tensor, monthly_true: torch.Tensor,
                daily_insample: torch.Tensor, weekly_insample: torch.Tensor, monthly_insample: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchical WRMSSE loss.
        
        Args:
            daily_pred, weekly_pred, monthly_pred: Model predictions at different scales
            daily_true, weekly_true, monthly_true: True values at different scales
            daily_insample, weekly_insample, monthly_insample: Historical data for scaling
            
        Returns:
            torch.Tensor: Weighted combination of RMSSE losses
        """
        # Compute RMSSE for each scale
        daily_rmsse = self._compute_rmsse(daily_pred, daily_true, daily_insample)
        weekly_rmsse = self._compute_rmsse(weekly_pred, weekly_true, weekly_insample)
        monthly_rmsse = self._compute_rmsse(monthly_pred, monthly_true, monthly_insample)
        
        # Combine with weights
        total_loss = (
            self.daily_weight * daily_rmsse + 
            self.weekly_weight * weekly_rmsse + 
            self.monthly_weight * monthly_rmsse
        )
        
        return total_loss.mean()

def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def bootstrap_fifos(model: HierForecastNet, lookback: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bootstrap weekly and monthly FIFO queues from historical data.
    
    Args:
        model: The forecasting model
        lookback: Historical data (B, 365, F)
        
    Returns:
        week_fifo: (B, 7, hidden_dim)
        month_fifo: (B, 12, hidden_dim)
    """
    model.eval()
    B, _, _ = lookback.shape
    week_tokens: List[torch.Tensor] = []

    # Generate weekly tokens from sliding 14-day windows
    for start in range(0, 365-13, 7):
        x14 = lookback[:, start:start+14, :]
        _, _, wk0 = model.daily_encoder(x14)   # take most recent 7-day token
        week_tokens.append(wk0)

    week_fifo = torch.stack(week_tokens[-7:], dim=1)      # (B,7,H)

    # Generate processed weekly tokens → month FIFO
    month_tokens: List[torch.Tensor] = []
    wk_fifo = week_fifo.clone()
    for _ in range(12):
        _, wk_tok = model.weekly_encoder(wk_fifo)
        month_tokens.append(wk_tok)
        wk_fifo = torch.cat([wk_fifo[:,1:], wk_tok.unsqueeze(1)], 1)

    month_fifo = torch.stack(month_tokens[-12:], dim=1)   # (B,12,H)
    model.train()
    return week_fifo, month_fifo

class HierarchicalTrainer:
    """Modular trainer for hierarchical forecasting models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = get_device(config.device)
        
        # Set up reproducibility
        set_random_seeds(config.seed)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.train_loader = None
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.best_loss = float('inf')
        
    def setup_data(self) -> None:
        """Set up data loaders for training."""
        # Load and merge calendar data
        dataframe = merge_calendars(self.config.data_file)
        
        # Build data loaders
        train_loader, val_loader, test_loader = build_loaders(
            dataframe, batch_size=self.config.batch_size
        )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Get number of features
        feature_columns = [col for col in dataframe.columns if col != "date"]
        self.num_features = len(feature_columns)
            
    def setup_model(self) -> None:
        """Set up the model, loss function, and optimizer."""
        # Initialize model
        self.model = HierForecastNet(
            input_features=self.num_features,
            hidden_dim=self.config.hidden_size
        ).to(self.device)
        
        # Initialize loss function
        self.loss_function = HierarchicalWRMSSE(
            daily_weight=self.config.daily_weight,
            weekly_weight=self.config.weekly_weight,
            monthly_weight=self.config.monthly_weight
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
            
    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for lookback, daily_window, daily_target, weekly_target, monthly_target in progress_bar:
            # Move data to device
            lookback = lookback.to(self.device)
            daily_window = daily_window.to(self.device)
            daily_target = daily_target.to(self.device)
            weekly_target = weekly_target.to(self.device)
            monthly_target = monthly_target.to(self.device)
            
            # Bootstrap context FIFOs from lookback data
            weekly_fifo, monthly_fifo = bootstrap_fifos(self.model, lookback)
            
            # Forward pass
            daily_pred, weekly_pred, monthly_pred, _, _ = self.model(daily_window, weekly_fifo, monthly_fifo)
            
            # Prepare insample data for loss calculation
            daily_insample = lookback[:, -15:-1, -1]  # Last 14 days
            weekly_insample = lookback[:, -15:-1:7, -1]  # Weekly sampling
            monthly_insample = lookback[:, -365:, -1]  # Full year
            
            # Compute loss
            loss = self.loss_function(
                daily_pred, weekly_pred, monthly_pred,
                daily_target, weekly_target, monthly_target,
                daily_insample, weekly_insample, monthly_insample
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.train_loader)
        
    def train(self) -> None:
        """Execute the complete training pipeline."""
        # Setup components
        self.setup_data()
        self.setup_model()
        assert self.model is not None, "Model was not initialized. Call setup_model() before training."
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            avg_loss = self.train_epoch()
            self.train_losses.append(avg_loss)
            
            # Update best loss
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                
            # Print progress
            if self.config.verbose:
                print(f"Epoch {epoch + 1:>3}/{self.config.epochs} - Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_epochs == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
                
        # Save final checkpoint
        self.save_checkpoint(self.config.epochs, self.train_losses[-1])
        
    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint."""
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)
        checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch:03d}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)


# Backward compatibility: provide the old interface
def train():
    """Original training function for backward compatibility."""
    # Create default configuration matching original parameters
    config = TrainingConfig(
        data_file="processed_data/calendar_scaled.parquet",
        batch_size=64,
        epochs=40,
        learning_rate=2e-4,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        daily_weight=0.1,
        weekly_weight=0.3,
        monthly_weight=0.6,
        seed=42,
        checkpoint_dir="checkpoints",
        save_every_epochs=1,
        verbose=True
    )
    
    # Create and run trainer
    trainer = HierarchicalTrainer(config)
    trainer.train()


