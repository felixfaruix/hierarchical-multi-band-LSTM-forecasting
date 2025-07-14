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
import torch.distributions as td


sys.path.append(str(Path(__file__).parent))
from model import HierForecastNet, default_hidden_size
from torch.utils.data import DataLoader
from timeseries_datamodule import merge_calendars, build_loaders


@dataclass
class TrainingConfig:
    """Configuration class for hierarchical forecasting training.
    This class encapsulates all hyperparameters and settings required for training the HierForecastNet model.
    It can be easily extended or modified to adapt to different training scenarios.
    Attributes:
        data_file (str): Path to the preprocessed data file.
        batch_size (int): Number of samples per batch.
        hidden_size (int): Size of the hidden layers in the model.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        gradient_clip_norm (float): Maximum norm for gradient clipping.
        daily_weight (float): Weight for daily prediction loss in hierarchical loss function.
        weekly_weight (float): Weight for weekly prediction loss in hierarchical loss function.
        monthly_weight (float): Weight for monthly prediction loss in hierarchical loss function.
        seed (int): Random seed for reproducibility.
    """
    # Model and training parameters
    target_variable: str = "ethanol_scaled"
    # Data Configuration
    data_file: str = "processed_data/calendar_scaled.parquet"
    batch_size: int = 64
    
    # Model Configuration
    hidden_size: int = default_hidden_size
    # Training Configuration
    epochs: int = 40
    learning_rate: float = 2e-4
    gradient_clip_norm: float = 1.0
    # Loss Configuration
    daily_weight: float = 0.1
    weekly_weight: float = 0.3
    monthly_weight: float = 0.6
    lambda_wrmsse: float = 0.2   
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
    
def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """Proper scoring rule for N(μ,σ²)."""
    dist = td.Normal(mu, sigma)
    return -dist.log_prob(target).mean()


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
    """Modular trainer that receives its *dataloaders* from the caller."""

    def __init__(
        self,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        model: Optional[HierForecastNet] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """
        Initialize the trainer with configuration, data loaders, model, and optimizer.
        """
        self.cfg = config
        self.device = get_device(config.device)
        set_random_seeds(config.seed)

        # Dataloaders
        if train_loader is None:
            raise ValueError("train_loader must be provided")
        if val_loader is None:
            val_loader = DataLoader(train_loader.dataset, batch_size=config.batch_size, shuffle=False)
        if test_loader is None:
            test_loader = DataLoader(train_loader.dataset, batch_size=config.batch_size, shuffle=False)    

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Model
        # If no model is provided, infer the number of features from the first batch
        if model is None:
            # Infer *F* (feature‑dim) from a dataset sample: (lookback, daily, ...)
            sample = train_loader.dataset[0][0]  # -> torch.Size([365, F])
            num_features = sample.shape[-1]
            model = HierForecastNet(num_features, config.hidden_size)
        self.model = model.to(self.device)
        # Loss function
        self.loss_fn = HierarchicalWRMSSE(config.daily_weight, config.weekly_weight, 
                                          config.monthly_weight).to(self.device)
        # Optimizer
        self.optim = (optimizer if optimizer is not None
            else torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate))
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.best_loss = float('inf')

    def train_pipeline(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train() # Set model to training mode
        epoch_loss = 0.0

        progress_bar = tqdm(self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.cfg.epochs}",
            disable=not self.cfg.verbose,)
        # Iterate over batches
        for lookback, daily_window, daily_target, weekly_target, monthly_target in progress_bar:
            # Move data to device
            # We do this to ensure that all tensors are on the same device
            # Also, we convert targets to float32 for compatibility with the loss function
            lookback = lookback.to(self.device).float()
            daily_window = daily_window.to(self.device).float()
            daily_target = daily_target.to(self.device).float()
            weekly_target = weekly_target.to(self.device).float()
            monthly_target = monthly_target.to(self.device).float()
            
            # Bootstrap context FIFOs from lookback data
            weekly_fifo, monthly_fifo = bootstrap_fifos(self.model, lookback)
            # Forward pass through the model
            # The model returns (mu_daily, sigma_daily, mu_weekly, sigma_weekly, mu_monthly, sigma_monthly)
            daily_pred, sigma_daily, weekly_pred, sigma_weekly, monthly_pred, sigma_monthly = self.model(daily_window, weekly_fifo, monthly_fifo)
            # Compute NLL for each scale
            nll_daily = gaussian_nll(daily_pred, sigma_daily, daily_target)
            nll_weekly = gaussian_nll(weekly_pred, sigma_weekly, weekly_target)
            nll_monthly = gaussian_nll(monthly_pred, sigma_monthly, monthly_target)

            S = self.cfg.daily_weight + self.cfg.weekly_weight + self.cfg.monthly_weight
            loss_nll = (self.cfg.daily_weight * nll_daily + 
                        self.cfg.weekly_weight * nll_weekly + 
                        self.cfg.monthly_weight * nll_monthly) / S
            # Prepare insample data for loss calculation
            daily_insample = lookback[:, -15:-1, -1]  # Last 14 days
            weekly_insample = lookback[:, -15:-1:7, -1]  # Weekly sampling
            monthly_insample = lookback[:, -365:, -1]  # Full year

            # Compute loss
            loss_wr = self.loss_fn(daily_pred, weekly_pred, monthly_pred, daily_target, weekly_target, monthly_target, daily_insample, weekly_insample, monthly_insample)
            total_loss = loss_nll + self.cfg.lambda_wrmsse * loss_wr
            # Backward pass
            self.optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_norm)
            self.optim.step()
            epoch_loss += total_loss.item()

        return epoch_loss / len(self.train_loader)
        
    def train(self) -> None:
        """Execute the complete training pipeline."""
        # Training loop
        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch
            # Train for one epoch
            avg_loss = self.train_pipeline()
            self.train_losses.append(avg_loss)
            
            # Update best loss
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                
            # Print progress
            if self.cfg.verbose:
                print(f"Epoch {epoch + 1:>3}/{self.cfg.epochs} - Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
                
        # Save final checkpoint
        self.save_checkpoint(self.cfg.epochs, self.train_losses[-1])

    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint."""
        Path(self.cfg.checkpoint_dir).mkdir(exist_ok=True)
        checkpoint_path = Path(self.cfg.checkpoint_dir) / f"epoch_{epoch:03d}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)

