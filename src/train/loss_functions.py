"""
Hierarchical Forecasting Loss Functions

This module contains specialized loss functions for hierarchical time series forecasting,
particularly the HierarchicalWRMSSE (Weighted Root Mean Squared Scaled Error) loss function.

Key Features:
- Scale-invariant loss computation using RMSSE
- Hierarchical weighting across daily, weekly, monthly scales
- Proper handling of multi-horizon forecasts
- Numerical stability with epsilon regularization
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.distributions as td
from typing import Tuple, Union


class HierarchicalWRMSSE(nn.Module):
    """
    Hierarchical Weighted Root Mean Squared Scaled Error (WRMSSE) loss function.
    
    This loss function computes RMSSE for daily, weekly, and monthly predictions
    and combines them using weighted averages. The RMSSE normalizes prediction
    errors by the historical naive forecast error, making it scale-invariant.
    
    RMSSE is preferred over MSE/MAE for time series forecasting because:
    - It's scale-invariant (can compare series with different magnitudes)
    - It normalizes by historical forecast difficulty
    - It provides interpretable error measures
    
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
        
        # Normalize weights to ensure they sum to 1
        total_weight = daily_weight + weekly_weight + monthly_weight
        self.daily_weight = daily_weight / total_weight
        self.weekly_weight = weekly_weight / total_weight
        self.monthly_weight = monthly_weight / total_weight
        self.epsilon = epsilon
        
        print(f"HierarchicalWRMSSE initialized with weights:")
        print(f"  Daily: {self.daily_weight:.3f}")
        print(f"  Weekly: {self.weekly_weight:.3f}")
        print(f"  Monthly: {self.monthly_weight:.3f}")
        
    def _compute_rmsse(self, predictions: torch.Tensor, targets: torch.Tensor, 
                      insample_data: torch.Tensor) -> torch.Tensor:
        """
        Compute Root Mean Squared Scaled Error for a single scale.
        
        RMSSE = sqrt(MSE(predictions, targets) / MSE(naive_forecast, targets))
        where naive_forecast uses seasonal naive with lag=1.
        
        Args:
            predictions (torch.Tensor): Model predictions [batch_size, horizon]
            targets (torch.Tensor): True values [batch_size, horizon]
            insample_data (torch.Tensor): Historical data for scaling [batch_size, history_length]
            
        Returns:
            torch.Tensor: RMSSE value for each sample in batch [batch_size]
        """
        # Compute mean squared error of predictions
        prediction_mse = torch.mean((predictions - targets) ** 2, dim=-1)
        
        # Compute naive forecast error using seasonal naive (lag=1)
        # This represents the "baseline difficulty" of forecasting this series
        if insample_data.dim() > 1:
            naive_diff = insample_data[:, 1:] - insample_data[:, :-1]
        else:
            naive_diff = insample_data[1:] - insample_data[:-1]
            
        naive_mse = torch.mean(naive_diff ** 2, dim=-1)
        
        # Compute RMSSE with epsilon for numerical stability
        # The epsilon prevents division by zero when naive_mse is very small
        rmsse = torch.sqrt(prediction_mse / (naive_mse + self.epsilon))
        
        return rmsse
        
    def forward(self, 
                daily_pred: torch.Tensor, weekly_pred: torch.Tensor, monthly_pred: torch.Tensor,
                daily_true: torch.Tensor, weekly_true: torch.Tensor, monthly_true: torch.Tensor,
                daily_insample: torch.Tensor, weekly_insample: torch.Tensor, monthly_insample: torch.Tensor,
                return_individual_losses: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute hierarchical WRMSSE loss.
        
        Args:
            daily_pred, weekly_pred, monthly_pred: Model predictions at different scales
            daily_true, weekly_true, monthly_true: True values at different scales
            daily_insample, weekly_insample, monthly_insample: Historical data for scaling
            return_individual_losses: If True, return individual scale losses along with total
            
        Returns:
            torch.Tensor: Weighted combination of RMSSE losses
            or Tuple[total_loss, daily_rmsse, weekly_rmsse, monthly_rmsse] if return_individual_losses=True
        """
        # Compute RMSSE for each scale
        # We take the mean across the batch dimension to get a single loss value
        daily_rmsse = self._compute_rmsse(daily_pred, daily_true, daily_insample).mean()
        weekly_rmsse = self._compute_rmsse(weekly_pred, weekly_true, weekly_insample).mean()
        monthly_rmsse = self._compute_rmsse(monthly_pred, monthly_true, monthly_insample).mean()

        # Combine with learned weights
        total_loss = (self.daily_weight * daily_rmsse + 
                      self.weekly_weight * weekly_rmsse + 
                      self.monthly_weight * monthly_rmsse)
        
        if return_individual_losses:
            return total_loss, daily_rmsse, weekly_rmsse, monthly_rmsse
        else:
            return total_loss
    
    def get_weights(self) -> Tuple[float, float, float]:
        """Get the current weights for each scale."""
        return self.daily_weight, self.weekly_weight, self.monthly_weight


class GaussianNLL(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for probabilistic forecasting.
    
    This is a proper scoring rule for N(μ,σ²) distributions that encourages
    well-calibrated probabilistic predictions.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.
        
        Args:
            mu: Mean predictions [batch_size, horizon]
            sigma: Standard deviation predictions [batch_size, horizon]
            target: True values [batch_size, horizon]
            
        Returns:
            torch.Tensor: NLL loss
        """
        dist = td.Normal(mu, sigma)
        nll = -dist.log_prob(target)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standalone function for Gaussian Negative Log-Likelihood.
    
    Args:
        mu: Mean predictions
        sigma: Standard deviation predictions  
        target: True values
        
    Returns:
        torch.Tensor: Mean NLL loss
    """
    dist = td.Normal(mu, sigma)
    return -dist.log_prob(target).mean()


# Factory function for creating loss functions
def create_hierarchical_loss(daily_weight: float = 0.1, 
                           weekly_weight: float = 0.3, 
                           monthly_weight: float = 0.6,
                           loss_type: str = 'wrmsse') -> nn.Module:
    """
    Factory function for creating hierarchical loss functions.
    
    Args:
        daily_weight: Weight for daily predictions
        weekly_weight: Weight for weekly predictions
        monthly_weight: Weight for monthly predictions
        loss_type: Type of loss ('wrmsse' or 'gaussian_nll')
        
    Returns:
        nn.Module: Configured loss function
    """
    if loss_type.lower() == 'wrmsse':
        return HierarchicalWRMSSE(daily_weight, weekly_weight, monthly_weight)
    elif loss_type.lower() == 'gaussian_nll':
        return GaussianNLL()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Example usage and testing
    print("Hierarchical Loss Functions Module")
    print("=" * 40)
    
    # Create sample tensors for testing
    batch_size, daily_horizon, weekly_horizon, monthly_horizon = 32, 7, 4, 3
    history_length = 100
    
    # Sample predictions and targets
    daily_pred = torch.randn(batch_size, daily_horizon)
    weekly_pred = torch.randn(batch_size, weekly_horizon)
    monthly_pred = torch.randn(batch_size, monthly_horizon)
    
    daily_true = torch.randn(batch_size, daily_horizon)
    weekly_true = torch.randn(batch_size, weekly_horizon)
    monthly_true = torch.randn(batch_size, monthly_horizon)
    
    # Sample insample data
    daily_insample = torch.randn(batch_size, history_length)
    weekly_insample = torch.randn(batch_size, history_length // 7)
    monthly_insample = torch.randn(batch_size, history_length // 30)
    
    # Test HierarchicalWRMSSE
    print("Testing HierarchicalWRMSSE:")
    loss_fn = HierarchicalWRMSSE(daily_weight=0.2, weekly_weight=0.3, monthly_weight=0.5)
    
    total_loss = loss_fn(daily_pred, weekly_pred, monthly_pred,
                        daily_true, weekly_true, monthly_true,
                        daily_insample, weekly_insample, monthly_insample)
    
    print(f"Total WRMSSE Loss: {total_loss.item():.4f}")
    
    # Test with individual losses
    total_loss, daily_loss, weekly_loss, monthly_loss = loss_fn(
        daily_pred, weekly_pred, monthly_pred,
        daily_true, weekly_true, monthly_true,
        daily_insample, weekly_insample, monthly_insample,
        return_individual_losses=True
    )
    
    print(f"Individual losses:")
    print(f"  Daily RMSSE: {daily_loss.item():.4f}")
    print(f"  Weekly RMSSE: {weekly_loss.item():.4f}")
    print(f"  Monthly RMSSE: {monthly_loss.item():.4f}")
    print(f"  Weighted Total: {total_loss.item():.4f}")
    
    # Test factory function
    print("\nTesting factory function:")
    loss_fn2 = create_hierarchical_loss(0.1, 0.4, 0.5, 'wrmsse')
    weights = loss_fn2.get_weights()
    print(f"Factory-created loss weights: {weights}")
