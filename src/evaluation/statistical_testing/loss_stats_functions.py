"""
Loss statistics function utilities for statistical testing.

This module provides various loss functions and related computations
for use in statistical testing of forecast accuracy.
"""

from typing import Union, Callable, Optional, Tuple
import numpy as np
import warnings


def compute_loss_differential(errors1: np.ndarray,
                             errors2: np.ndarray,
                             loss_function: Union[str, Callable] = "squared") -> np.ndarray:
    """
    Compute loss differential between two forecast error series.
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        loss_function: Loss function - "squared", "absolute", "pinball", or callable
        
    Returns:
        Array of loss differentials (loss1 - loss2)
    """
    if callable(loss_function):
        loss1 = loss_function(errors1)
        loss2 = loss_function(errors2)
    elif loss_function == "squared":
        loss1 = errors1 ** 2
        loss2 = errors2 ** 2
    elif loss_function == "absolute":
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    elif loss_function == "pinball":
        # Pinball loss for quantile forecasting (assuming median, tau=0.5)
        tau = 0.5
        loss1 = np.where(errors1 >= 0, tau * errors1, (tau - 1) * errors1)
        loss2 = np.where(errors2 >= 0, tau * errors2, (tau - 1) * errors2)
    elif loss_function == "wape":
        # Weighted Absolute Percentage Error
        # Note: This requires actual values, not just errors
        warnings.warn("WAPE requires actual values - using absolute error instead")
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
    
    return loss1 - loss2


def compute_long_run_variance_proper(loss_diff: np.ndarray,
                                   forecast_horizon: int,
                                   max_lag: Optional[int] = None) -> float:
    """
    Compute long-run variance with proper handling of forecast horizon.
    
    For h-step ahead forecasts, errors are serially correlated up to lag h-1.
    This function properly accounts for this in the Newey-West estimation.
    
    Args:
        loss_diff: Loss differential series
        forecast_horizon: Forecast horizon h
        max_lag: Maximum lag for Newey-West (default: max(h-1, rule-of-thumb))
        
    Returns:
        Long-run variance estimate
    """
    n = len(loss_diff)
    
    if max_lag is None:
        # For h-step ahead forecasts, we need at least h-1 lags
        rule_of_thumb = int(np.floor(4 * (n / 100) ** (2 / 9)))
        max_lag = max(forecast_horizon - 1, rule_of_thumb)
    
    # Ensure we don't exceed reasonable bounds
    max_lag = min(max_lag, n - 1, n // 4)
    
    # Center the series
    centered_diff = loss_diff - np.mean(loss_diff)
    
    # Compute sample variance (lag 0)
    variance = np.var(centered_diff, ddof=1)
    
    # Add autocovariances with Bartlett kernel weights
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        
        # Bartlett kernel weight
        weight = 1 - lag / (max_lag + 1)
        
        # Autocovariance at lag
        autocovariance = np.mean(centered_diff[:-lag] * centered_diff[lag:])
        
        # Add weighted autocovariance (factor of 2 for symmetric kernel)
        variance += 2 * weight * autocovariance
    
    return float(max(variance, 1e-10))  # Ensure positive variance


def harvey_leybourne_newbold_correction(dm_stat: float, 
                                      n: int, 
                                      forecast_horizon: int) -> Tuple[float, float]:
    """
    Apply proper Harvey-Leybourne-Newbold small-sample correction.
    
    The correct HLN correction from Harvey, Leybourne, and Newbold (1997):
    - Adjusts the DM statistic for small samples
    - Uses t-distribution with (n-1) degrees of freedom instead of normal
    
    Args:
        dm_stat: Original DM test statistic
        n: Sample size
        forecast_horizon: Forecast horizon
        
    Returns:
        Tuple of (corrected_statistic, degrees_of_freedom)
    """
    if n >= 50:
        # No correction needed for large samples
        return dm_stat, np.inf
    
    # HLN correction factor - proper formula from the paper
    # This accounts for the bias in small-sample variance estimation
    correction_factor = np.sqrt((n + 1 - 2 * forecast_horizon + forecast_horizon * (forecast_horizon - 1) / n) / n)
    
    corrected_stat = dm_stat * correction_factor
    degrees_of_freedom = n - 1
    
    return corrected_stat, degrees_of_freedom
