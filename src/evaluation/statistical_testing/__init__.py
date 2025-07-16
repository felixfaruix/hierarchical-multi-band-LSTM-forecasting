"""
Core statistical testing utilities for forecast comparison.

This module provides the fundamental building blocks for statistical testing
of forecasting models, including data validation and test result structures.
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings


@dataclass
class ForecastComparison:
    """Container for aligned forecast comparison data."""
    model1_errors: np.ndarray
    model2_errors: np.ndarray
    forecast_origins: np.ndarray
    forecast_horizon: int
    loss_function: Union[str, Callable]
    additional_info: Dict[str, Any]


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float]
    null_hypothesis: str
    alternative_hypothesis: str
    significance_level: float
    is_significant: bool
    forecast_horizon: int
    sample_size: int
    additional_info: Dict[str, Any]


def validate_forecast_alignment(errors1: np.ndarray, 
                               errors2: np.ndarray,
                               origins1: Optional[np.ndarray] = None,
                               origins2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate that forecasts are properly aligned by forecast origins.
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2  
        origins1: Forecast origin timestamps/indices for model 1
        origins2: Forecast origin timestamps/indices for model 2
        
    Returns:
        Tuple of aligned error arrays
        
    Raises:
        ValueError: If forecasts cannot be properly aligned
    """
    if len(errors1) != len(errors2):
        raise ValueError(f"Error arrays have different lengths: {len(errors1)} vs {len(errors2)}")
    
    # If no origins provided, assume they're already aligned
    if origins1 is None and origins2 is None:
        warnings.warn("No forecast origins provided - assuming arrays are pre-aligned")
        return errors1, errors2
    
    if origins1 is None or origins2 is None:
        raise ValueError("If providing origins, must provide for both models")
    
    if len(origins1) != len(errors1) or len(origins2) != len(errors2):
        raise ValueError("Origins must have same length as error arrays")
    
    # Find common forecast origins
    common_origins = np.intersect1d(origins1, origins2)
    
    if len(common_origins) == 0:
        raise ValueError("No common forecast origins found between models")
    
    if len(common_origins) < len(errors1) * 0.8:
        warnings.warn(f"Only {len(common_origins)}/{len(errors1)} origins align - significant data loss")
    
    # Align arrays by common origins
    mask1 = np.isin(origins1, common_origins)
    mask2 = np.isin(origins2, common_origins)
    
    # Sort by origins to ensure proper alignment
    sort_idx1 = np.argsort(origins1[mask1])
    sort_idx2 = np.argsort(origins2[mask2])
    
    aligned_errors1 = errors1[mask1][sort_idx1]
    aligned_errors2 = errors2[mask2][sort_idx2]
    
    return aligned_errors1, aligned_errors2


def extract_per_origin_losses(predictions: np.ndarray,
                            actuals: np.ndarray,
                            forecast_origins: np.ndarray,
                            loss_function: Union[str, Callable] = "squared") -> Dict[int, float]:
    """
    Extract per-origin scalar losses for detailed analysis.
    
    This is needed for proper statistical testing where we want to examine
    loss differentials at each forecast origin.
    
    Args:
        predictions: Model predictions [n_origins, horizon]
        actuals: Actual values [n_origins, horizon]
        forecast_origins: Forecast origin indices [n_origins]
        loss_function: Loss function to use
        
    Returns:
        Dictionary mapping origin -> scalar loss
    """
    from .loss_stats_functions import compute_loss_differential
    
    if predictions.ndim == 1:
        # Single-step forecasts
        errors = predictions - actuals
        losses = compute_loss_differential(errors, np.zeros_like(errors), loss_function)
        return dict(zip(forecast_origins, losses))
    else:
        # Multi-step forecasts - aggregate across horizon
        per_origin_losses = {}
        for i, origin in enumerate(forecast_origins):
            errors = predictions[i] - actuals[i]
            if callable(loss_function):
                loss = np.mean(loss_function(errors))
            elif loss_function == "squared":
                loss = np.mean(errors ** 2)
            elif loss_function == "absolute":
                loss = np.mean(np.abs(errors))
            else:
                # Use first element of compute_loss_differential for other losses
                loss_diff = compute_loss_differential(errors, np.zeros_like(errors), loss_function)
                loss = np.mean(loss_diff)
            
            per_origin_losses[origin] = loss
        
        return per_origin_losses
