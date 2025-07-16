"""
Improved Statistical Testing for Forecasting Models

This module fixes the identified issues in statistical testing:
1. Proper handling of forecast horizon in long-run variance estimation
2. Correct Harvey-Leybourne-Newbold small-sample correction
3. Support for custom loss functions (callable)
4. Proper input alignment checking with forecast origins
5. Per-origin scalar loss extraction support

Key improvements:
- Explicit horizon parameter for DM test long-run variance
- Correct HLN correction formula from the published paper
- Support for custom loss functions beyond squared/absolute
- Forecast origin timestamp matching validation
- Per-fold and per-origin loss extraction utilities
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, norm, t
import warnings
from dataclasses import dataclass
from pathlib import Path
import itertools
from abc import ABC, abstractmethod


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


class ImprovedDieboldMarianoTest:
    """
    Improved Diebold-Mariano test addressing all identified issues.
    """
    
    def __init__(self, loss_function: Union[str, Callable] = "squared"):
        """
        Initialize the improved DM test.
        
        Args:
            loss_function: Loss function - "squared", "absolute", "pinball", or callable
        """
        self.loss_function = loss_function
    
    def test(self, 
             forecast_errors_1: np.ndarray,
             forecast_errors_2: np.ndarray,
             forecast_horizon: int,
             forecast_origins_1: Optional[np.ndarray] = None,
             forecast_origins_2: Optional[np.ndarray] = None,
             significance_level: float = 0.05,
             alternative: str = "two-sided",
             max_lag: Optional[int] = None) -> StatisticalTestResult:
        """
        Perform improved Diebold-Mariano test.
        
        Args:
            forecast_errors_1: Forecast errors from model 1
            forecast_errors_2: Forecast errors from model 2
            forecast_horizon: Forecast horizon h
            forecast_origins_1: Forecast origin timestamps for model 1
            forecast_origins_2: Forecast origin timestamps for model 2
            significance_level: Significance level for the test
            alternative: "two-sided", "greater", or "less"
            max_lag: Maximum lag for long-run variance (default: max(h-1, rule))
            
        Returns:
            StatisticalTestResult object
        """
        # Step 1: Validate and align forecasts
        aligned_errors_1, aligned_errors_2 = validate_forecast_alignment(
            forecast_errors_1, forecast_errors_2, 
            forecast_origins_1, forecast_origins_2
        )
        
        n = len(aligned_errors_1)
        
        if n < 10:
            warnings.warn(f"Sample size ({n}) is very small for DM test")
        
        # Step 2: Compute loss differential with proper loss function
        loss_diff = compute_loss_differential(
            aligned_errors_1, aligned_errors_2, self.loss_function
        )
        
        # Step 3: Compute test statistic with proper long-run variance
        mean_diff = np.mean(loss_diff)
        long_run_var = compute_long_run_variance_proper(
            loss_diff, forecast_horizon, max_lag
        )
        
        if long_run_var <= 0:
            warnings.warn("Non-positive long-run variance estimate")
            dm_stat = np.inf if mean_diff > 0 else -np.inf
            corrected_stat = dm_stat
            df = np.inf
        else:
            dm_stat = mean_diff / np.sqrt(long_run_var / n)
            
            # Step 4: Apply proper small-sample correction
            corrected_stat, df = harvey_leybourne_newbold_correction(
                dm_stat, n, forecast_horizon
            )
        
        # Step 5: Compute p-value using appropriate distribution
        if df == np.inf:
            # Large sample: use normal distribution
            if alternative == "two-sided":
                p_value = 2 * (1 - norm.cdf(np.abs(corrected_stat)))
                critical_value = norm.ppf(1 - significance_level / 2)
                is_significant = np.abs(corrected_stat) > critical_value
            elif alternative == "greater":
                p_value = 1 - norm.cdf(corrected_stat)
                critical_value = norm.ppf(1 - significance_level)
                is_significant = corrected_stat > critical_value
            else:  # less
                p_value = norm.cdf(corrected_stat)
                critical_value = norm.ppf(significance_level)
                is_significant = corrected_stat < critical_value
        else:
            # Small sample: use t-distribution
            if alternative == "two-sided":
                p_value = 2 * (1 - t.cdf(np.abs(corrected_stat), df))
                critical_value = t.ppf(1 - significance_level / 2, df)
                is_significant = np.abs(corrected_stat) > critical_value
            elif alternative == "greater":
                p_value = 1 - t.cdf(corrected_stat, df)
                critical_value = t.ppf(1 - significance_level, df)
                is_significant = corrected_stat > critical_value
            else:  # less
                p_value = t.cdf(corrected_stat, df)
                critical_value = t.ppf(significance_level, df)
                is_significant = corrected_stat < critical_value
        
        return StatisticalTestResult(
            test_name="Improved Diebold-Mariano Test",
            test_statistic=float(corrected_stat),
            p_value=float(p_value),
            critical_value=float(critical_value),
            null_hypothesis="Forecasts have equal accuracy",
            alternative_hypothesis=f"Model 1 accuracy {alternative} model 2",
            significance_level=significance_level,
            is_significant=bool(is_significant),
            forecast_horizon=forecast_horizon,
            sample_size=n,
            additional_info={
                "original_dm_statistic": float(dm_stat),
                "mean_loss_differential": float(mean_diff),
                "long_run_variance": float(long_run_var),
                "max_lag_used": max_lag or max(forecast_horizon - 1, 
                                              int(np.floor(4 * (n / 100) ** (2 / 9)))),
                "degrees_of_freedom": float(df),
                "small_sample_correction_applied": df != np.inf,
                "loss_function": str(self.loss_function),
                "alignment_method": "forecast_origins" if forecast_origins_1 is not None else "assumed_aligned"
            }
        )


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


if __name__ == "__main__":
    print("Improved Statistical Testing for Forecasting Models")
    print("=" * 55)
    print("Key improvements:")
    print("1. ✅ Proper forecast horizon handling in long-run variance")
    print("2. ✅ Correct Harvey-Leybourne-Newbold small-sample correction")
    print("3. ✅ Support for custom loss functions (callable)")
    print("4. ✅ Forecast origin alignment validation")
    print("5. ✅ Per-origin scalar loss extraction")
    print()
    print("Usage example:")
    print("  dm_test = ImprovedDieboldMarianoTest('squared')")
    print("  result = dm_test.test(errors1, errors2, forecast_horizon=7,")
    print("                       forecast_origins_1=origins1, forecast_origins_2=origins2)")
