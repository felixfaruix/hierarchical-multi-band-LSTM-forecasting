"""
Diebold-Mariano Test Implementation.

This module provides the improved Diebold-Mariano test for comparing
forecast accuracy between models with proper handling of forecast horizons
and small-sample corrections.
"""

from typing import Union, Callable, Optional
import numpy as np
import warnings
from scipy.stats import norm, t

from . import StatisticalTestResult, validate_forecast_alignment
from .loss_stats_functions import (
    compute_loss_differential, 
    compute_long_run_variance_proper, 
    harvey_leybourne_newbold_correction
)


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
