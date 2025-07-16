"""
Statistical Evaluation Module for Forecasting Models.

This module provides high-level interfaces for statistical testing
and evaluation of forecasting models.
"""

from typing import Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd

from .diebold_mariano import ImprovedDieboldMarianoTest
from . import StatisticalTestResult, ForecastComparison, extract_per_origin_losses


class StatisticalEvaluator:
    """
    High-level interface for statistical evaluation of forecasting models.
    """
    
    def __init__(self):
        """Initialize the statistical evaluator."""
        self.dm_test = ImprovedDieboldMarianoTest()
    
    def compare_models(self,
                      model1_errors: np.ndarray,
                      model2_errors: np.ndarray,
                      forecast_horizon: int,
                      model1_origins: Optional[np.ndarray] = None,
                      model2_origins: Optional[np.ndarray] = None,
                      test_type: str = "diebold_mariano",
                      loss_function: Union[str, Callable] = "squared",
                      significance_level: float = 0.05) -> StatisticalTestResult:
        """
        Compare two forecasting models using statistical tests.
        
        Args:
            model1_errors: Forecast errors from first model
            model2_errors: Forecast errors from second model
            forecast_horizon: Forecast horizon
            model1_origins: Forecast origins for model 1
            model2_origins: Forecast origins for model 2
            test_type: Type of statistical test ("diebold_mariano")
            loss_function: Loss function to use
            significance_level: Significance level for the test
            
        Returns:
            StatisticalTestResult object
        """
        if test_type == "diebold_mariano":
            self.dm_test.loss_function = loss_function
            return self.dm_test.test(
                forecast_errors_1=model1_errors,
                forecast_errors_2=model2_errors,
                forecast_horizon=forecast_horizon,
                forecast_origins_1=model1_origins,
                forecast_origins_2=model2_origins,
                significance_level=significance_level
            )
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    def pairwise_comparison(self,
                           model_errors: Dict[str, np.ndarray],
                           forecast_horizon: int,
                           model_origins: Optional[Dict[str, np.ndarray]] = None,
                           loss_function: Union[str, Callable] = "squared",
                           significance_level: float = 0.05) -> Dict[str, StatisticalTestResult]:
        """
        Perform pairwise comparisons between multiple models.
        
        Args:
            model_errors: Dictionary mapping model names to error arrays
            forecast_horizon: Forecast horizon
            model_origins: Dictionary mapping model names to origin arrays
            loss_function: Loss function to use
            significance_level: Significance level for tests
            
        Returns:
            Dictionary mapping comparison pairs to test results
        """
        results = {}
        model_names = list(model_errors.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                
                origins1 = model_origins.get(model1) if model_origins else None
                origins2 = model_origins.get(model2) if model_origins else None
                
                result = self.compare_models(
                    model1_errors=model_errors[model1],
                    model2_errors=model_errors[model2],
                    forecast_horizon=forecast_horizon,
                    model1_origins=origins1,
                    model2_origins=origins2,
                    loss_function=loss_function,
                    significance_level=significance_level
                )
                
                results[comparison_key] = result
        
        return results
    
    def model_ranking(self,
                     model_errors: Dict[str, np.ndarray],
                     forecast_horizon: int,
                     model_origins: Optional[Dict[str, np.ndarray]] = None,
                     loss_function: Union[str, Callable] = "squared",
                     significance_level: float = 0.05) -> pd.DataFrame:
        """
        Rank models based on pairwise comparisons.
        
        Args:
            model_errors: Dictionary mapping model names to error arrays
            forecast_horizon: Forecast horizon
            model_origins: Dictionary mapping model names to origin arrays
            loss_function: Loss function to use
            significance_level: Significance level for tests
            
        Returns:
            DataFrame with model rankings and win/loss statistics
        """
        pairwise_results = self.pairwise_comparison(
            model_errors, forecast_horizon, model_origins, 
            loss_function, significance_level
        )
        
        model_names = list(model_errors.keys())
        ranking_data = []
        
        for model in model_names:
            wins = 0
            losses = 0
            total_comparisons = 0
            
            for comparison, result in pairwise_results.items():
                if model in comparison and result.is_significant:
                    total_comparisons += 1
                    if comparison.startswith(model) and result.test_statistic < 0:
                        wins += 1  # Model 1 is better (lower loss)
                    elif comparison.endswith(model) and result.test_statistic > 0:
                        wins += 1  # Model 2 is better (lower loss)
                    else:
                        losses += 1
            
            win_rate = wins / total_comparisons if total_comparisons > 0 else 0
            
            ranking_data.append({
                'model': model,
                'wins': wins,
                'losses': losses,
                'total_comparisons': total_comparisons,
                'win_rate': win_rate,
                'mean_loss': np.mean(np.abs(model_errors[model]))
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values(['win_rate', 'mean_loss'], 
                                          ascending=[False, True])
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df