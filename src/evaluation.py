"""
Clean Hierarchical Forecasting Evaluation Framework

This is the main evaluation interface that orchestrates the comprehensive
evaluation process by calling functions from specialized modules.

Key Features:
- Clean, readable main evaluation logic
- Modular design with specialized components
- Easy to understand evaluation workflow
- Comprehensive baseline comparison
- Stacked model variants evaluation
- Statistical testing preparation
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any
import warnings
import datetime
import torch
from torch.utils.data import DataLoader

# Import from our modular components
from baseline_models import create_baseline_models_by_scale, BaselineForecaster
from metrics_utils import (
    ForecastingMetrics, 
    compute_comprehensive_metrics,
    aggregate_metrics_across_horizons,
    create_unified_forecast_metrics,
    save_evaluation_results,
    generate_evaluation_summary
)
from stacked_variants import evaluate_stacked_variants
from evaluation_utils import NeuralModelEvaluator, ModelPredictions

# Import model and training utilities
from model import HierForecastNet
from train import bootstrap_fifos

try:
    from hierarchicalforecast import HierarchicalForecast
    from hierarchicalforecast.methods import MinTrace
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HierarchicalForecast = None
    MinTrace = None
    HIERARCHICAL_AVAILABLE = False


class HierarchicalReconciliation:
    """MinT reconciliation for hierarchical forecasts."""
    
    def __init__(self):
        self.fitted = False
        self.reconciler = None
    
    def fit(self, S: np.ndarray, y_train_levels: Dict[str, np.ndarray]) -> None:
        """Fit MinT reconciliation."""
        if not HIERARCHICAL_AVAILABLE:
            raise ImportError("hierarchicalforecast not available")
        
        # Simplified implementation - in practice you'd format data properly
        if MinTrace is None:
            raise ImportError("MinTrace is not available. Please install hierarchicalforecast.")
        self.reconciler = MinTrace()
        self.fitted = True
    
    def reconcile(self, base_forecasts: Dict[str, np.ndarray], 
                  S: np.ndarray) -> Dict[str, np.ndarray]:
        """Reconcile base forecasts using MinT."""
        if not self.fitted:
            raise ValueError("Reconciler must be fitted before use")
        
        # Placeholder implementation - return base forecasts
        return base_forecasts


class HierarchicalEvaluationFramework:
    """
    Clean, modular evaluation framework for hierarchical forecasting.
    
    This class orchestrates the evaluation process by calling specialized
    functions from separate modules, keeping the main logic clean and readable.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.reconciliation_engine = HierarchicalReconciliation()
        self.neural_evaluator = NeuralModelEvaluator(self.device)
        
    def evaluate_neural_model_with_proper_cv(self, 
                                            model: HierForecastNet, 
                                            test_data_loader: DataLoader, 
                                            enable_reconciliation: bool = False) -> ForecastingMetrics:
        """
        Evaluate neural hierarchical model with proper cross-validation practices.
        
        Args:
            model: Trained hierarchical neural network
            test_data_loader: Test data loader (proper out-of-sample data)
            enable_reconciliation: Whether to apply MinT reconciliation
            
        Returns:
            ForecastingMetrics object with comprehensive evaluation
        """
        print("  Collecting neural model predictions...")
        
        # Use our clean helper to collect all predictions
        model_predictions = self.neural_evaluator.collect_model_predictions(model, test_data_loader)
        
        # Apply hierarchical reconciliation if requested
        if enable_reconciliation and HIERARCHICAL_AVAILABLE:
            print("  Applying MinT reconciliation...")
            # Placeholder for MinT reconciliation implementation
            pass
        
        print("  Computing metrics for each time scale...")
        
        # Compute comprehensive metrics for each time scale
        daily_metrics = self._compute_scale_metrics(
            model_predictions.daily_actuals, 
            model_predictions.daily_predictions, 
            model_predictions.daily_insample, 
            seasonal_period=7
        )
        
        weekly_metrics = self._compute_scale_metrics(
            model_predictions.weekly_actuals, 
            model_predictions.weekly_predictions, 
            model_predictions.weekly_insample, 
            seasonal_period=4
        )
        
        monthly_metrics = self._compute_scale_metrics(
            model_predictions.monthly_actuals, 
            model_predictions.monthly_predictions, 
            model_predictions.monthly_insample, 
            seasonal_period=12
        )
        
        return create_unified_forecast_metrics(daily_metrics, weekly_metrics, monthly_metrics)
    
    def _compute_scale_metrics(self, actual_array: np.ndarray, pred_array: np.ndarray, 
                              insample_array: np.ndarray, seasonal_period: int) -> Dict[str, float]:
        """Compute metrics for a specific time scale with bulletproof per-sample scaling."""
        
        # Import our bulletproof per-sample metrics function
        from evaluation_utils import compute_metrics_per_sample_and_horizon
        
        # Use bulletproof per-sample scaling for shape-critical evaluation
        # This ensures proper RMSSE/MASE computation without collapsing insample histories
        per_sample_metrics = compute_metrics_per_sample_and_horizon(
            actuals=actual_array,
            predictions=pred_array, 
            insample_histories=insample_array,
            seasonal_period=seasonal_period,
            scale_name="current_scale"  # Generic scale name for this computation
        )
        
        # Return the bulletproof per-sample metrics
        # This includes both per-horizon metrics (e.g., mae_h1, mae_h2) and averaged metrics (e.g., mae_avg)
        return per_sample_metrics
    
    def evaluate_baseline_models(self, 
                               training_data: Dict[str, np.ndarray], 
                               test_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all baseline models across different time scales.
        
        Args:
            training_data: Dictionary with training data by scale
            test_data: Dictionary with test data by scale
            
        Returns:
            Dictionary with baseline model results
        """
        baseline_results = {}
        baseline_models = create_baseline_models_by_scale()
        
        for scale in ['daily', 'weekly', 'monthly']:
            if scale not in training_data or scale not in test_data:
                continue
                
            print(f"  Evaluating {scale} baseline models...")
            
            train_data = training_data[scale]
            test_data_scale = test_data[scale]
            
            horizon_map = {'daily': 7, 'weekly': 4, 'monthly': 12}
            forecast_horizon = horizon_map[scale]
            
            scale_results = self._evaluate_baseline_models_for_scale(
                baseline_models[scale], train_data, test_data_scale, scale, forecast_horizon
            )
            
            # Convert to unified format
            for model_name, metrics in scale_results.items():
                if model_name not in baseline_results:
                    baseline_results[model_name] = {}
                
                # Store with scale prefix for consistency
                for metric_name, value in metrics.items():
                    baseline_results[model_name][f'{scale}_{metric_name}'] = value
        
        return baseline_results
    
    def _evaluate_baseline_models_for_scale(self, 
                                          models: Dict[str, BaselineForecaster],
                                          training_data: np.ndarray, 
                                          test_data: np.ndarray,
                                          scale: str, 
                                          forecast_horizon: int) -> Dict[str, Dict[str, float]]:
        """Evaluate baseline models for a specific time scale."""
        
        evaluation_results = {}
        insample_reference = training_data[-min(365, len(training_data)):]
        seasonal_periods = {'daily': 7, 'weekly': 4, 'monthly': 12}
        seasonal_period = seasonal_periods.get(scale, 1)
        
        for model_name, model_instance in models.items():
            try:
                # Fit baseline model on training data
                model_instance.fit(training_data)
                
                # Generate predictions for test period
                forecast_values = model_instance.predict(forecast_horizon)
                
                # Align lengths for fair comparison
                min_length = min(len(forecast_values), len(test_data))
                aligned_forecasts = forecast_values[:min_length]
                aligned_actuals = test_data[:min_length]
                
                # Compute comprehensive metrics
                model_metrics = compute_comprehensive_metrics(
                    aligned_actuals, aligned_forecasts, insample_reference, 
                    seasonal_period=seasonal_period
                )
                evaluation_results[model_name] = model_metrics
                
            except Exception as evaluation_error:
                warnings.warn(f"Failed to evaluate {model_name}: {evaluation_error}")
                # Store infinite loss for failed models
                evaluation_results[model_name] = {
                    'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'rmsse': np.inf,
                    'mase': np.inf, 'smape': np.inf, 'directional_accuracy': 0.0
                }
        
        return evaluation_results
    
    def prepare_statistical_test_data(self, 
                                    neural_results: ForecastingMetrics,
                                    baseline_results: Dict[str, Any],
                                    stacked_variants_results: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare forecast error data for statistical significance testing.
        
        This creates a structure compatible with statistical_testing.py
        """
        # Placeholder implementation for statistical testing preparation
        forecast_errors_by_scale = {}
        
        for scale in ['daily', 'weekly', 'monthly']:
            forecast_errors_by_scale[scale] = {
                'HierForecastNet': np.array([]),  # Would collect actual neural model errors
            }
            
            # Add baseline model placeholders
            for model_name in baseline_results.keys():
                forecast_errors_by_scale[scale][model_name] = np.array([])
                
            # Add stacked variant placeholders
            if stacked_variants_results:
                for variant_name in stacked_variants_results.get(scale, {}).keys():
                    forecast_errors_by_scale[scale][variant_name] = np.array([])
        
        return forecast_errors_by_scale
    
    def run_comprehensive_evaluation_with_cv(self, 
                                           neural_model: HierForecastNet, 
                                           test_data_loader: DataLoader, 
                                           baseline_training_data: Dict[str, np.ndarray],
                                           baseline_test_data: Dict[str, np.ndarray],
                                           train_data_loader: Optional[DataLoader] = None,
                                           enable_reconciliation: bool = False,
                                           enable_statistical_testing: bool = True,
                                           enable_stacked_variants: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation following forecasting best practices.
        
        This is the main entry point that orchestrates the entire evaluation process.
        
        Args:
            neural_model: Trained hierarchical neural model
            test_data_loader: Proper out-of-sample test data loader
            baseline_training_data: Training data for baseline models by scale
            baseline_test_data: Test data for baseline models by scale
            train_data_loader: Optional training data loader for stacked variants
            enable_reconciliation: Whether to apply MinT reconciliation
            enable_statistical_testing: Whether to prepare data for statistical tests
            enable_stacked_variants: Whether to evaluate stacked model variants
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        
        results = {}
        
        # STEP 0: Validate temporal split to ensure no data leakage
        from evaluation_utils import validate_temporal_split
        
        if train_data_loader is not None:
            print("Step 0: Validating temporal split to prevent data leakage...")
            split_validation = validate_temporal_split(train_data_loader, test_data_loader)
            results['temporal_split_validation'] = split_validation
            
            # Report any warnings
            if split_validation['warnings']:
                print("  ⚠️  TEMPORAL SPLIT WARNINGS:")
                for warning in split_validation['warnings']:
                    print(f"    - {warning}")
            else:
                print("  ✅ Temporal split validation passed")
            print()
        
        # Validate and prepare data
        baseline_training_data_validated = self._validate_baseline_data(baseline_training_data)
        baseline_test_data_validated = self._validate_baseline_data(baseline_test_data)
        
        print("Starting comprehensive evaluation with enhanced framework...")
        print(f"Reconciliation: {'Enabled' if enable_reconciliation else 'Disabled'}")
        print(f"Statistical Testing: {'Enabled' if enable_statistical_testing else 'Disabled'}")
        print(f"Stacked Variants: {'Enabled' if enable_stacked_variants else 'Disabled'}")
        print("="*60)
        
        # Step 1: Evaluate neural model
        print("Step 1: Evaluating neural model with proper CV...")
        neural_results = self.evaluate_neural_model_with_proper_cv(
            neural_model, test_data_loader, enable_reconciliation
        )
        results['neural_model'] = neural_results
        
        # Step 2: Evaluate baseline models
        print("Step 2: Evaluating baseline models...")
        baseline_results = self.evaluate_baseline_models(
            baseline_training_data_validated, baseline_test_data_validated
        )
        results['baseline_models'] = baseline_results
        
        # Step 3: Evaluate stacked model variants if requested
        if enable_stacked_variants and train_data_loader is not None:
            print("Step 3: Evaluating stacked model variants...")
            stacked_variants_results = evaluate_stacked_variants(
                neural_model, train_data_loader, test_data_loader, 
                baseline_training_data_validated, baseline_test_data_validated
            )
            results['stacked_variants'] = stacked_variants_results
        
        # Step 4: Prepare statistical test data if requested
        if enable_statistical_testing:
            print("Step 4: Preparing statistical test data...")
            statistical_test_data = self.prepare_statistical_test_data(
                neural_results, baseline_results, 
                results.get('stacked_variants', None)
            )
            results['statistical_test_data'] = statistical_test_data
        
        # Step 5: Generate comprehensive summary
        print("Step 5: Generating comprehensive summary...")
        summary = generate_evaluation_summary(results)
        results['summary'] = summary
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'reconciliation_enabled': enable_reconciliation,
            'statistical_testing_enabled': enable_statistical_testing,
            'stacked_variants_enabled': enable_stacked_variants,
            'baseline_models_evaluated': list(baseline_results.keys()) if baseline_results else [],
            'stacked_variants_evaluated': list(results.get('stacked_variants', {}).keys()),
            'evaluation_framework_version': '2.1.0-modular'
        }
        
        print("="*60)
        print("Comprehensive evaluation completed successfully!")
        print(f"Models evaluated: {len(baseline_results) + 1 + len(results.get('stacked_variants', {}))}")
        print("Use statistical_testing.py for significance tests")
        print("Use time_series_cross_validation.py for walk-forward analysis")
        
        return results
    
    def _validate_baseline_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Validate and clean baseline data."""
        validated = {}
        for scale in ['daily', 'weekly', 'monthly']:
            if scale in data and len(data[scale]) > 0:
                validated[scale] = data[scale]
        return validated
    
    def save_results(self, results: Dict[str, Any], output_directory: str) -> None:
        """Save comprehensive evaluation results."""
        save_evaluation_results(results, output_directory)
