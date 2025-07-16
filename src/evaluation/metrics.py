"""
Metrics and Results Utilities

This module contains all metrics computation functions and result handling
utilities for the hierarchical forecasting evaluation framework.
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ForecastingMetrics:
    """Container for comprehensive forecasting metrics across all time scales."""
    # Daily metrics
    daily_mae: float
    daily_rmse: float
    daily_mape: float
    daily_rmsse: float
    daily_mase: float
    daily_smape: float
    daily_directional_accuracy: float
    
    # Weekly metrics
    weekly_mae: float
    weekly_rmse: float
    weekly_mape: float
    weekly_rmsse: float
    weekly_mase: float
    weekly_smape: float
    weekly_directional_accuracy: float
    
    # Monthly metrics
    monthly_mae: float
    monthly_rmse: float
    monthly_mape: float
    monthly_rmsse: float
    monthly_mase: float
    monthly_smape: float
    monthly_directional_accuracy: float


def compute_comprehensive_metrics(actual_values: np.ndarray, 
                                predicted_values: np.ndarray, 
                                insample_data: np.ndarray, 
                                seasonal_period: int = 1,
                                prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                epsilon: float = 1e-8) -> Dict[str, float]:
    """
    Compute comprehensive forecasting metrics following M4/M5 competition standards.
    
    Args:
        actual_values: True values
        predicted_values: Predicted values
        insample_data: Historical in-sample data for scaling
        seasonal_period: Seasonal period for scaled error metrics
        prediction_intervals: Optional prediction intervals (lower, upper)
        epsilon: Small value for numerical stability
        
    Returns:
        Dictionary with comprehensive metrics
    """
    # Ensure arrays are numpy arrays
    y_true = np.asarray(actual_values).flatten()
    y_pred = np.asarray(predicted_values).flatten()
    insample = np.asarray(insample_data).flatten()
    
    # Basic error metrics
    absolute_errors = np.abs(y_true - y_pred)
    squared_errors = (y_true - y_pred) ** 2
    
    # Core metrics
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    
    # MAPE with zero protection
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    
    # Symmetric MAPE (better for values near zero)
    smape = np.mean(2 * absolute_errors / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
    
    # RMSSE (Root Mean Squared Scaled Error) - from M4 competition
    if len(insample) > seasonal_period:
        seasonal_naive_errors = insample[seasonal_period:] - insample[:-seasonal_period]
        seasonal_mse = np.mean(seasonal_naive_errors ** 2)
        rmsse = np.sqrt(np.mean(squared_errors) / (seasonal_mse + epsilon))
    else:
        naive_errors = insample[1:] - insample[:-1] if len(insample) > 1 else np.array([1.0])
        naive_mse = np.mean(naive_errors ** 2)
        rmsse = np.sqrt(np.mean(squared_errors) / (naive_mse + epsilon))
    
    # MASE (Mean Absolute Scaled Error) - scale-free alternative to MAPE
    if len(insample) > seasonal_period:
        seasonal_naive_mae = np.mean(np.abs(insample[seasonal_period:] - insample[:-seasonal_period]))
        mase = mae / (seasonal_naive_mae + epsilon)
    else:
        naive_mae = np.mean(np.abs(insample[1:] - insample[:-1])) if len(insample) > 1 else 1.0
        mase = mae / (naive_mae + epsilon)
    
    # Directional accuracy (sign prediction)
    if len(insample) > 0:
        actual_direction = np.sign(y_true - insample[-1])
        predicted_direction = np.sign(y_pred - insample[-1])
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    else:
        directional_accuracy = 50.0  # Random guess baseline
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'smape': float(smape),
        'rmsse': float(rmsse),
        'mase': float(mase),
        'directional_accuracy': float(directional_accuracy)
    }
    
    # Prediction interval coverage if provided
    if prediction_intervals is not None:
        lower_bound, upper_bound = prediction_intervals
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
        metrics['prediction_interval_coverage'] = float(coverage)
    
    return metrics


def aggregate_metrics_across_horizons(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple forecast horizons.
    
    Args:
        metrics_list: List of metric dictionaries from different horizons
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = [m.get(key, np.nan) for m in metrics_list]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            aggregated[key] = np.mean(valid_values)
        else:
            aggregated[key] = np.nan
    
    return aggregated


def create_unified_forecast_metrics(daily_metrics: Dict[str, float],
                                  weekly_metrics: Dict[str, float],
                                  monthly_metrics: Dict[str, float]) -> ForecastingMetrics:
    """
    Create a unified ForecastingMetrics object from individual scale metrics.
    
    Args:
        daily_metrics: Daily scale metrics
        weekly_metrics: Weekly scale metrics  
        monthly_metrics: Monthly scale metrics
        
    Returns:
        ForecastingMetrics object
    """
    return ForecastingMetrics(
        daily_mae=daily_metrics.get('mae', np.nan),
        daily_rmse=daily_metrics.get('rmse', np.nan),
        daily_mape=daily_metrics.get('mape', np.nan),
        daily_rmsse=daily_metrics.get('rmsse', np.nan),
        daily_mase=daily_metrics.get('mase', np.nan),
        daily_smape=daily_metrics.get('smape', np.nan),
        daily_directional_accuracy=daily_metrics.get('directional_accuracy', np.nan),
        
        weekly_mae=weekly_metrics.get('mae', np.nan),
        weekly_rmse=weekly_metrics.get('rmse', np.nan),
        weekly_mape=weekly_metrics.get('mape', np.nan),
        weekly_rmsse=weekly_metrics.get('rmsse', np.nan),
        weekly_mase=weekly_metrics.get('mase', np.nan),
        weekly_smape=weekly_metrics.get('smape', np.nan),
        weekly_directional_accuracy=weekly_metrics.get('directional_accuracy', np.nan),
        
        monthly_mae=monthly_metrics.get('mae', np.nan),
        monthly_rmse=monthly_metrics.get('rmse', np.nan),
        monthly_mape=monthly_metrics.get('mape', np.nan),
        monthly_rmsse=monthly_metrics.get('rmsse', np.nan),
        monthly_mase=monthly_metrics.get('mase', np.nan),
        monthly_smape=monthly_metrics.get('smape', np.nan),
        monthly_directional_accuracy=monthly_metrics.get('directional_accuracy', np.nan)
    )


def save_evaluation_results(results: Dict[str, Any], output_directory: str) -> None:
    """Save comprehensive evaluation results with proper organization."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save complete results as pickle
    with open(output_path / 'comprehensive_evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary as CSV for easy analysis
    save_evaluation_summary_csv(results, output_path / 'evaluation_summary.csv')
    
    # Save comparison report if available
    if 'summary' in results and 'best_models_by_metric' in results['summary']:
        save_comparison_report_csv(results['summary'], output_path / 'model_comparison_report.csv')
    
    print(f"Comprehensive evaluation results saved to: {output_path}")


def save_evaluation_summary_csv(results: Dict[str, Any], csv_path: Path) -> None:
    """Save evaluation summary as structured CSV."""
    summary_rows = []
    
    # Neural model results
    if 'neural_model' in results:
        neural_metrics = results['neural_model']
        if isinstance(neural_metrics, ForecastingMetrics):
            # Convert to dictionary for easier processing
            for attr in dir(neural_metrics):
                if not attr.startswith('_') and not callable(getattr(neural_metrics, attr)):
                    value = getattr(neural_metrics, attr)
                    parts = attr.split('_')
                    if len(parts) >= 2:
                        scale = parts[0]
                        metric = '_'.join(parts[1:])
                        summary_rows.append({
                            'model_type': 'Neural',
                            'model_name': 'HierForecastNet',
                            'time_scale': scale,
                            'metric': metric,
                            'value': value
                        })
    
    # Baseline model results
    if 'baseline_models' in results:
        for model_name, model_metrics in results['baseline_models'].items():
            if isinstance(model_metrics, dict):
                for metric_key, value in model_metrics.items():
                    if '_' in metric_key:
                        parts = metric_key.split('_')
                        scale = parts[0]
                        metric = '_'.join(parts[1:])
                        summary_rows.append({
                            'model_type': 'Baseline',
                            'model_name': model_name,
                            'time_scale': scale,
                            'metric': metric,
                            'value': value
                        })
    
    # Stacked variant results
    if 'stacked_variants' in results:
        for scale, variants in results['stacked_variants'].items():
            for variant_name, metrics in variants.items():
                for metric, value in metrics.items():
                    summary_rows.append({
                        'model_type': 'Stacked',
                        'model_name': variant_name,
                        'time_scale': scale,
                        'metric': metric,
                        'value': value
                    })
    
    # Create DataFrame and save
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(csv_path, index=False)
        print(f"Evaluation summary saved to: {csv_path}")


def save_comparison_report_csv(summary: Dict[str, Any], csv_path: Path) -> None:
    """Save model comparison report as CSV."""
    comparison_rows = []
    
    if 'best_models_by_metric' in summary:
        for scale, metrics in summary['best_models_by_metric'].items():
            for metric, info in metrics.items():
                comparison_rows.append({
                    'time_scale': scale,
                    'metric': metric,
                    'best_model': info.get('model', 'Unknown'),
                    'best_value': info.get('value', np.nan)
                })
    
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(csv_path, index=False)
        print(f"Model comparison report saved to: {csv_path}")


def generate_evaluation_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive summary of all evaluation results."""
    summary = {
        'best_models_by_metric': {},
        'performance_ranking': {},
        'metric_comparison': {}
    }
    
    # Collect all model results
    all_models = {}
    
    # Add neural model
    if 'neural_model' in results:
        all_models['HierForecastNet'] = results['neural_model']
        
    # Add baseline models
    if 'baseline_models' in results:
        all_models.update(results['baseline_models'])
        
    # Add stacked variants
    if 'stacked_variants' in results:
        for scale, variants in results['stacked_variants'].items():
            for variant_name, metrics in variants.items():
                model_key = f"{variant_name}_{scale}"
                all_models[model_key] = metrics
    
    # Find best models for each metric and scale
    metrics_to_compare = ['mae', 'rmse', 'mape', 'mase', 'smape']
    scales = ['daily', 'weekly', 'monthly']
    
    for scale in scales:
        summary['best_models_by_metric'][scale] = {}
        for metric in metrics_to_compare:
            metric_key = f'{scale}_{metric}'
            best_model = None
            best_value = float('inf')
            
            for model_name, model_results in all_models.items():
                value = None
                if hasattr(model_results, metric_key):
                    value = getattr(model_results, metric_key)
                elif isinstance(model_results, dict) and metric_key in model_results:
                    value = model_results[metric_key]
                elif isinstance(model_results, dict) and metric in model_results:
                    value = model_results[metric]
                
                if value is not None and not np.isnan(value) and value < best_value:
                    best_value = value
                    best_model = model_name
            
            if best_model:
                summary['best_models_by_metric'][scale][metric] = {
                    'model': best_model,
                    'value': best_value
                }
    
    return summary
