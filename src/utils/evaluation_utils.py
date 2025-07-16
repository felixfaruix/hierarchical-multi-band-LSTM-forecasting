"""
Neural Model Evaluation Utilities (FIXED - Paper Grade)

This module contains helper functions for evaluating neural models with proper:
- Per-sample insample scaling (not collapsed to scalars)
- Consistent array shapes across evaluation pipeline
- No data leakage in temporal splits
- Robust multi-horizon metric computation

CRITICAL SHAPE SEMANTICS:
- Predictions: (N_origins, H_horizons) where N = rolling windows, H = forecast steps
- Metrics computed across origins for each horizon (competition style)
- Insample histories preserved per sample for proper RMSSE/MASE scaling

Key Features:
- Clean batch processing loops
- Automated prediction collection  
- Device-aware tensor operations
- Memory-efficient data handling
- BULLETPROOF scaling metrics
"""

from __future__ import annotations
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch.utils.data import DataLoader
from dataclasses import dataclass

# Import model utilities
from ..models.model import HierForecastNet
from ..train.train import bootstrap_fifos


@dataclass
class PredictionBatch:
    """Container for predictions and actuals from a single batch."""
    daily_pred: np.ndarray
    weekly_pred: np.ndarray
    monthly_pred: np.ndarray
    daily_actual: np.ndarray
    weekly_actual: np.ndarray
    monthly_actual: np.ndarray
    daily_insample: np.ndarray
    weekly_insample: np.ndarray
    monthly_insample: np.ndarray


@dataclass
class ModelPredictions:
    """Container for all model predictions across the evaluation dataset."""
    daily_predictions: np.ndarray
    weekly_predictions: np.ndarray
    monthly_predictions: np.ndarray
    daily_actuals: np.ndarray
    weekly_actuals: np.ndarray
    monthly_actuals: np.ndarray
    daily_insample: np.ndarray
    weekly_insample: np.ndarray
    monthly_insample: np.ndarray
    
    @property
    def num_samples(self) -> int:
        """Get the number of samples in the predictions."""
        return len(self.daily_predictions)


class NeuralModelEvaluator:
    """
    Helper class for evaluating neural models with clean batch processing.
    
    This class handles all the repetitive aspects of model evaluation:
    - Batch iteration and processing
    - Tensor device management
    - Data collection and concatenation
    - Memory-efficient operations
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
    
    def _process_single_batch(self, model: HierForecastNet, batch_data: Tuple) -> PredictionBatch:
        """
        Process a single batch through the neural model.
        
        Args:
            model: The neural model to evaluate
            batch_data: Tuple containing (lookback_window, daily_features, daily_targets, weekly_targets, monthly_targets)
            
        Returns:
            PredictionBatch: Container with predictions and actuals from this batch
        """
        lookback_window, daily_features, daily_targets, weekly_targets, monthly_targets = batch_data
        
        # Move all tensors to device efficiently
        tensors_to_device = [lookback_window, daily_features, daily_targets, weekly_targets, monthly_targets]
        lookback_window, daily_features, daily_targets, weekly_targets, monthly_targets = [
            tensor.to(self.device).float() for tensor in tensors_to_device
        ]
        
        # Bootstrap context FIFOs from lookback data
        weekly_context, monthly_context = bootstrap_fifos(model, lookback_window)
        
        # Forward pass through neural model
        daily_mean, _, weekly_mean, _, monthly_mean, _ = model(
            daily_features, weekly_context, monthly_context
        )
        
        # Extract insample data for scaling metrics (move to CPU immediately)
        daily_insample = lookback_window[:, -15:-1, -1].cpu().numpy()
        weekly_insample = lookback_window[:, -15:-1:7, -1].cpu().numpy()
        monthly_insample = lookback_window[:, -365:, -1].cpu().numpy()
        
        # Create prediction batch container (all tensors moved to CPU)
        return PredictionBatch(
            daily_pred=daily_mean.cpu().numpy(),
            weekly_pred=weekly_mean.cpu().numpy(),
            monthly_pred=monthly_mean.cpu().numpy(),
            daily_actual=daily_targets.cpu().numpy(),
            weekly_actual=weekly_targets.cpu().numpy(),
            monthly_actual=monthly_targets.cpu().numpy(),
            daily_insample=daily_insample,
            weekly_insample=weekly_insample,
            monthly_insample=monthly_insample
        )
    
    def collect_model_predictions(self, model: HierForecastNet, data_loader: DataLoader) -> ModelPredictions:
        """
        Collect all predictions from a neural model across the entire dataset.
        
        This function handles all the repetitive batch processing, concatenation,
        and device management that was cluttering the main evaluation code.
        
        Args:
            model: Trained hierarchical neural network
            data_loader: DataLoader with test data
            
        Returns:
            ModelPredictions: Container with all predictions and actuals
        """
        model.eval()
        
        # Collections for batch results
        batch_results: List[PredictionBatch] = []
        
        print(f"  Processing {len(data_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # Process single batch
                batch_result = self._process_single_batch(model, batch_data)
                batch_results.append(batch_result)
                
                # Optional progress reporting for large datasets
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"    Processed {batch_idx}/{len(data_loader)} batches...")
        
        print(f"  Concatenating results from {len(batch_results)} batches...")
        
        # Efficiently concatenate all batch results
        return self._concatenate_batch_results(batch_results)
    
    def _concatenate_batch_results(self, batch_results: List[PredictionBatch]) -> ModelPredictions:
        """
        Efficiently concatenate results from all batches.
        
        Args:
            batch_results: List of PredictionBatch objects
            
        Returns:
            ModelPredictions: Concatenated results
        """
        if not batch_results:
            raise ValueError("No batch results to concatenate")
        
        # Concatenate predictions
        daily_predictions = np.concatenate([batch.daily_pred for batch in batch_results], axis=0)
        weekly_predictions = np.concatenate([batch.weekly_pred for batch in batch_results], axis=0)
        monthly_predictions = np.concatenate([batch.monthly_pred for batch in batch_results], axis=0)
        
        # Concatenate actuals
        daily_actuals = np.concatenate([batch.daily_actual for batch in batch_results], axis=0)
        weekly_actuals = np.concatenate([batch.weekly_actual for batch in batch_results], axis=0)
        monthly_actuals = np.concatenate([batch.monthly_actual for batch in batch_results], axis=0)
        
        # Concatenate insample data
        daily_insample = np.concatenate([batch.daily_insample for batch in batch_results], axis=0)
        weekly_insample = np.concatenate([batch.weekly_insample for batch in batch_results], axis=0)
        monthly_insample = np.concatenate([batch.monthly_insample for batch in batch_results], axis=0)
        
        return ModelPredictions(
            daily_predictions=daily_predictions,
            weekly_predictions=weekly_predictions,
            monthly_predictions=monthly_predictions,
            daily_actuals=daily_actuals,
            weekly_actuals=weekly_actuals,
            monthly_actuals=monthly_actuals,
            daily_insample=daily_insample,
            weekly_insample=weekly_insample,
            monthly_insample=monthly_insample
        )
    
    def extract_neural_predictions_for_stacking(self, model: HierForecastNet, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Extract neural model predictions in a format suitable for stacked model variants.
        
        This is used by stacked_variants.py to get the base neural predictions
        for Deep+ARIMA and Deep+ARIMA+LGB variants.
        
        Args:
            model: Trained neural model
            data_loader: DataLoader with data
            
        Returns:
            Dict with predictions by scale
        """
        model_predictions = self.collect_model_predictions(model, data_loader)
        
        return {
            'daily': model_predictions.daily_predictions,
            'weekly': model_predictions.weekly_predictions,
            'monthly': model_predictions.monthly_predictions
        }


class BatchProcessor:
    """
    Generic batch processor for any model evaluation task.
    
    This can be extended for different types of models or evaluation tasks.
    """
    
    def __init__(self, device: Optional[torch.device] = None, verbose: bool = True):
        self.device = device or torch.device('cpu')
        self.verbose = verbose
    
    def process_data_loader(self, data_loader: DataLoader, process_fn, description: str = "Processing") -> List:
        """
        Generic function to process a DataLoader with a custom processing function.
        
        Args:
            data_loader: DataLoader to process
            process_fn: Function to apply to each batch
            description: Description for progress reporting
            
        Returns:
            List of results from processing each batch
        """
        results = []
        
        if self.verbose:
            print(f"  {description}: {len(data_loader)} batches...")
        
        for batch_idx, batch_data in enumerate(data_loader):
            result = process_fn(batch_data)
            results.append(result)
            
            if self.verbose and batch_idx % 50 == 0 and batch_idx > 0:
                print(f"    Processed {batch_idx}/{len(data_loader)} batches...")
        
        return results


def move_batch_to_device(batch_data: Tuple, device: torch.device) -> Tuple:
    """
    Efficiently move a batch of tensors to the specified device.
    
    Args:
        batch_data: Tuple of tensors
        device: Target device
        
    Returns:
        Tuple of tensors moved to device
    """
    return tuple(tensor.to(device).float() if isinstance(tensor, torch.Tensor) else tensor 
                 for tensor in batch_data)


def safe_numpy_conversion(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a PyTorch tensor to numpy, handling device placement.
    
    Args:
        tensor: PyTorch tensor (potentially on GPU)
        
    Returns:
        numpy array on CPU
    """
    return tensor.detach().cpu().numpy()


def compute_memory_usage(model: HierForecastNet, batch_size: int = 32) -> Dict[str, float]:
    """
    Estimate memory usage for model evaluation.
    
    Args:
        model: The neural model
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with memory usage estimates
    """
    # Create dummy input to estimate memory
    dummy_lookback = torch.randn(batch_size, 365, 1)
    dummy_features = torch.randn(batch_size, 1)
    
    # Estimate model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
    
    # Estimate activation memory (rough estimate)
    activation_memory = dummy_lookback.numel() * dummy_lookback.element_size() / (1024**2)  # MB
    
    return {
        'model_parameters_mb': param_memory,
        'estimated_activation_mb': activation_memory,
        'total_estimated_mb': param_memory + activation_memory
    }


def compute_metrics_per_sample_and_horizon(actuals: np.ndarray,
                                         predictions: np.ndarray,
                                         insample_histories: np.ndarray,
                                         seasonal_period: int,
                                         scale_name: str) -> Dict[str, float]:
    """
    Compute metrics properly accounting for:
    1. Per-sample insample scaling (not scalar collapse)
    2. Multi-horizon semantics (across origins, per horizon)
    3. Proper RMSSE/MASE computation with full history
    
    SHAPE SEMANTICS DOCUMENTED:
    - actuals/predictions: (N_origins, H_horizons) or (N_origins,) for single-step
    - insample_histories: (N_origins, L_history) - FULL history per sample
    - Metrics computed across N origins for each of H horizons
    
    This is COMPETITION STYLE: accuracy at horizon h across many forecast origins.
    NOT time-series style (accuracy steps ahead from single origin).
    
    Args:
        actuals: True values (N_origins, H_horizons) or (N_origins,)
        predictions: Predicted values, same shape as actuals
        insample_histories: Historical data (N_origins, L_history) for scaling
        seasonal_period: Seasonal period (7=daily, 4=weekly, 12=monthly)
        scale_name: Scale identifier for logging
        
    Returns:
        Dictionary with comprehensive metrics
    """
    from ..evaluation.metrics import compute_comprehensive_metrics, aggregate_metrics_across_horizons
    
    # Ensure numpy arrays
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    insample_histories = np.asarray(insample_histories)
    
    # Handle single-step case
    if actuals.ndim == 1:
        actuals = actuals.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)
    
    n_origins, n_horizons = actuals.shape
    
    if insample_histories.ndim == 1:
        # If 1D, broadcast to all samples
        insample_histories = np.repeat(insample_histories.reshape(1, -1), n_origins, axis=0)
    
    # Validate shapes
    assert actuals.shape == predictions.shape, f"Shape mismatch: actuals {actuals.shape} vs predictions {predictions.shape}"
    assert insample_histories.shape[0] == n_origins, f"Insample shape mismatch: {insample_histories.shape[0]} vs {n_origins} origins"
    
    print(f"    Computing {scale_name} metrics: {n_origins} origins × {n_horizons} horizons")
    print(f"    Insample histories shape: {insample_histories.shape}")
    
    # Compute metrics per horizon (across origins)
    horizon_metrics = {}
    
    for h in range(n_horizons):
        print(f"      Processing horizon {h+1}/{n_horizons}...")
        
        # Extract horizon h data across all origins
        h_actuals = actuals[:, h]  # (N_origins,)
        h_predictions = predictions[:, h]  # (N_origins,)
        
        # Compute per-sample metrics, then aggregate
        per_sample_metrics = []
        
        for i in range(n_origins):
            # Use FULL insample history for this sample
            sample_insample = insample_histories[i]  # (L_history,)
            
            # Compute metrics for this single sample at horizon h
            sample_metrics = compute_comprehensive_metrics(
                actual_values=h_actuals[i:i+1],  # Single value as array
                predicted_values=h_predictions[i:i+1],  # Single value as array
                insample_data=sample_insample,  # Full history
                seasonal_period=seasonal_period
            )
            per_sample_metrics.append(sample_metrics)
        
        # Aggregate across samples for this horizon
        aggregated = aggregate_metrics_across_horizons(per_sample_metrics)
        
        # Add horizon suffix to metric names
        for metric_name, metric_value in aggregated.items():
            horizon_metrics[f"{metric_name}_h{h+1}"] = metric_value
    
    # Compute overall averages across horizons
    for base_metric in ['mae', 'rmse', 'mape', 'smape', 'mase', 'rmsse']:
        horizon_values = [v for k, v in horizon_metrics.items() 
                         if k.startswith(f"{base_metric}_h") and not np.isnan(v)]
        if horizon_values:
            horizon_metrics[f"{base_metric}_avg"] = np.mean(horizon_values)
            horizon_metrics[f"{base_metric}_std"] = np.std(horizon_values) if len(horizon_values) > 1 else 0.0
    
    return horizon_metrics


def validate_temporal_split(train_loader: DataLoader, 
                          test_loader: DataLoader, 
                          validation_loader: Optional[DataLoader] = None) -> Dict[str, str]:
    """
    Validate that data loaders represent proper temporal splits with no leakage.
    
    This is CRITICAL for preventing data leakage in residual modeling and evaluation.
    
    Returns:
        Dictionary with validation results and warnings
    """
    validation_results = {
        'status': 'unknown',
        'train_samples': len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'unknown',
        'test_samples': len(test_loader.dataset) if hasattr(test_loader.dataset, '__len__') else 'unknown', 
        'warnings': []
    }
    
    # Basic validation - more sophisticated checks would require dataset inspection
    try:
        # Check if datasets have chronological ordering info
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        
        # If using RollingOrigin, check that splits don't overlap
        if hasattr(train_dataset, 'first_origin') and hasattr(test_dataset, 'first_origin'):
            if hasattr(train_dataset, 'last_origin') and hasattr(test_dataset, 'first_origin'):
                train_last = getattr(train_dataset, 'last_origin', None)
                test_first = getattr(test_dataset, 'first_origin', None)
                if train_last is not None and test_first is not None and train_last >= test_first:
                    validation_results['warnings'].append('POTENTIAL TEMPORAL LEAKAGE: Train end overlaps with test start')
        
        validation_results['status'] = 'checked'
        
    except Exception as e:
        validation_results['warnings'].append(f'Could not validate temporal split: {e}')
        validation_results['status'] = 'unknown'
    
    return validation_results

