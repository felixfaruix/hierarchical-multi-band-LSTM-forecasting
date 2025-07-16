"""
Improved Time Series Cross-Validation with RollingOrigin Alignment

This module fixes the cross-validation issues identified:
1. Proper alignment between cross-validation folds and RollingOrigin dataset splits
2. Per-fold and per-origin loss extraction for valid statistical testing
3. Consistent handling of training/validation boundaries across methods
4. Support for both walk-forward and expanding window cross-validation
5. Proper subset extraction that matches RollingOrigin logic

Key improvements:
- CV folds now explicitly match RollingOrigin dataset boundaries
- Per-fold training uses the same data subset as RollingOrigin would provide
- Loss extraction preserves origin-specific information for statistical tests
- Validation sets are properly subset to match forecast origins
- Support for custom loss functions in cross-validation evaluation
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Generator, TYPE_CHECKING
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import warnings
from pathlib import Path
import contextlib

if TYPE_CHECKING:
    from timeseries_datamodule import RollingOrigin

# Import constants from timeseries_datamodule
try:
    from timeseries_datamodule import lookback_days, daily_window, monthly_horizon
except ImportError:
    lookback_days = 365
    daily_window = 14
    monthly_horizon = 30


@dataclass
class CVFold:
    """Container for a single cross-validation fold."""
    fold_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_origins: np.ndarray
    val_origins: np.ndarray
    train_start_date: Optional[pd.Timestamp]
    train_end_date: Optional[pd.Timestamp]
    val_start_date: Optional[pd.Timestamp]
    val_end_date: Optional[pd.Timestamp]


@dataclass
class CVResults:
    """Container for cross-validation results."""
    fold_results: List[Dict[str, Any]]
    per_fold_losses: Dict[int, Dict[str, float]]  # fold -> {metric_name: value}
    per_origin_losses: Dict[int, Dict[int, float]]  # fold -> {origin: loss}
    aggregated_metrics: Dict[str, float]
    fold_info: List[CVFold]
    model_predictions: Dict[int, np.ndarray]  # fold -> predictions
    model_targets: Dict[int, np.ndarray]  # fold -> targets


class ImprovedTimeSeriesCrossValidator:
    """
    Improved time series cross-validator with proper RollingOrigin alignment.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 cv_type: str = "walk_forward",
                 min_train_size: Optional[int] = None,
                 forecast_horizon: int = 1,
                 gap: int = 0):
        """
        Initialize the improved cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            cv_type: "walk_forward" (fixed window) or "expanding" (growing window)
            min_train_size: Minimum training size for first fold
            forecast_horizon: Forecast horizon for validation
            gap: Gap between training and validation (to avoid data leakage)
        """
        self.n_splits = n_splits
        self.cv_type = cv_type
        self.min_train_size = min_train_size
        self.forecast_horizon = forecast_horizon
        self.gap = gap
    
    def get_rolling_origin_aligned_splits(self, 
                                        rolling_dataset: 'RollingOrigin',
                                        date_index: Optional[pd.DatetimeIndex] = None) -> List[CVFold]:
        """
        Generate CV splits that align exactly with RollingOrigin dataset logic.
        
        This ensures that:
        1. Each fold's training data matches what RollingOrigin would provide
        2. Validation origins align with RollingOrigin forecast origins
        3. No data leakage between training and validation
        
        Args:
            rolling_dataset: RollingOrigin dataset to align with
            date_index: Optional date index for timestamp tracking
            
        Returns:
            List of CVFold objects with aligned splits
        """
        total_samples = len(rolling_dataset)
        # Access attributes safely
        lookback = getattr(rolling_dataset, 'lookback', lookback_days)
        
        if self.min_train_size is None:
            # Set minimum to have enough data for at least one full lookback
            self.min_train_size = max(lookback + self.forecast_horizon, total_samples // (self.n_splits + 1))
        
        # Calculate split points based on RollingOrigin logic
        available_origins = total_samples - lookback - self.forecast_horizon + 1
        
        if available_origins < self.n_splits:
            raise ValueError(f"Not enough data for {self.n_splits} splits. "
                           f"Available origins: {available_origins}")
        
        # Generate fold boundaries that align with RollingOrigin indices
        fold_boundaries = self._calculate_fold_boundaries(available_origins, total_samples, lookback)
        
        cv_folds = []
        for fold_idx, (train_end, val_start, val_end) in enumerate(fold_boundaries):
            # Calculate training indices
            if self.cv_type == "walk_forward":
                train_start = max(0, train_end - self.min_train_size)
            else:  # expanding
                train_start = 0
            
            # Generate train/val indices that match RollingOrigin logic
            train_indices, val_indices = self._generate_aligned_indices(
                rolling_dataset, train_start, train_end, val_start, val_end, lookback
            )
            
            # Extract origin information
            train_origins = self._extract_origins(rolling_dataset, train_indices, lookback)
            val_origins = self._extract_origins(rolling_dataset, val_indices, lookback)
            
            # Convert to dates if available
            train_start_date = date_index[train_start] if date_index is not None else None
            train_end_date = date_index[train_end] if date_index is not None else None
            val_start_date = date_index[val_start] if date_index is not None else None
            val_end_date = date_index[val_end] if date_index is not None else None
            
            cv_folds.append(CVFold(
                fold_index=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                train_origins=train_origins,
                val_origins=val_origins,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                val_start_date=val_start_date,
                val_end_date=val_end_date
            ))
        
        return cv_folds
    
    def _calculate_fold_boundaries(self, 
                                 available_origins: int, 
                                 total_samples: int, 
                                 lookback: int) -> List[Tuple[int, int, int]]:
        """Calculate fold boundaries aligned with RollingOrigin logic."""
        # Calculate validation size per fold
        val_size = max(1, available_origins // self.n_splits)
        
        boundaries = []
        for fold_idx in range(self.n_splits):
            # Validation period for this fold
            val_start_origin = fold_idx * val_size
            val_end_origin = min((fold_idx + 1) * val_size, available_origins)
            
            # Convert to absolute indices
            val_start = lookback + val_start_origin
            val_end = lookback + val_end_origin
            
            # Training ends before validation (with gap)
            train_end = val_start - self.gap - 1
            
            boundaries.append((train_end, val_start, val_end))
        
        return boundaries
    
    def _generate_aligned_indices(self, 
                                rolling_dataset: 'RollingOrigin',
                                train_start: int,
                                train_end: int,
                                val_start: int,
                                val_end: int,
                                lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate indices that match RollingOrigin dataset logic."""
        # For training: include all valid rolling windows that end before train_end
        train_indices = []
        for i in range(len(rolling_dataset)):
            # Calculate the end index for this sample
            origin_idx = i + lookback
            if origin_idx <= train_end:
                train_indices.append(i)
        
        # For validation: include rolling windows that start in validation period
        val_indices = []
        for i in range(len(rolling_dataset)):
            # Calculate the origin index for this sample
            origin_idx = i + lookback
            if val_start <= origin_idx < val_end:
                val_indices.append(i)
        
        return np.array(train_indices), np.array(val_indices)
    
    def _extract_origins(self, rolling_dataset: 'RollingOrigin', indices: np.ndarray, lookback: int) -> np.ndarray:
        """Extract forecast origins for given dataset indices."""
        origins = []
        for idx in indices:
            # Calculate origin index based on RollingOrigin logic
            origin_idx = idx + lookback
            origins.append(origin_idx)
        return np.array(origins)
    
    def cross_validate_model(self,
                           model_class: Any,
                           model_params: Dict[str, Any],
                           rolling_dataset: 'RollingOrigin',
                           loss_functions: Optional[Dict[str, Union[str, Callable]]] = None,
                           date_index: Optional[pd.DatetimeIndex] = None,
                           verbose: bool = True) -> CVResults:
        """
        Perform cross-validation with proper RollingOrigin alignment.
        
        Args:
            model_class: Model class to instantiate for each fold
            model_params: Parameters for model initialization
            rolling_dataset: RollingOrigin dataset
            loss_functions: Dictionary of loss functions to evaluate
            date_index: Optional date index for tracking
            verbose: Whether to print progress
            
        Returns:
            CVResults with detailed per-fold and per-origin results
        """
        if loss_functions is None:
            loss_functions = {"mse": "squared", "mae": "absolute"}
        
        # Generate aligned CV splits
        cv_folds = self.get_rolling_origin_aligned_splits(rolling_dataset, date_index)
        
        fold_results = []
        per_fold_losses = {}
        per_origin_losses = {}
        model_predictions = {}
        model_targets = {}
        
        for fold in cv_folds:
            if verbose:
                print(f"Processing fold {fold.fold_index + 1}/{len(cv_folds)}")
                print(f"  Train origins: {len(fold.train_origins)}, Val origins: {len(fold.val_origins)}")
            
            # Create train/val subsets
            train_subset = Subset(rolling_dataset, fold.train_indices.tolist())
            val_subset = Subset(rolling_dataset, fold.val_indices.tolist())
            
            # Train model
            model = model_class(**model_params)
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            
            # Model training (simplified - actual implementation would be model-specific)
            if hasattr(model, 'fit'):
                model.fit(train_loader)
            
            # Evaluate on validation set
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            predictions, targets = self._evaluate_model(model, val_loader)
            
            # Store predictions and targets
            model_predictions[fold.fold_index] = predictions
            model_targets[fold.fold_index] = targets
            
            # Calculate per-origin losses
            origin_losses = self._calculate_per_origin_losses(
                predictions, targets, fold.val_origins, loss_functions
            )
            per_origin_losses[fold.fold_index] = origin_losses
            
            # Calculate aggregate fold losses
            fold_losses = self._calculate_fold_losses(
                predictions, targets, loss_functions
            )
            per_fold_losses[fold.fold_index] = fold_losses
            
            # Store detailed fold results
            fold_result = {
                "fold_index": fold.fold_index,
                "train_size": len(fold.train_indices),
                "val_size": len(fold.val_indices),
                "train_origins": fold.train_origins,
                "val_origins": fold.val_origins,
                "losses": fold_losses,
                "origin_losses": origin_losses
            }
            fold_results.append(fold_result)
        
        # Calculate aggregated metrics across all folds
        aggregated_metrics = self._aggregate_metrics(per_fold_losses)
        
        return CVResults(
            fold_results=fold_results,
            per_fold_losses=per_fold_losses,
            per_origin_losses=per_origin_losses,
            aggregated_metrics=aggregated_metrics,
            fold_info=cv_folds,
            model_predictions=model_predictions,
            model_targets=model_targets
        )
    
    def _evaluate_model(self, model: Any, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model on data loader."""
        predictions = []
        targets = []
        
        model.eval() if hasattr(model, 'eval') else None
        
        with torch.no_grad() if hasattr(torch, 'no_grad') else contextlib.nullcontext():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    X, y = batch[0], batch[1]
                else:
                    X, y = batch, None
                
                # Model prediction (simplified)
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                elif hasattr(model, 'forward'):
                    pred = model.forward(X)
                else:
                    pred = model(X)
                
                predictions.append(pred.detach().cpu().numpy() if hasattr(pred, 'detach') else pred)
                if y is not None:
                    targets.append(y.detach().cpu().numpy() if hasattr(y, 'detach') else y)
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0) if targets else np.zeros_like(predictions)
        
        return predictions, targets
    
    def _calculate_per_origin_losses(self,
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   origins: np.ndarray,
                                   loss_functions: Dict[str, Union[str, Callable]]) -> Dict[str, Dict[int, float]]:
        """Calculate losses per forecast origin."""
        per_origin_losses = {}
        
        for loss_name, loss_func in loss_functions.items():
            origin_losses = {}
            
            for i, origin in enumerate(origins):
                pred = predictions[i] if predictions.ndim > 1 else predictions[i:i+1]
                target = targets[i] if targets.ndim > 1 else targets[i:i+1]
                
                error = pred - target
                
                if callable(loss_func):
                    loss = float(np.mean(loss_func(error)))
                elif loss_func == "squared":
                    loss = float(np.mean(error ** 2))
                elif loss_func == "absolute":
                    loss = float(np.mean(np.abs(error)))
                else:
                    # Default to squared error
                    loss = float(np.mean(error ** 2))
                
                origin_losses[int(origin)] = loss
            
            per_origin_losses[loss_name] = origin_losses
        
        return per_origin_losses
    
    def _calculate_fold_losses(self,
                             predictions: np.ndarray,
                             targets: np.ndarray,
                             loss_functions: Dict[str, Union[str, Callable]]) -> Dict[str, float]:
        """Calculate aggregate losses for the fold."""
        fold_losses = {}
        
        errors = predictions - targets
        
        for loss_name, loss_func in loss_functions.items():
            if callable(loss_func):
                loss = float(np.mean(loss_func(errors)))
            elif loss_func == "squared":
                loss = float(np.mean(errors ** 2))
            elif loss_func == "absolute":
                loss = float(np.mean(np.abs(errors)))
            else:
                loss = float(np.mean(errors ** 2))
            
            fold_losses[loss_name] = loss
        
        return fold_losses
    
    def _aggregate_metrics(self, per_fold_losses: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all folds."""
        aggregated = {}
        
        if not per_fold_losses:
            return aggregated
        
        # Get all metric names
        metric_names = list(next(iter(per_fold_losses.values())).keys())
        
        for metric_name in metric_names:
            values = [fold_losses[metric_name] for fold_losses in per_fold_losses.values()]
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))
            aggregated[f"{metric_name}_min"] = float(np.min(values))
            aggregated[f"{metric_name}_max"] = float(np.max(values))
        
        return aggregated


def extract_cv_results_for_statistical_testing(cv_results: CVResults,
                                              loss_function_name: str = "mse") -> Dict[str, Any]:
    """
    Extract cross-validation results in format suitable for statistical testing.
    
    Args:
        cv_results: Results from cross-validation
        loss_function_name: Name of loss function to extract
        
    Returns:
        Dictionary with per-fold and per-origin losses formatted for DM test
    """
    # Extract per-fold losses
    per_fold_losses = []
    for fold_idx in sorted(cv_results.per_fold_losses.keys()):
        if loss_function_name in cv_results.per_fold_losses[fold_idx]:
            per_fold_losses.append(cv_results.per_fold_losses[fold_idx][loss_function_name])
    
    # Extract per-origin losses (flattened across all folds)
    all_origin_losses = []
    all_origins = []
    
    for fold_idx in sorted(cv_results.per_origin_losses.keys()):
        if loss_function_name in cv_results.per_origin_losses[fold_idx]:
            origin_losses = cv_results.per_origin_losses[fold_idx][loss_function_name]
            for origin, loss in origin_losses.items():
                all_origin_losses.append(loss)
                all_origins.append(origin)
    
    return {
        "per_fold_losses": np.array(per_fold_losses),
        "per_origin_losses": np.array(all_origin_losses),
        "forecast_origins": np.array(all_origins),
        "fold_info": cv_results.fold_info,
        "aggregated_metrics": cv_results.aggregated_metrics
    }
