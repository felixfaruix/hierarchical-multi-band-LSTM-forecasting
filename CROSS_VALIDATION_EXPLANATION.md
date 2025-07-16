"""
COMPREHENSIVE LINE-BY-LINE EXPLANATION: improved_cross_validation.py
================================================================

This document provides a complete theoretical background, scope, and line-by-line
explanation of the improved time series cross-validation implementation.

## THEORETICAL BACKGROUND & SCOPE

### 🎯 CORE PROBLEM BEING SOLVED

Traditional cross-validation is inappropriate for time series because:
1. **Temporal Dependencies**: Past values predict future values
2. **Data Leakage**: Using future data to predict past violates causality
3. **Non-IID Data**: Time series observations are not independent
4. **Forecast Origin Alignment**: CV splits must match real forecasting scenarios

### 📊 TIME SERIES CV THEORY

#### Walk-Forward Analysis (Fixed Window)
```
Training Window:     [----Training----] -> [Val] -> [----Training----] -> [Val]
Timeline:            t1              t2    t3     t4              t5    t6
```

#### Expanding Window Analysis  
```
Training Window:     [--Train--] -> [Val] -> [------Train------] -> [Val]
Timeline:            t1       t2    t3     t1               t4    t5
```

#### RollingOrigin Alignment
Each CV fold must match exactly what RollingOrigin would provide:
- Same lookback period
- Same forecast horizon
- Same data boundaries
- Same temporal gaps

### 🔍 KEY CONCEPTS

1. **Forecast Origin**: The point in time from which a forecast is made
2. **Lookback Period**: Historical data window used for prediction
3. **Forecast Horizon**: How far into the future to predict
4. **Data Leakage Prevention**: Ensuring future data never influences past predictions
5. **Per-Origin Metrics**: Individual loss per forecast origin for statistical testing

## LINE-BY-LINE EXPLANATION

### LINES 1-16: MODULE DOCSTRING & PURPOSE
"""

"""
Improved Time Series Cross-Validation with RollingOrigin Alignment

This module fixes the cross-validation issues identified:
1. Proper alignment between cross-validation folds and RollingOrigin dataset splits
2. Per-fold and per-origin loss extraction for valid statistical testing
3. Consistent handling of training/validation boundaries across methods
4. Support for both walk-forward and expanding window cross-validation
5. Proper subset extraction that matches RollingOrigin logic
"""

EXPLANATION:
- Defines the module's core mission: fixing time series CV issues
- Lists 5 specific problems being solved
- Emphasizes "RollingOrigin Alignment" - the key innovation

### LINES 17-24: IMPORTS SECTION

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

EXPLANATION:
- typing: Provides type hints for better code documentation and IDE support
- numpy/pandas: Core data manipulation libraries
- dataclass: Creates structured data containers without boilerplate
- torch: PyTorch for deep learning models
- warnings/contextlib: Error handling and context management

### LINES 25-27: CONDITIONAL IMPORT

if TYPE_CHECKING:
    from timeseries_datamodule import RollingOrigin

EXPLANATION:
- TYPE_CHECKING: Only imports during type checking, not runtime
- Prevents circular imports while maintaining type safety
- RollingOrigin is the core dataset class we're aligning with

### LINES 29-35: FALLBACK CONSTANTS

try:
    from timeseries_datamodule import lookback_days, daily_window, monthly_horizon
except ImportError:
    lookback_days = 365
    daily_window = 14
    monthly_horizon = 30

EXPLANATION:
- Attempts to import configuration constants from the main module
- Provides sensible defaults if import fails
- Ensures code works even in isolated testing environments

### LINES 38-49: CVFold DATACLASS

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

EXPLANATION:
- @dataclass: Automatically generates __init__, __repr__, etc.
- Container for all information about a single CV fold
- fold_index: Which fold this is (0, 1, 2, ...)
- train_indices/val_indices: Dataset indices for training/validation
- train_origins/val_origins: Forecast origin points for each set
- Date fields: Optional timestamp tracking for interpretability

WHY THIS MATTERS:
- Encapsulates all fold information in one place
- Makes CV results traceable and debuggable
- Enables proper temporal alignment verification

### LINES 52-62: CVResults DATACLASS

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

EXPLANATION:
- Complete container for all CV results
- fold_results: Detailed info per fold
- per_fold_losses: Aggregate loss per fold (for Diebold-Mariano testing)
- per_origin_losses: Individual loss per forecast origin (critical for statistical tests)
- aggregated_metrics: Overall performance across all folds
- fold_info: CVFold objects with split details
- model_predictions/targets: Raw predictions for further analysis

KEY INSIGHT:
- per_origin_losses is CRITICAL for proper statistical testing
- Standard CV only gives fold-level metrics, losing origin-specific information
- This enables proper Diebold-Mariano tests between models

### LINES 65-85: CLASS INITIALIZATION

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

EXPLANATION:
- Main class for improved time series cross-validation
- n_splits: How many CV folds to create
- cv_type: "walk_forward" (fixed window) vs "expanding" (growing window)
- min_train_size: Minimum training samples (auto-calculated if None)
- forecast_horizon: How many time steps ahead to predict
- gap: Buffer between train/validation to prevent data leakage

CV_TYPE EXPLAINED:
- walk_forward: Each fold uses same-size training window (realistic for online learning)
- expanding: Each fold includes all previous data (better for offline learning)

### LINES 86-99: MAIN SPLIT GENERATION METHOD

def get_rolling_origin_aligned_splits(self, 
                                    rolling_dataset: 'RollingOrigin',
                                    date_index: Optional[pd.DatetimeIndex] = None) -> List[CVFold]:

EXPLANATION:
- Core method that generates CV splits aligned with RollingOrigin logic
- rolling_dataset: The actual RollingOrigin dataset to align with
- date_index: Optional timestamps for human-readable fold boundaries
- Returns: List of CVFold objects with properly aligned splits

WHY "ALIGNED"?
- Traditional CV randomly splits data, violating temporal order
- RollingOrigin creates specific train/test boundaries based on time
- This method ensures CV folds respect the same boundaries

### LINES 100-117: DATASET SIZE VALIDATION

total_samples = len(rolling_dataset)
lookback = getattr(rolling_dataset, 'lookback', lookback_days)

if self.min_train_size is None:
    self.min_train_size = max(lookback + self.forecast_horizon, total_samples // (self.n_splits + 1))

available_origins = total_samples - lookback - self.forecast_horizon + 1

if available_origins < self.n_splits:
    raise ValueError(f"Not enough data for {self.n_splits} splits. "
                   f"Available origins: {available_origins}")

EXPLANATION:
- total_samples: How many samples in the dataset
- lookback: Historical window size (from RollingOrigin or default)
- Auto-calculate min_train_size if not provided
- available_origins: How many valid forecast origins exist
- Error if insufficient data for requested splits

MATHEMATICAL INSIGHT:
available_origins = total_samples - lookback - forecast_horizon + 1

This formula accounts for:
- lookback: Need historical data before first forecast origin
- forecast_horizon: Need future data after last forecast origin
- +1: Off-by-one correction for inclusive ranges

### LINES 118-125: FOLD BOUNDARY CALCULATION

fold_boundaries = self._calculate_fold_boundaries(available_origins, total_samples, lookback)

cv_folds = []
for fold_idx, (train_end, val_start, val_end) in enumerate(fold_boundaries):

EXPLANATION:
- Delegates boundary calculation to helper method
- Iterates through each fold's boundaries
- Each boundary tuple: (train_end, val_start, val_end)

### LINES 126-135: TRAIN/VALIDATION INDEX GENERATION

if self.cv_type == "walk_forward":
    train_start = max(0, train_end - self.min_train_size)
else:  # expanding
    train_start = 0

train_indices, val_indices = self._generate_aligned_indices(
    rolling_dataset, train_start, train_end, val_start, val_end, lookback
)

EXPLANATION:
- walk_forward: Use fixed window (train_start moves forward)
- expanding: Use all data from beginning (train_start = 0)
- _generate_aligned_indices: Core method ensuring RollingOrigin alignment

### LINES 136-140: ORIGIN EXTRACTION

train_origins = self._extract_origins(rolling_dataset, train_indices, lookback)
val_origins = self._extract_origins(rolling_dataset, val_indices, lookback)

EXPLANATION:
- Extracts forecast origin points for train/validation sets
- Essential for per-origin loss tracking
- Uses dataset indices + lookback to find origin positions

### LINES 141-155: CVFOLD CREATION

train_start_date = date_index[train_start] if date_index is not None else None
# ... (similar for other dates)

cv_folds.append(CVFold(
    fold_index=fold_idx,
    train_indices=train_indices,
    val_indices=val_indices,
    train_origins=train_origins,
    val_origins=val_origins,
    train_start_date=train_start_date,
    # ... (other dates)
))

EXPLANATION:
- Converts indices to human-readable dates if available
- Creates CVFold object with all fold information
- Stores complete provenance for debugging and validation

### LINES 160-180: FOLD BOUNDARY CALCULATION HELPER

def _calculate_fold_boundaries(self, 
                             available_origins: int, 
                             total_samples: int, 
                             lookback: int) -> List[Tuple[int, int, int]]:

EXPLANATION:
- Calculates where each fold should start/end
- Returns list of (train_end, val_start, val_end) tuples
- Core algorithm ensuring equal-sized validation periods

### LINES 181-195: BOUNDARY LOGIC

val_size = max(1, available_origins // self.n_splits)

boundaries = []
for fold_idx in range(self.n_splits):
    val_start_origin = fold_idx * val_size
    val_end_origin = min((fold_idx + 1) * val_size, available_origins)
    
    val_start = lookback + val_start_origin
    val_end = lookback + val_end_origin
    
    train_end = val_start - self.gap - 1

EXPLANATION:
- val_size: How many origins per validation fold
- val_start_origin/val_end_origin: Origin-space coordinates
- val_start/val_end: Convert to absolute dataset indices
- train_end: Training ends before validation (with gap for leakage prevention)

CRITICAL INSIGHT:
The gap parameter prevents data leakage:
- Without gap: training data at time T, validation at time T+1
- With gap=1: training data at time T, validation at time T+2
- Prevents models from memorizing immediate future patterns

### LINES 200-220: ALIGNED INDEX GENERATION

def _generate_aligned_indices(self, 
                            rolling_dataset: 'RollingOrigin',
                            train_start: int,
                            train_end: int,
                            val_start: int,
                            val_end: int,
                            lookback: int) -> Tuple[np.ndarray, np.ndarray]:

EXPLANATION:
- Generates dataset indices that match RollingOrigin logic
- Ensures each sample's origin falls within the appropriate time window
- Returns indices suitable for PyTorch Subset creation

### LINES 221-240: INDEX GENERATION LOGIC

train_indices = []
for i in range(len(rolling_dataset)):
    origin_idx = i + lookback
    if origin_idx <= train_end:
        train_indices.append(i)

val_indices = []
for i in range(len(rolling_dataset)):
    origin_idx = i + lookback
    if val_start <= origin_idx < val_end:
        val_indices.append(i)

EXPLANATION:
- For each sample in dataset, calculate its forecast origin
- origin_idx = i + lookback (where forecast is made from)
- Include sample if origin falls in appropriate time window
- This ensures temporal consistency with RollingOrigin

MATHEMATICAL RELATIONSHIP:
Sample i -> uses data[i:i+lookback] -> forecasts at origin i+lookback

### LINES 245-255: ORIGIN EXTRACTION HELPER

def _extract_origins(self, rolling_dataset: 'RollingOrigin', indices: np.ndarray, lookback: int) -> np.ndarray:
    """Extract forecast origins for given dataset indices."""
    origins = []
    for idx in indices:
        origin_idx = idx + lookback
        origins.append(origin_idx)
    return np.array(origins)

EXPLANATION:
- Simple helper to convert dataset indices to forecast origins
- Essential for per-origin loss tracking in statistical tests
- Maintains mapping between samples and their temporal positions

### LINES 260-285: MAIN CROSS-VALIDATION METHOD

def cross_validate_model(self,
                       model_class: Any,
                       model_params: Dict[str, Any],
                       rolling_dataset: 'RollingOrigin',
                       loss_functions: Optional[Dict[str, Union[str, Callable]]] = None,
                       date_index: Optional[pd.DatetimeIndex] = None,
                       verbose: bool = True) -> CVResults:

EXPLANATION:
- Main entry point for performing cross-validation
- model_class: Class to instantiate (e.g., neural network, ARIMA)
- model_params: Parameters for model initialization
- loss_functions: Dict of loss functions to evaluate
- Returns: Complete CVResults with all metrics and provenance

### LINES 286-295: LOSS FUNCTION DEFAULTS

if loss_functions is None:
    loss_functions = {"mse": "squared", "mae": "absolute"}

cv_folds = self.get_rolling_origin_aligned_splits(rolling_dataset, date_index)

EXPLANATION:
- Provides sensible defaults for loss functions
- Generates aligned CV splits using method explained above
- Sets up infrastructure for fold-by-fold processing

### LINES 296-310: FOLD PROCESSING LOOP

fold_results = []
per_fold_losses = {}
per_origin_losses = {}
model_predictions = {}
model_targets = {}

for fold in cv_folds:
    if verbose:
        print(f"Processing fold {fold.fold_index + 1}/{len(cv_folds)}")

EXPLANATION:
- Initializes result containers
- Iterates through each CV fold
- Provides progress feedback if verbose=True

### LINES 311-325: MODEL TRAINING PER FOLD

train_subset = Subset(rolling_dataset, fold.train_indices.tolist())
val_subset = Subset(rolling_dataset, fold.val_indices.tolist())

model = model_class(**model_params)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

if hasattr(model, 'fit'):
    model.fit(train_loader)

EXPLANATION:
- Creates PyTorch Subset objects for train/validation
- Instantiates fresh model for each fold (prevents data leakage)
- Creates DataLoader for batch processing
- Trains model using .fit() method if available

CRITICAL POINT:
- Fresh model per fold prevents information leakage between folds
- Subset creation ensures exact alignment with RollingOrigin logic

### LINES 326-340: MODEL EVALUATION

val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
predictions, targets = self._evaluate_model(model, val_loader)

model_predictions[fold.fold_index] = predictions
model_targets[fold.fold_index] = targets

origin_losses = self._calculate_per_origin_losses(
    predictions, targets, fold.val_origins, loss_functions
)

EXPLANATION:
- Creates validation DataLoader (shuffle=False preserves order)
- Evaluates model to get predictions and targets
- Stores raw predictions/targets for later analysis
- Calculates per-origin losses (critical for statistical testing)

### LINES 341-360: LOSS CALCULATION AND STORAGE

per_origin_losses[fold.fold_index] = origin_losses

fold_losses = self._calculate_fold_losses(
    predictions, targets, loss_functions
)
per_fold_losses[fold.fold_index] = fold_losses

fold_result = {
    "fold_index": fold.fold_index,
    "train_size": len(fold.train_indices),
    "val_size": len(fold.val_indices),
    "train_origins": fold.train_origins,
    "val_origins": fold.val_origins,
    "losses": fold_losses,
    "origin_losses": origin_losses
}

EXPLANATION:
- Stores per-origin and per-fold losses
- Creates comprehensive fold result dictionary
- Maintains complete provenance for debugging and analysis

### LINES 365-375: RESULT AGGREGATION

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

EXPLANATION:
- Aggregates metrics across all folds (mean, std, min, max)
- Returns comprehensive CVResults object
- Provides complete information for further analysis

### LINES 380-420: MODEL EVALUATION HELPER

def _evaluate_model(self, model: Any, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on data loader."""
    predictions = []
    targets = []
    
    model.eval() if hasattr(model, 'eval') else None
    
    with torch.no_grad() if hasattr(torch, 'no_grad') else contextlib.nullcontext():
        for batch in data_loader:
            # ... (batch processing logic)

EXPLANATION:
- Generic model evaluation that works with different model types
- Sets model to evaluation mode if PyTorch model
- Disables gradients for efficiency
- Handles different batch formats gracefully

### LINES 425-455: PER-ORIGIN LOSS CALCULATION

def _calculate_per_origin_losses(self,
                               predictions: np.ndarray,
                               targets: np.ndarray,
                               origins: np.ndarray,
                               loss_functions: Dict[str, Union[str, Callable]]) -> Dict[str, Dict[int, float]]:

EXPLANATION:
- Calculates loss for each individual forecast origin
- Essential for proper statistical testing (Diebold-Mariano)
- Supports custom loss functions
- Returns nested dict: {loss_name: {origin: loss_value}}

### LINES 456-485: LOSS FUNCTION EVALUATION

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

EXPLANATION:
- Iterates through each forecast origin
- Extracts corresponding prediction and target
- Calculates error and applies loss function
- Supports callable functions and string shortcuts

### LINES 490-520: FOLD-LEVEL LOSS AGGREGATION

def _calculate_fold_losses(self,
                         predictions: np.ndarray,
                         targets: np.ndarray,
                         loss_functions: Dict[str, Union[str, Callable]]) -> Dict[str, float]:

EXPLANATION:
- Calculates aggregate loss across entire fold
- Used for overall model comparison
- Complements per-origin losses for complete picture

### LINES 525-540: METRIC AGGREGATION ACROSS FOLDS

def _aggregate_metrics(self, per_fold_losses: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across all folds."""
    # ... (aggregation logic)

EXPLANATION:
- Computes mean, std, min, max across all folds
- Provides summary statistics for model performance
- Essential for final model comparison and selection

### LINES 545-575: STATISTICAL TESTING EXTRACTION

def extract_cv_results_for_statistical_testing(cv_results: CVResults,
                                              loss_function_name: str = "mse") -> Dict[str, Any]:

EXPLANATION:
- Extracts CV results in format suitable for Diebold-Mariano testing
- Flattens per-origin losses across all folds
- Returns structured data for statistical analysis

CRITICAL IMPORTANCE:
- Standard CV loses per-origin information through aggregation
- This function preserves origin-level data needed for proper statistical tests
- Enables rigorous model comparison following forecasting best practices

### LINES 580-504: HELPER FUNCTION TEMPLATE

def add_sample_info_method():
    """
    This should be added to the existing RollingOrigin class...
    """

EXPLANATION:
- Template for enhancing RollingOrigin class
- Provides detailed sample information for debugging
- Helps verify alignment between CV and RollingOrigin

## KEY INNOVATIONS SUMMARY

### 1. TEMPORAL ALIGNMENT
- CV folds respect exact RollingOrigin boundaries
- No violation of temporal causality
- Realistic simulation of actual forecasting scenarios

### 2. PER-ORIGIN TRACKING
- Individual loss per forecast origin preserved
- Enables proper statistical testing (Diebold-Mariano)
- Maintains granular information for analysis

### 3. FLEXIBLE CV TYPES
- Walk-forward: Fixed training window (online learning)
- Expanding: Growing training window (offline learning)
- Configurable gaps to prevent data leakage

### 4. ROBUST ERROR HANDLING
- Validates sufficient data for requested splits
- Graceful handling of different model types
- Comprehensive result provenance

### 5. STATISTICAL RIGOR
- Per-origin losses for proper significance testing
- Fold-level aggregation for overall comparison
- Complete result structure for further analysis

## USAGE IMPLICATIONS

### FOR RESEARCH
- Enables proper statistical comparison between models
- Provides publication-quality cross-validation methodology
- Maintains complete audit trail for reproducibility

### FOR PRODUCTION
- Realistic estimation of model performance
- Proper temporal validation methodology
- Supports different deployment scenarios (online vs offline)

### FOR MODEL SELECTION
- Statistically valid model comparison
- Per-origin insights for understanding model behavior
- Comprehensive performance metrics across time periods

This implementation represents a significant advancement in time series cross-validation,
addressing fundamental issues that compromise the validity of traditional CV approaches
in temporal forecasting contexts.
"""
