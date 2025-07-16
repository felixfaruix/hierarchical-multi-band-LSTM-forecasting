"""
Stacked Model Variants Evaluation (Clean Version)

This module implements evaluation of stacked deep learning variants:
1. Deep only (baseline neural model)
2. Deep + ARIMA residual modeling
3. Deep + ARIMA + LightGBM residual modeling

Key improvements:
- Uses evaluation_utils for clean batch processing
- Improved device handling and memory management
- Modular design for easy extension
- Clean helper functions without repetitive code
"""

import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader
from typing import Dict, Optional

from evaluation.metrics import compute_comprehensive_metrics
from utils import evaluation_utils
from utils.evaluation_utils import NeuralModelEvaluator, compute_metrics_per_sample_and_horizon
from models.model import HierForecastNet


# Optional dependencies with import guards
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    pm = None
    PMDARIMA_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False


def extract_neural_predictions_for_stacking(neural_model: HierForecastNet,
                                           test_data_loader: DataLoader, 
                                           device: Optional[torch.device] = None) -> Dict[str, np.ndarray]:
    """
    Extract predictions from neural model using clean helper functions.
    
    Args:
        neural_model: Trained neural model
        test_data_loader: Test data loader
        device: Device for computation
        
    Returns:
        Dictionary with predictions by scale
    """
    # Use our clean evaluation helper
    evaluator = NeuralModelEvaluator(device)
    model_predictions = evaluator.collect_model_predictions(neural_model, test_data_loader)
    
    return {
        'daily': model_predictions.daily_predictions,
        'weekly': model_predictions.weekly_predictions, 
        'monthly': model_predictions.monthly_predictions,
        'daily_actuals': model_predictions.daily_actuals,
        'weekly_actuals': model_predictions.weekly_actuals,
        'monthly_actuals': model_predictions.monthly_actuals,
        'daily_insample': model_predictions.daily_insample,
        'weekly_insample': model_predictions.weekly_insample,
        'monthly_insample': model_predictions.monthly_insample
    }


def evaluate_deep_only_variant(predictions: np.ndarray,
                              actuals: np.ndarray,
                              insample_data: np.ndarray,
                              scale: str) -> Dict[str, float]:
    """
    Evaluate the deep-only variant using bulletproof per-sample scaling.
    
    Args:
        predictions: Neural model predictions, shape (N_origins, H_horizons)
        actuals: True values, shape (N_origins, H_horizons)
        insample_data: Historical data for scaling metrics, shape (N_origins, L_history)
        scale: Time scale ('daily', 'weekly', 'monthly')
        
    Returns:
        Dictionary with evaluation metrics per horizon using proper per-sample scaling
    """
    
    seasonal_periods = {'daily': 7, 'weekly': 4, 'monthly': 12}
    seasonal_period = seasonal_periods.get(scale, 1)
    
    # Handle case where insample_data is 1D
    if insample_data.ndim == 1:
        # Broadcast to match number of origins
        n_origins = predictions.shape[0] if predictions.ndim > 1 else 1
        insample_data = np.repeat(insample_data.reshape(1, -1), n_origins, axis=0)
    
    # Use bulletproof per-sample scaling for deep-only variant evaluation

    return compute_metrics_per_sample_and_horizon(
        actuals=actuals,
        predictions=predictions,
        insample_histories=insample_data,
        seasonal_period=seasonal_period,
        scale_name=f"deep_only_{scale}"
    )


def fit_arima_on_residuals(neural_predictions: np.ndarray, 
                          actuals: np.ndarray,
                          training_residuals: Optional[np.ndarray] = None) -> Optional[object]:
    """
    Fit ARIMA model on neural network residuals.
    
    Args:
        neural_predictions: Neural model predictions
        actuals: True values
        training_residuals: Optional pre-computed training residuals
        
    Returns:
        Fitted ARIMA model or None if pmdarima not available
    """
    if not PMDARIMA_AVAILABLE:
        warnings.warn("pmdarima not available, skipping ARIMA residual modeling")
        return None
    
    try:
        # Compute residuals
        if training_residuals is not None:
            residuals = training_residuals
        else:
            residuals = actuals.flatten() - neural_predictions.flatten()
        
        # Remove any infinite or NaN values
        clean_residuals = residuals[np.isfinite(residuals)]
        
        if len(clean_residuals) < 10:
            warnings.warn("Insufficient clean residuals for ARIMA fitting")
            return None
        
        # Fit ARIMA model with conservative settings
        arima_model = pm.auto_arima(  # type: ignore
            clean_residuals,
            start_p=0, start_q=0, start_P=0, start_Q=0,
            max_p=3, max_q=3, max_P=2, max_Q=2,
            seasonal=True, m=7,  # Weekly seasonality
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_order=5,
            trace=False
        ) 
        return arima_model
        
    except Exception as e:
        warnings.warn(f"ARIMA fitting failed: {e}")
        return None

def evaluate_deep_arima_variant(neural_predictions: np.ndarray,
                               actuals: np.ndarray,
                               insample_data: np.ndarray,
                               scale: str,
                               training_residuals: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate Deep + ARIMA variant.
    
    Args:
        neural_predictions: Neural model predictions
        actuals: True values
        insample_data: Historical data for scaling metrics
        scale: Time scale
        training_residuals: Optional training residuals for ARIMA fitting
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not PMDARIMA_AVAILABLE:
        # Fallback to deep-only if ARIMA not available
        return evaluate_deep_only_variant(neural_predictions, actuals, insample_data, scale)
    
    # Fit ARIMA on residuals
    arima_model = fit_arima_on_residuals(neural_predictions, actuals, training_residuals)
    
    if arima_model is None:
        # Fallback to deep-only if ARIMA fitting failed
        return evaluate_deep_only_variant(neural_predictions, actuals, insample_data, scale)
    
    try:
        # Predict residuals
        n_forecast = len(neural_predictions) if neural_predictions.ndim == 1 else neural_predictions.shape[0]
        residual_forecast = arima_model.predict(n_periods=n_forecast)  # type: ignore
        
        # FIXED: Proper residual forecast application for multi-horizon
        # Combine neural predictions with ARIMA residual forecasts
        if neural_predictions.ndim > 1:
            # Multi-horizon case: apply residual correction per sample (not per horizon)
            combined_predictions = neural_predictions.copy()
            n_samples = neural_predictions.shape[0]
            
            # Ensure we have enough residual forecasts
            if len(residual_forecast) >= n_samples:
                # Apply one residual correction per sample (chronological order)
                for i in range(n_samples):
                    combined_predictions[i, :] += residual_forecast[i]
            else:
                warnings.warn(f"Insufficient residual forecasts ({len(residual_forecast)}) for samples ({n_samples})")
                # Fallback: cycle through available residuals
                for i in range(n_samples):
                    combined_predictions[i, :] += residual_forecast[i % len(residual_forecast)]
        else:
            # Single horizon case
            combined_predictions = neural_predictions + residual_forecast
        
        # Evaluate combined predictions
        return evaluate_deep_only_variant(combined_predictions, actuals, insample_data, scale)
        
    except Exception as e:
        warnings.warn(f"ARIMA prediction failed: {e}")
        # Fallback to deep-only
        return evaluate_deep_only_variant(neural_predictions, actuals, insample_data, scale)