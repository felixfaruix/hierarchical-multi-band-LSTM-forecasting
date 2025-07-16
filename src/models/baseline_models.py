"""
Baseline Forecasting Models

This module contains all baseline forecasting models used for comparison
in the hierarchical forecasting evaluation framework.

Models included:
- Naive (Random Walk)
- Seasonal Naive
- Exponential Smoothing
- ARIMA (with pmdarima)
"""

import numpy as np
import warnings
from typing import Optional
from abc import ABC, abstractmethod
import pmdarima as pm
import lightgbm as lgb


class BaselineForecaster(ABC):
    """Base class for baseline forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted = False
    
    @abstractmethod
    def fit(self, y_train: np.ndarray, X_train: Optional[np.ndarray] = None) -> None:
        """Fit the model to training data."""
        pass
    

    @abstractmethod
    def predict(self, horizon: int, X_test: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate forecasts for given horizon."""
        pass


class NaiveForecastModel(BaselineForecaster):
    """Naive forecast: last observation carried forward (Random Walk)."""
    
    def __init__(self):
        super().__init__("Naive")
        self.last_observation = None

    def fit(self, training_data: np.ndarray) -> None:
        if len(training_data) == 0:
            raise ValueError("Training data cannot be empty")
        self.last_observation = training_data[-1]
        self.fitted = True

    def predict(self, forecast_horizon: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(forecast_horizon, self.last_observation)

class SeasonalNaiveForecastModel(BaselineForecaster):
    """Seasonal naive forecast using seasonal patterns."""
    
    def __init__(self, seasonal_period: int):
        super().__init__(f"SeasonalNaive_{seasonal_period}")
        self.seasonal_period = seasonal_period
        self.training_data = None
    
    def fit(self, training_data: np.ndarray) -> None:
        if len(training_data) < self.seasonal_period:
            raise ValueError(f"Training data must have at least {self.seasonal_period} observations")
        self.training_data = training_data
        self.fitted = True
    
    def predict(self, forecast_horizon: int) -> np.ndarray:
        if not self.fitted or self.training_data is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Use seasonal pattern from training data
        seasonal_pattern = self.training_data[-self.seasonal_period:]
        forecast_values = []
        
        for step in range(forecast_horizon):
            seasonal_index = step % self.seasonal_period
            forecast_values.append(seasonal_pattern[seasonal_index])
        
        return np.array(forecast_values)

class ExponentialSmoothingModel(BaselineForecaster):
    """Simple exponential smoothing for comparison."""
    
    def __init__(self, smoothing_parameter: float = 0.3):
        super().__init__("ExponentialSmoothing")
        self.smoothing_parameter = smoothing_parameter
        self.smoothed_value = None
    
    def fit(self, training_data: np.ndarray) -> None:
        if len(training_data) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Simple exponential smoothing
        smoothed = training_data[0]
        for value in training_data[1:]:
            smoothed = self.smoothing_parameter * value + (1 - self.smoothing_parameter) * smoothed
        self.smoothed_value = smoothed
        self.fitted = True
    
    def predict(self, forecast_horizon: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(forecast_horizon, self.smoothed_value)


class ARIMAForecastModel(BaselineForecaster):
    """ARIMA model using pmdarima for automatic order selection."""
    
    def __init__(self, seasonal: bool = True, max_p: int = 3, max_q: int = 3, max_d: int = 2):
        super().__init__("ARIMA")
        self.seasonal = seasonal
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.fitted_model = None
    
    def fit(self, training_data: np.ndarray) -> None:
        try:
            self.fitted_model = pm.auto_arima(
                training_data,
                seasonal=self.seasonal,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=self.max_p, max_q=self.max_q, max_d=self.max_d
            )
            self.fitted = True
        except Exception as e:
            warnings.warn(f"ARIMA fitting failed: {e}")
            # Fallback to simple ARIMA(1,1,1)
            self.fitted_model = pm.ARIMA(order=(1, 1, 1))
            self.fitted_model.fit(training_data)
            self.fitted = True
    
    def predict(self, forecast_horizon: int) -> np.ndarray:
        if not self.fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts, _ = self.fitted_model.predict(n_periods=forecast_horizon, return_conf_int=True)
        return np.asarray(forecasts)


def create_baseline_models_by_scale() -> dict:
    """
    Create comprehensive baseline models following M4 competition standards.
    
    Returns:
        Dictionary with baseline models for each time scale
    """
    baseline_configurations = {
        'daily': {
            'naive': NaiveForecastModel(),
            'seasonal_naive_7': SeasonalNaiveForecastModel(7),
            'exponential_smoothing': ExponentialSmoothingModel(0.3),
            'arima_auto': ARIMAForecastModel(seasonal=True),
        },
        'weekly': {
            'naive': NaiveForecastModel(),
            'seasonal_naive_4': SeasonalNaiveForecastModel(4),  # Monthly cycle
            'exponential_smoothing': ExponentialSmoothingModel(0.2),
            'arima_auto': ARIMAForecastModel(seasonal=True),
        },
        'monthly': {
            'naive': NaiveForecastModel(),
            'seasonal_naive_12': SeasonalNaiveForecastModel(12),  # Annual cycle
            'exponential_smoothing': ExponentialSmoothingModel(0.1),
            'arima_auto': ARIMAForecastModel(seasonal=True),
        }
    }
    return baseline_configurations
