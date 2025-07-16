"""
Baseline Forecasting Models

This module contains all baseline forecasting models used for comparison
in the hierarchical forecasting evaluation framework.

Models included:
- Naive (Random Walk)
- Drift (Linear Trend)
- Mean (Historical Average)
- Seasonal Naive
- Exponential Smoothing
- ARIMA (with pmdarima)
- LightGBM (with lag features)
"""

import numpy as np
import warnings
from typing import Optional
from abc import ABC, abstractmethod

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
    
    def fit(self, training_data: np.ndarray, external_features: Optional[np.ndarray] = None) -> None:
        if len(training_data) == 0:
            raise ValueError("Training data cannot be empty")
        self.last_observation = training_data[-1]
        self.fitted = True
    
    def predict(self, forecast_horizon: int, external_features_test: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(forecast_horizon, self.last_observation)


class DriftForecastModel(BaselineForecaster):
    """Drift forecast: linear trend from first to last observation."""
    
    def __init__(self):
        super().__init__("Drift")
        self.last_observation = None
        self.drift_slope = None
    
    def fit(self, training_data: np.ndarray, external_features: Optional[np.ndarray] = None) -> None:
        if len(training_data) < 2:
            raise ValueError("Drift model requires at least 2 observations")
        
        self.last_observation = training_data[-1]
        # Calculate drift as average change per period
        self.drift_slope = (training_data[-1] - training_data[0]) / (len(training_data) - 1)
        self.fitted = True
    
    def predict(self, forecast_horizon: int, external_features_test: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.array([self.last_observation + self.drift_slope * (h + 1) for h in range(forecast_horizon)])


class MeanForecastModel(BaselineForecaster):
    """Historical mean forecast."""
    
    def __init__(self):
        super().__init__("Mean")
        self.historical_mean = None
    
    def fit(self, training_data: np.ndarray, external_features: Optional[np.ndarray] = None) -> None:
        if len(training_data) == 0:
            raise ValueError("Training data cannot be empty")
        self.historical_mean = np.mean(training_data)
        self.fitted = True
    
    def predict(self, forecast_horizon: int, external_features_test: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(forecast_horizon, self.historical_mean)


class SeasonalNaiveForecastModel(BaselineForecaster):
    """Seasonal naive forecast using seasonal patterns."""
    
    def __init__(self, seasonal_period: int):
        super().__init__(f"SeasonalNaive_{seasonal_period}")
        self.seasonal_period = seasonal_period
        self.training_data = None
    
    def fit(self, training_data: np.ndarray, external_features: Optional[np.ndarray] = None) -> None:
        if len(training_data) < self.seasonal_period:
            raise ValueError(f"Training data must have at least {self.seasonal_period} observations")
        self.training_data = training_data
        self.fitted = True
    
    def predict(self, forecast_horizon: int, external_features_test: Optional[np.ndarray] = None) -> np.ndarray:
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
    
    def fit(self, training_data: np.ndarray, external_features: Optional[np.ndarray] = None) -> None:
        if len(training_data) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Simple exponential smoothing
        smoothed = training_data[0]
        for value in training_data[1:]:
            smoothed = self.smoothing_parameter * value + (1 - self.smoothing_parameter) * smoothed
        self.smoothed_value = smoothed
        self.fitted = True
    
    def predict(self, forecast_horizon: int, external_features_test: Optional[np.ndarray] = None) -> np.ndarray:
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
    
    def fit(self, training_data: np.ndarray, external_features: Optional[np.ndarray] = None) -> None:
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima not available. Install with: pip install pmdarima")
        
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
    
    def predict(self, forecast_horizon: int, external_features_test: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts, _ = self.fitted_model.predict(n_periods=forecast_horizon, return_conf_int=True)
        return forecasts


class LightGBMResidualModel(BaselineForecaster):
    """LightGBM model for residual forecasting with lag features."""
    
    def __init__(self, lag_features: int = 14, **lgb_params):
        super().__init__("LightGBM")
        self.lag_features = lag_features
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            **lgb_params
        }
        self.trained_model = None
        self.training_data = None
    
    def _create_lag_feature_matrix(self, time_series: np.ndarray, start_index: int = 0) -> np.ndarray:
        """Create lag feature matrix for time series."""
        n_samples = len(time_series) - start_index - self.lag_features + 1
        if n_samples <= 0:
            raise ValueError(f"Not enough data for lag features. Need at least {self.lag_features} observations.")
        
        feature_matrix = np.zeros((n_samples, self.lag_features))
        
        for i in range(n_samples):
            actual_index = start_index + i
            feature_matrix[i, :] = time_series[actual_index:actual_index + self.lag_features]
        
        return feature_matrix
    
    def fit(self, training_data: np.ndarray, external_features: Optional[np.ndarray] = None) -> None:
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm not available. Install with: pip install lightgbm")
        
        if len(training_data) < self.lag_features + 10:
            raise ValueError(f"Need at least {self.lag_features + 10} observations for LightGBM")
        
        self.training_data = training_data.copy()
        
        # Create lag features
        lag_feature_matrix = self._create_lag_feature_matrix(training_data)
        target_values = training_data[self.lag_features:]
        
        # Add external features if provided
        if external_features is not None:
            external_aligned = external_features[self.lag_features:len(target_values) + self.lag_features]
            feature_matrix = np.column_stack([lag_feature_matrix, external_aligned])
        else:
            feature_matrix = lag_feature_matrix
        
        # Train LightGBM model
        training_dataset = lgb.Dataset(feature_matrix, label=target_values)
        self.trained_model = lgb.train(
            self.lgb_params,
            training_dataset,
            num_boost_round=100,
            valid_sets=[training_dataset],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        self.fitted = True
    
    def predict(self, forecast_horizon: int, external_features_test: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.fitted or self.trained_model is None or self.training_data is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        history_values = self.training_data[-self.lag_features:].tolist()
        
        for step in range(forecast_horizon):
            # Create features for current prediction step
            lag_features = np.array(history_values[-self.lag_features:]).reshape(1, -1)
            
            if external_features_test is not None and step < len(external_features_test):
                step_features = np.column_stack([lag_features, external_features_test[step].reshape(1, -1)])
            else:
                step_features = lag_features
            
            # Generate prediction
            prediction = self.trained_model.predict(step_features)[0]
            predictions.append(prediction)
            history_values.append(prediction)
        
        return np.array(predictions)


def create_baseline_models_by_scale() -> dict:
    """
    Create comprehensive baseline models following M4 competition standards.
    
    Returns:
        Dictionary with baseline models for each time scale
    """
    baseline_configurations = {
        'daily': {
            'naive': NaiveForecastModel(),
            'drift': DriftForecastModel(),
            'mean': MeanForecastModel(),
            'seasonal_naive_7': SeasonalNaiveForecastModel(7),
            'seasonal_naive_14': SeasonalNaiveForecastModel(14),
            'exponential_smoothing': ExponentialSmoothingModel(0.3),
            'arima_auto': ARIMAForecastModel(seasonal=True),
            'lightgbm_residual': LightGBMResidualModel(lag_features=14)
        },
        'weekly': {
            'naive': NaiveForecastModel(),
            'drift': DriftForecastModel(),
            'mean': MeanForecastModel(),
            'seasonal_naive_4': SeasonalNaiveForecastModel(4),  # Monthly cycle
            'seasonal_naive_13': SeasonalNaiveForecastModel(13),  # Quarterly cycle
            'exponential_smoothing': ExponentialSmoothingModel(0.2),
            'arima_auto': ARIMAForecastModel(seasonal=True),
            'lightgbm_residual': LightGBMResidualModel(lag_features=8)
        },
        'monthly': {
            'naive': NaiveForecastModel(),
            'drift': DriftForecastModel(),
            'mean': MeanForecastModel(),
            'seasonal_naive_12': SeasonalNaiveForecastModel(12),  # Annual cycle
            'exponential_smoothing': ExponentialSmoothingModel(0.1),
            'arima_auto': ARIMAForecastModel(seasonal=True),
            'lightgbm_residual': LightGBMResidualModel(lag_features=12)
        }
    }
    return baseline_configurations
