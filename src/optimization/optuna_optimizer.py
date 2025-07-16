"""
Optuna Hyperparameter Optimization for Hierarchical Multi-Band Ethanol Forecasting

This module provides comprehensive hyperparameter optimization using Optuna with support for:
- Hierarchical model architectures
- Multi-band frequency decomposition
- Cross-validation with temporal splits
- Integration with AzureML for distributed training
- Advanced pruning strategies
- Multi-objective optimization

Author: Scientific Pipeline Framework
Version: 1.0
"""

import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from optuna.trial import TrialState
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import pickle
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.model import HierForecastNet
from data.timeseries_datamodule import TimeSeriesDataModule
from data.dataset_preprocessing import FinancialDataPreprocessor
from evaluation.evaluation_pipeline import ForecastingMetrics, ModelEvaluator
from evaluation.ts_cross_validation import TimeSeriesCrossValidator
from utils.data_quality import DataQualityAssessor
from utils.advanced_quality_assessment import AdvancedQualityAssessor
from train import LightningHierForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationConfig:
    """Configuration class for hyperparameter optimization"""
    
    def __init__(self):
        # Optimization settings
        self.n_trials = 100
        self.n_jobs = 1  # For AzureML, usually single job per node
        self.timeout = 3600 * 6  # 6 hours
        self.n_startup_trials = 10
        self.n_warmup_steps = 5
        
        # Cross-validation settings
        self.cv_folds = 5
        self.test_size = 0.2
        self.val_size = 0.15
        
        # Training settings
        self.max_epochs = 100
        self.patience = 15
        self.min_delta = 1e-4
        
        # Pruning settings
        self.pruning_percentile = 25
        self.pruning_interval = 10
        
        # Multi-objective settings
        self.optimize_multiple_metrics = True
        self.primary_metric = 'mape'
        self.secondary_metric = 'directional_accuracy'
        
        # Azure ML settings
        self.use_azureml_logging = True
        self.experiment_name = "ethanol_hierarchical_optimization"


class HyperparameterSpace:
    """Define the hyperparameter search space"""
    
    @staticmethod
    def suggest_model_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest model architecture hyperparameters"""
        return {
            # Network architecture
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'attention_heads': trial.suggest_categorical('attention_heads', [4, 8, 16]),
            
            # Hierarchical settings
            'hierarchy_levels': trial.suggest_int('hierarchy_levels', 2, 4),
            'reconciliation_method': trial.suggest_categorical(
                'reconciliation_method', ['bottom_up', 'top_down', 'mint', 'optimal']
            ),
            
            # Multi-band settings
            'frequency_bands': trial.suggest_int('frequency_bands', 2, 5),
            'band_decomposition': trial.suggest_categorical(
                'band_decomposition', ['wavelet', 'fft', 'emd', 'stl']
            ),
            
            # Sequence settings
            'lookback_window': trial.suggest_categorical('lookback_window', [30, 60, 90, 120]),
            'forecast_horizon': trial.suggest_categorical('forecast_horizon', [7, 14, 21, 30]),
            
            # Attention mechanisms
            'use_temporal_attention': trial.suggest_categorical('use_temporal_attention', [True, False]),
            'use_feature_attention': trial.suggest_categorical('use_feature_attention', [True, False]),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3),
        }
    
    @staticmethod
    def suggest_training_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest training hyperparameters"""
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop'])
        
        params = {
            # Optimizer settings
            'optimizer': optimizer_name,
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            
            # Scheduler settings
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'step', 'exponential']),
            'scheduler_patience': trial.suggest_int('scheduler_patience', 5, 15),
            'scheduler_factor': trial.suggest_float('scheduler_factor', 0.1, 0.8),
            
            # Batch settings
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'gradient_clip_val': trial.suggest_float('gradient_clip_val', 0.1, 2.0),
            
            # Loss function
            'loss_function': trial.suggest_categorical(
                'loss_function', ['mse', 'mae', 'huber', 'quantile', 'mape']
            ),
            'loss_weights': {
                'mse_weight': trial.suggest_float('mse_weight', 0.1, 1.0),
                'mae_weight': trial.suggest_float('mae_weight', 0.1, 1.0),
                'directional_weight': trial.suggest_float('directional_weight', 0.0, 0.5),
            }
        }
        
        # Optimizer-specific parameters
        if optimizer_name == 'adam':
            params['beta1'] = trial.suggest_float('beta1', 0.8, 0.99)
            params['beta2'] = trial.suggest_float('beta2', 0.9, 0.999)
        elif optimizer_name == 'adamw':
            params['beta1'] = trial.suggest_float('beta1', 0.8, 0.99)
            params['beta2'] = trial.suggest_float('beta2', 0.9, 0.999)
            params['eps'] = trial.suggest_float('eps', 1e-8, 1e-6, log=True)
        elif optimizer_name == 'rmsprop':
            params['alpha'] = trial.suggest_float('alpha', 0.9, 0.999)
            params['momentum'] = trial.suggest_float('momentum', 0.0, 0.9)
        
        return params
    
    @staticmethod
    def suggest_data_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest data preprocessing hyperparameters"""
        return {
            # Feature engineering
            'use_calendar_features': trial.suggest_categorical('use_calendar_features', [True, False]),
            'use_lag_features': trial.suggest_categorical('use_lag_features', [True, False]),
            'max_lags': trial.suggest_int('max_lags', 1, 14) if trial.suggest_categorical('use_lag_features', [True, False]) else 0,
            
            # Scaling
            'scaler_type': trial.suggest_categorical('scaler_type', ['standard', 'minmax', 'robust']),
            'target_transform': trial.suggest_categorical('target_transform', ['none', 'log', 'boxcox', 'diff']),
            
            # Feature selection
            'feature_selection': trial.suggest_categorical('feature_selection', [True, False]),
            'max_features': trial.suggest_int('max_features', 5, 20) if trial.suggest_categorical('feature_selection', [True, False]) else None,
            
            # Data augmentation
            'use_noise_injection': trial.suggest_categorical('use_noise_injection', [True, False]),
            'noise_level': trial.suggest_float('noise_level', 0.01, 0.1) if trial.suggest_categorical('use_noise_injection', [True, False]) else 0.0,
            
            # Outlier handling
            'outlier_threshold': trial.suggest_float('outlier_threshold', 2.0, 4.0),
            'outlier_method': trial.suggest_categorical('outlier_method', ['hampel', 'iqr', 'isolation_forest']),
        }


class OptimizationObjective:
    """Optimization objective function for Optuna"""
    
    def __init__(self, config: OptimizationConfig, data_path: str, output_dir: str):
        self.config = config
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_preprocessor = FinancialDataPreprocessor()
        self.quality_assessor = DataQualityAssessor()
        self.advanced_quality_assessor = AdvancedQualityAssessor()
        self.evaluator = ModelEvaluator()
        self.cv_validator = TimeSeriesCrossValidator(
            n_splits=config.cv_folds,
            test_size=config.test_size
        )
        
        # Load and preprocess data once
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """Load and prepare data for optimization"""
        logger.info("Loading and preparing data for optimization...")
        
        # Load raw data
        raw_data = {}
        for file_path in self.data_path.glob("*.csv"):
            dataset_name = file_path.stem
            raw_data[dataset_name] = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Quality assessment
        quality_results = {}
        for name, df in raw_data.items():
            basic_metrics = self.quality_assessor.comprehensive_assessment(df)
            advanced_metrics = self.advanced_quality_assessor.comprehensive_advanced_assessment(df)
            quality_results[name] = {'basic': basic_metrics, 'advanced': advanced_metrics}
        
        # Basic preprocessing
        self.processed_data = {}
        for name, df in raw_data.items():
            processed_df = self.data_preprocessor.preprocess_financial_data(
                df, target_column=df.columns[0]  # Assume first column is target
            )
            self.processed_data[name] = processed_df
        
        logger.info(f"Loaded {len(self.processed_data)} datasets for optimization")
    
    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        """Objective function for Optuna optimization"""
        try:
            # Suggest hyperparameters
            model_params = HyperparameterSpace.suggest_model_params(trial)
            training_params = HyperparameterSpace.suggest_training_params(trial)
            data_params = HyperparameterSpace.suggest_data_params(trial)
            
            # Prepare data with trial-specific parameters
            datamodule = self._prepare_datamodule(trial, data_params)
            
            # Create model
            model = self._create_model(trial, model_params, training_params)
            
            # Setup training
            trainer = self._setup_trainer(trial, training_params)
            
            # Cross-validation
            cv_scores = self._cross_validate(model, datamodule, trainer, trial)
            
            # Calculate objective value(s)
            if self.config.optimize_multiple_metrics:
                primary_score = np.mean([scores[self.config.primary_metric] for scores in cv_scores])
                secondary_score = np.mean([scores[self.config.secondary_metric] for scores in cv_scores])
                return primary_score, secondary_score
            else:
                return np.mean([scores[self.config.primary_metric] for scores in cv_scores])
        
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            if self.config.optimize_multiple_metrics:
                return float('inf'), float('-inf')
            else:
                return float('inf')
    
    def _prepare_datamodule(self, trial: optuna.Trial, data_params: Dict[str, Any]) -> TimeSeriesDataModule:
        """Prepare data module with trial-specific parameters"""
        # Use first dataset for optimization (can be extended for multi-dataset)
        main_dataset = list(self.processed_data.values())[0]
        
        # Apply trial-specific preprocessing
        processed_data = main_dataset.copy()
        
        # Feature engineering based on parameters
        if data_params['use_calendar_features']:
            processed_data = self._add_calendar_features(processed_data)
        
        if data_params['use_lag_features']:
            processed_data = self._add_lag_features(processed_data, data_params['max_lags'])
        
        # Scaling
        if data_params['scaler_type'] == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif data_params['scaler_type'] == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:  # robust
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        # Create datamodule
        datamodule = TimeSeriesDataModule(
            data=processed_data,
            target_column=processed_data.columns[0],
            sequence_length=data_params.get('lookback_window', 60),
            prediction_length=data_params.get('forecast_horizon', 7),
            batch_size=trial.params['batch_size'],
            train_size=1.0 - self.config.test_size - self.config.val_size,
            val_size=self.config.val_size,
            test_size=self.config.test_size,
            scaler=scaler
        )
        
        return datamodule
    
    def _create_model(self, trial: optuna.Trial, model_params: Dict[str, Any], 
                     training_params: Dict[str, Any]) -> LightningHierForecaster:
        """Create model with trial-specific parameters"""
        
        # Get input dimension from datamodule (will be set after setup)
        input_dim = len(list(self.processed_data.values())[0].columns)
        
        # Create base model
        base_model = HierForecastNet(
            input_dim=input_dim,
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            output_dim=model_params['forecast_horizon'],
            dropout_rate=model_params['dropout_rate'],
            attention_heads=model_params['attention_heads'],
            hierarchy_levels=model_params['hierarchy_levels'],
            frequency_bands=model_params['frequency_bands'],
            use_temporal_attention=model_params['use_temporal_attention'],
            use_feature_attention=model_params['use_feature_attention']
        )
        
        # Create Lightning wrapper
        lightning_model = LightningHierForecaster(
            model=base_model,
            learning_rate=training_params['learning_rate'],
            weight_decay=training_params['weight_decay'],
            optimizer_name=training_params['optimizer'],
            scheduler_name=training_params['scheduler'],
            loss_function=training_params['loss_function'],
            **training_params.get('loss_weights', {})
        )
        
        return lightning_model
    
    def _setup_trainer(self, trial: optuna.Trial, training_params: Dict[str, Any]) -> pl.Trainer:
        """Setup PyTorch Lightning trainer with pruning"""
        
        callbacks = [
            # Optuna pruning
            PyTorchLightningPruningCallback(trial, monitor=f"val_{self.config.primary_metric}"),
            
            # Early stopping
            EarlyStopping(
                monitor=f"val_{self.config.primary_metric}",
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                mode='min' if self.config.primary_metric in ['mse', 'mae', 'mape'] else 'max'
            ),
            
            # Model checkpointing
            ModelCheckpoint(
                dirpath=self.output_dir / f"trial_{trial.number}",
                filename="best_model",
                monitor=f"val_{self.config.primary_metric}",
                mode='min' if self.config.primary_metric in ['mse', 'mae', 'mape'] else 'max',
                save_top_k=1
            )
        ]
        
        # Setup logger
        logger_instance = None
        if self.config.use_azureml_logging:
            try:
                logger_instance = WandbLogger(
                    project=self.config.experiment_name,
                    name=f"trial_{trial.number}",
                    tags=[f"optuna_optimization"]
                )
            except Exception as e:
                logger.warning(f"Failed to setup W&B logger: {e}")
        
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            callbacks=callbacks,
            logger=logger_instance,
            enable_progress_bar=False,  # Reduce noise in optimization
            enable_model_summary=False,
            gradient_clip_val=training_params.get('gradient_clip_val', 1.0),
            deterministic=True,  # For reproducibility
            devices=1 if torch.cuda.is_available() else None,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        )
        
        return trainer
    
    def _cross_validate(self, model: LightningHierForecaster, datamodule: TimeSeriesDataModule, 
                       trainer: pl.Trainer, trial: optuna.Trial) -> List[Dict[str, float]]:
        """Perform cross-validation"""
        cv_scores = []
        
        # Get data splits
        splits = self.cv_validator.split(datamodule.data)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Create fold-specific datamodule
            fold_datamodule = datamodule.create_fold(train_idx, val_idx)
            fold_datamodule.setup()
            
            # Clone model for this fold
            fold_model = type(model)(
                model=model.model,
                learning_rate=model.learning_rate,
                weight_decay=model.weight_decay,
                optimizer_name=model.optimizer_name,
                scheduler_name=model.scheduler_name,
                loss_function=model.loss_function
            )
            
            # Train model
            trainer.fit(fold_model, datamodule=fold_datamodule)
            
            # Evaluate
            val_results = trainer.validate(fold_model, datamodule=fold_datamodule, verbose=False)
            
            # Extract metrics
            fold_scores = {}
            for result in val_results:
                for key, value in result.items():
                    if key.startswith('val_'):
                        metric_name = key.replace('val_', '')
                        fold_scores[metric_name] = value
            
            cv_scores.append(fold_scores)
            
            # Report intermediate value for pruning
            if len(cv_scores) > 0:
                current_score = np.mean([scores[self.config.primary_metric] for scores in cv_scores])
                trial.report(current_score, fold)
        
        return cv_scores
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar features to dataframe"""
        df = df.copy()
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        # Binary features
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_year_end'] = df.index.is_year_end.astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, max_lags: int) -> pd.DataFrame:
        """Add lag features to dataframe"""
        df = df.copy()
        target_col = df.columns[0]
        
        for lag in range(1, max_lags + 1):
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Drop rows with NaN values from lagging
        df = df.dropna()
        
        return df


class EthanolOptimizer:
    """Main optimizer class for ethanol forecasting hyperparameters"""
    
    def __init__(self, config: OptimizationConfig, data_path: str, output_dir: str):
        self.config = config
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup study
        self.study = None
        self._setup_study()
    
    def _setup_study(self):
        """Setup Optuna study with appropriate sampler and pruner"""
        
        # Choose sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            n_ei_candidates=24,
            seed=42
        )
        
        # Choose pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps,
            interval_steps=self.config.pruning_interval
        )
        
        # Create study
        if self.config.optimize_multiple_metrics:
            directions = ['minimize', 'maximize']  # Minimize MAPE, maximize directional accuracy
            self.study = optuna.create_study(
                directions=directions,
                sampler=sampler,
                pruner=pruner,
                study_name=f"ethanol_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            direction = 'minimize' if self.config.primary_metric in ['mse', 'mae', 'mape'] else 'maximize'
            self.study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                study_name=f"ethanol_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def optimize(self) -> optuna.Study:
        """Run hyperparameter optimization"""
        logger.info(f"Starting optimization with {self.config.n_trials} trials...")
        
        # Create objective function
        objective = OptimizationObjective(self.config, self.data_path, self.output_dir)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            callbacks=[self._log_trial_callback]
        )
        
        # Save results
        self._save_results()
        
        return self.study
    
    def _log_trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback to log trial results"""
        if trial.state == TrialState.COMPLETE:
            if self.config.optimize_multiple_metrics:
                values = trial.values
                logger.info(f"Trial {trial.number}: {self.config.primary_metric}={values[0]:.4f}, "
                           f"{self.config.secondary_metric}={values[1]:.4f}")
            else:
                logger.info(f"Trial {trial.number}: {self.config.primary_metric}={trial.value:.4f}")
        elif trial.state == TrialState.PRUNED:
            logger.info(f"Trial {trial.number}: Pruned")
        elif trial.state == TrialState.FAIL:
            logger.info(f"Trial {trial.number}: Failed")
    
    def _save_results(self):
        """Save optimization results"""
        # Save study
        study_path = self.output_dir / "optuna_study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        # Save best parameters
        if self.config.optimize_multiple_metrics:
            # For multi-objective, save Pareto front
            pareto_trials = self.study.best_trials
            best_params = [trial.params for trial in pareto_trials]
            best_values = [trial.values for trial in pareto_trials]
            
            results = {
                'pareto_front_params': best_params,
                'pareto_front_values': best_values,
                'n_pareto_solutions': len(pareto_trials)
            }
        else:
            # Single objective
            results = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'best_trial': self.study.best_trial.number
            }
        
        # Add study statistics
        results.update({
            'n_trials': len(self.study.trials),
            'n_complete_trials': len([t for t in self.study.trials if t.state == TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in self.study.trials if t.state == TrialState.PRUNED]),
            'n_failed_trials': len([t for t in self.study.trials if t.state == TrialState.FAIL])
        })
        
        # Save as JSON
        results_path = self.output_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def get_best_model_config(self) -> Dict[str, Any]:
        """Get best model configuration for deployment"""
        if self.config.optimize_multiple_metrics:
            # For multi-objective, return the solution with best primary metric
            best_trial = min(self.study.best_trials, key=lambda t: t.values[0])
            return best_trial.params
        else:
            return self.study.best_params
    
    def create_visualization_report(self) -> str:
        """Create visualization report of optimization results"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create visualization directory
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Optimization history
            if self.config.optimize_multiple_metrics:
                fig = optuna.visualization.plot_pareto_front(self.study)
            else:
                fig = optuna.visualization.plot_optimization_history(self.study)
            
            fig.write_html(viz_dir / "optimization_history.html")
            
            # Parameter importance
            try:
                fig = optuna.visualization.plot_param_importances(self.study)
                fig.write_html(viz_dir / "parameter_importance.html")
            except Exception as e:
                logger.warning(f"Could not create parameter importance plot: {e}")
            
            # Parallel coordinate plot
            try:
                fig = optuna.visualization.plot_parallel_coordinate(self.study)
                fig.write_html(viz_dir / "parallel_coordinate.html")
            except Exception as e:
                logger.warning(f"Could not create parallel coordinate plot: {e}")
            
            return str(viz_dir)
            
        except ImportError:
            logger.warning("Plotly not available for visualization")
            return "Visualization skipped - plotly not available"


def main():
    """Main function for running optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ethanol Forecasting Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=21600, help="Timeout in seconds (6 hours)")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--multi_objective", action="store_true", help="Use multi-objective optimization")
    parser.add_argument("--experiment_name", type=str, default="ethanol_optimization", 
                       help="Experiment name for logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizationConfig()
    config.n_trials = args.n_trials
    config.timeout = args.timeout
    config.cv_folds = args.cv_folds
    config.optimize_multiple_metrics = args.multi_objective
    config.experiment_name = args.experiment_name
    
    # Run optimization
    optimizer = EthanolOptimizer(config, args.data_path, args.output_dir)
    study = optimizer.optimize()
    
    # Create visualization report
    viz_path = optimizer.create_visualization_report()
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETED")
    print("="*80)
    
    if config.optimize_multiple_metrics:
        print(f"Number of Pareto solutions: {len(study.best_trials)}")
        for i, trial in enumerate(study.best_trials):
            print(f"Solution {i+1}: {config.primary_metric}={trial.values[0]:.4f}, "
                 f"{config.secondary_metric}={trial.values[1]:.4f}")
    else:
        print(f"Best {config.primary_metric}: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
    
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Visualizations saved to: {viz_path}")
    print("="*80)


if __name__ == "__main__":
    main()
