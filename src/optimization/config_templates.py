"""
Configuration templates for Optuna optimization

This module provides pre-configured optimization setups for different scenarios:
- Quick development optimization (few trials, fast convergence)
- Full research optimization (comprehensive search)
- Production optimization (balanced speed/accuracy)
- Multi-objective optimization templates

Author: Scientific Pipeline Framework
Version: 1.0
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class QuickOptimizationConfig:
    """Quick optimization for development and testing"""
    
    # Basic settings
    n_trials: int = 20
    timeout: int = 1800  # 30 minutes
    n_jobs: int = 1
    
    # Cross-validation
    cv_folds: int = 3
    test_size: float = 0.2
    val_size: float = 0.15
    
    # Training
    max_epochs: int = 50
    patience: int = 10
    
    # Search space (limited for quick optimization)
    hidden_dims: List[int] = None
    num_layers_range: tuple = (2, 4)
    dropout_range: tuple = (0.1, 0.3)
    learning_rate_range: tuple = (1e-4, 1e-2)
    batch_sizes: List[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256]
        if self.batch_sizes is None:
            self.batch_sizes = [32, 64, 128]


@dataclass
class ResearchOptimizationConfig:
    """Comprehensive optimization for research purposes"""
    
    # Basic settings
    n_trials: int = 200
    timeout: int = 21600  # 6 hours
    n_jobs: int = 4
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.15
    
    # Training
    max_epochs: int = 150
    patience: int = 20
    
    # Search space (comprehensive)
    hidden_dims: List[int] = None
    num_layers_range: tuple = (2, 8)
    dropout_range: tuple = (0.0, 0.6)
    learning_rate_range: tuple = (1e-5, 1e-1)
    batch_sizes: List[int] = None
    attention_heads: List[int] = None
    hierarchy_levels_range: tuple = (2, 5)
    frequency_bands_range: tuple = (2, 6)
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512, 768]
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64, 128, 256]
        if self.attention_heads is None:
            self.attention_heads = [2, 4, 8, 16, 32]


@dataclass
class ProductionOptimizationConfig:
    """Balanced optimization for production deployment"""
    
    # Basic settings
    n_trials: int = 100
    timeout: int = 10800  # 3 hours
    n_jobs: int = 2
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.15
    
    # Training
    max_epochs: int = 100
    patience: int = 15
    
    # Search space (focused on production viability)
    hidden_dims: List[int] = None
    num_layers_range: tuple = (2, 6)
    dropout_range: tuple = (0.1, 0.4)
    learning_rate_range: tuple = (1e-4, 1e-2)
    batch_sizes: List[int] = None
    
    # Production constraints
    max_model_size_mb: float = 100.0  # Model size constraint
    max_inference_time_ms: float = 500.0  # Inference time constraint
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512]
        if self.batch_sizes is None:
            self.batch_sizes = [32, 64, 128]


@dataclass
class MultiObjectiveOptimizationConfig:
    """Multi-objective optimization configuration"""
    
    # Basic settings
    n_trials: int = 150
    timeout: int = 14400  # 4 hours
    n_jobs: int = 2
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.15
    
    # Training
    max_epochs: int = 100
    patience: int = 15
    
    # Multi-objective settings
    objectives: List[str] = None
    objective_weights: Dict[str, float] = None
    pareto_front_size: int = 10
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['mape', 'directional_accuracy', 'model_complexity']
        if self.objective_weights is None:
            self.objective_weights = {
                'mape': 0.5,
                'directional_accuracy': 0.3,
                'model_complexity': 0.2
            }


class OptimizationConfigManager:
    """Manager for optimization configurations"""
    
    PRESET_CONFIGS = {
        'quick': QuickOptimizationConfig,
        'research': ResearchOptimizationConfig,
        'production': ProductionOptimizationConfig,
        'multi_objective': MultiObjectiveOptimizationConfig
    }
    
    @classmethod
    def get_config(cls, preset: str, **overrides) -> Dict[str, Any]:
        """Get configuration with optional overrides"""
        if preset not in cls.PRESET_CONFIGS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(cls.PRESET_CONFIGS.keys())}")
        
        config_class = cls.PRESET_CONFIGS[preset]
        config = config_class()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        return asdict(config)
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, filepath: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @classmethod
    def create_azure_config(cls, base_preset: str, **azure_overrides) -> Dict[str, Any]:
        """Create Azure ML specific configuration"""
        config = cls.get_config(base_preset)
        
        # Add Azure ML specific settings
        azure_settings = {
            'use_azure_ml': True,
            'subscription_id': azure_overrides.get('subscription_id'),
            'resource_group': azure_overrides.get('resource_group'),
            'workspace_name': azure_overrides.get('workspace_name'),
            'compute_cluster': azure_overrides.get('compute_cluster', 'ethanol-optimization'),
            'instance_type': azure_overrides.get('instance_type', 'Standard_DS4_v2'),
            'instance_count': azure_overrides.get('instance_count', 4),
            'environment_name': azure_overrides.get('environment_name', 'ethanol-forecasting-env'),
            'experiment_name': azure_overrides.get('experiment_name', 'ethanol_optimization'),
            'use_mlflow': azure_overrides.get('use_mlflow', True),
            'register_best_model': azure_overrides.get('register_best_model', True)
        }
        
        config.update(azure_settings)
        return config


# Predefined optimization strategies
OPTIMIZATION_STRATEGIES = {
    'aggressive_search': {
        'sampler': 'TPE',
        'pruner': 'MedianPruner',
        'n_startup_trials': 20,
        'n_warmup_steps': 10,
        'pruning_percentile': 30
    },
    
    'conservative_search': {
        'sampler': 'TPE',
        'pruner': 'SuccessiveHalvingPruner',
        'n_startup_trials': 10,
        'n_warmup_steps': 5,
        'pruning_percentile': 15
    },
    
    'random_baseline': {
        'sampler': 'RandomSampler',
        'pruner': None,
        'n_startup_trials': 0,
        'n_warmup_steps': 0
    },
    
    'grid_search': {
        'sampler': 'GridSampler',
        'pruner': None,
        'n_startup_trials': 0,
        'n_warmup_steps': 0
    }
}


def create_optimization_config_file(
    preset: str = 'production',
    output_file: str = 'optimization_config.json',
    azure_ml: bool = False,
    **overrides
) -> str:
    """Create and save optimization configuration file"""
    
    if azure_ml:
        config = OptimizationConfigManager.create_azure_config(preset, **overrides)
    else:
        config = OptimizationConfigManager.get_config(preset, **overrides)
    
    # Add optimization strategy
    strategy = overrides.get('strategy', 'aggressive_search')
    if strategy in OPTIMIZATION_STRATEGIES:
        config['optimization_strategy'] = OPTIMIZATION_STRATEGIES[strategy]
    
    # Save configuration
    OptimizationConfigManager.save_config(config, output_file)
    
    return output_file


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate optimization configuration"""
    errors = []
    
    # Required fields
    required_fields = ['n_trials', 'cv_folds', 'max_epochs']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Value ranges
    if config.get('n_trials', 0) <= 0:
        errors.append("n_trials must be positive")
    
    if config.get('cv_folds', 0) <= 1:
        errors.append("cv_folds must be > 1")
    
    if config.get('test_size', 0) <= 0 or config.get('test_size', 1) >= 1:
        errors.append("test_size must be between 0 and 1")
    
    # Azure ML specific validation
    if config.get('use_azure_ml', False):
        azure_required = ['subscription_id', 'resource_group', 'workspace_name']
        for field in azure_required:
            if not config.get(field):
                errors.append(f"Azure ML requires {field}")
    
    return errors


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create optimization configuration")
    parser.add_argument("--preset", choices=['quick', 'research', 'production', 'multi_objective'],
                       default='production', help="Configuration preset")
    parser.add_argument("--output", default="optimization_config.json", help="Output file")
    parser.add_argument("--azure", action="store_true", help="Create Azure ML configuration")
    parser.add_argument("--strategy", choices=list(OPTIMIZATION_STRATEGIES.keys()),
                       default="aggressive_search", help="Optimization strategy")
    parser.add_argument("--n_trials", type=int, help="Override number of trials")
    parser.add_argument("--timeout", type=int, help="Override timeout (seconds)")
    
    args = parser.parse_args()
    
    # Create overrides dict
    overrides = {}
    if args.n_trials:
        overrides['n_trials'] = args.n_trials
    if args.timeout:
        overrides['timeout'] = args.timeout
    if args.strategy:
        overrides['strategy'] = args.strategy
    
    # Create config file
    config_file = create_optimization_config_file(
        preset=args.preset,
        output_file=args.output,
        azure_ml=args.azure,
        **overrides
    )
    
    print(f"Configuration saved to: {config_file}")
    
    # Validate config
    config = OptimizationConfigManager.load_config(config_file)
    errors = validate_config(config)
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")
