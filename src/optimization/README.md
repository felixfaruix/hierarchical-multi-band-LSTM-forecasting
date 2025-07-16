# Optimization Pipeline Documentation

## Overview

The optimization pipeline provides comprehensive hyperparameter optimization for the Ethanol Forecasting project using Optuna with Azure ML integration for distributed computing.

## Quick Start

### 1. Setup Environment

```bash
# Setup and validate environment
python src/optimization/setup_optimization.py --install-deps --test

# For Azure ML support
python src/optimization/setup_optimization.py --install-deps --azure --azure-setup
```

### 2. Local Optimization

```bash
# Quick test (3 trials, 1 minute)
python src/optimization/run_optimization.py --mode local --preset quick --data_path ./raw_data

# Research optimization (50 trials, 2 hours)
python src/optimization/run_optimization.py --mode local --preset research --data_path ./raw_data

# Production optimization (500 trials, 8 hours)
python src/optimization/run_optimization.py --mode local --preset production --data_path ./raw_data
```

### 3. Azure ML Optimization

```bash
# Submit Azure ML job
python src/optimization/run_optimization.py --mode azure --preset production --data_path ./raw_data

# Monitor running job
python src/optimization/run_optimization.py --mode monitor --job_name <job-name>

# Collect results when complete
python src/optimization/run_optimization.py --mode collect --job_name <job-name>
```

### 4. Generate Reports

```bash
# Create comprehensive summary report
python src/optimization/run_optimization.py --mode report
```

## Configuration

### Built-in Presets

- **quick**: 10 trials, 5 minutes - for testing
- **research**: 50 trials, 2 hours - for exploration
- **production**: 500 trials, 8 hours - for final optimization
- **multi_objective**: 200 trials, 4 hours - for Pareto optimization

### Custom Configuration

Create a JSON config file (see `src/optimization/examples/custom_config.json`):

```bash
python src/optimization/run_optimization.py --mode local --config my_config.json --data_path ./raw_data
```

### Configuration Override

```bash
# Override specific parameters
python src/optimization/run_optimization.py \
  --mode local \
  --preset production \
  --n_trials 1000 \
  --timeout 14400 \
  --cv_folds 10 \
  --data_path ./raw_data
```

## Azure ML Integration

### Prerequisites

1. Azure ML workspace configured
2. Compute cluster created
3. Azure CLI authenticated

### Configuration

Update Azure ML settings in config:

```json
{
  "azure_ml": {
    "use_azure_ml": true,
    "subscription_id": "your-subscription-id",
    "resource_group": "your-resource-group", 
    "workspace_name": "your-workspace-name",
    "compute_cluster": "cpu-cluster",
    "instance_count": 8,
    "max_concurrent_trials": 16,
    "experiment_name": "ethanol-optimization"
  }
}
```

### Benefits

- **Distributed Computing**: Parallel trial execution across multiple nodes
- **Scalability**: Scale from 1 to 100+ compute instances
- **Cost Efficiency**: Pay-per-use with automatic scaling
- **Experiment Tracking**: Full MLflow integration with Azure ML
- **Artifact Management**: Automatic model and result storage

## Hyperparameter Space

The optimization covers:

### Model Architecture
- Hidden layer sizes: [64,32] to [512,256]
- Dropout rates: 0.1 to 0.5
- Activation functions: ReLU, Tanh, GELU, Swish
- Lookback windows: 30 to 180 days
- Prediction horizons: 7 to 30 days

### Training Parameters
- Learning rates: 1e-5 to 1e-2 (log scale)
- Batch sizes: 16, 32, 64, 128
- Max epochs: 50 to 300
- Early stopping patience: 10 to 50 epochs
- Weight decay: 1e-6 to 1e-3 (log scale)
- Gradient clipping: 0.1 to 2.0

### Feature Engineering
- Base features: price, volume, calendar
- Optional: technical indicators, external factors
- Feature combinations optimized automatically

## Results and Monitoring

### Output Structure

```
optimization_results/
├── optimization_results.json         # Main results
├── data_validation_report.json       # Data quality metrics
├── azure_job_info.json              # Azure ML job info
├── optuna_study.db                   # Optuna database
├── plots/                            # Visualization plots
│   ├── optimization_history.html
│   ├── param_importances.html
│   ├── parallel_coordinate.html
│   └── slice_plot.html
├── models/                           # Best models (if enabled)
├── logs/                             # Training logs
└── optimization_summary_report.html  # Comprehensive report
```

### Key Metrics

- **RMSE**: Root Mean Square Error (primary)
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Directional Accuracy**: Trend prediction accuracy

### Visualization

Automatic generation of:
- Optimization history plots
- Parameter importance analysis
- Hyperparameter correlation heatmaps
- Cross-validation performance distributions
- Pareto front plots (multi-objective)

## Advanced Features

### Multi-Objective Optimization

Optimize multiple metrics simultaneously:

```bash
python src/optimization/run_optimization.py \
  --mode local \
  --preset multi_objective \
  --data_path ./raw_data
```

Returns Pareto-optimal solutions balancing:
- Accuracy vs. Model Complexity
- Performance vs. Training Time
- RMSE vs. Directional Accuracy

### Pruning and Early Stopping

- **Trial Pruning**: Stop unpromising trials early
- **Median Pruner**: Prune trials performing worse than median
- **Successive Halving**: Allocate more resources to promising trials
- **Hyperband**: Bandit-based resource allocation

### Cross-Validation Strategies

- **Time Series Split**: Respects temporal order
- **Walk-Forward Validation**: Expanding window approach
- **Blocked Cross-Validation**: Account for temporal dependencies
- **Nested CV**: Hyperparameter optimization within CV

## Troubleshooting

### Common Issues

1. **Import Errors**: Run setup script to validate environment
2. **Memory Issues**: Reduce batch size or model complexity
3. **Azure ML Connection**: Check credentials and workspace config
4. **Data Quality**: Review validation report for issues

### Debugging

```bash
# Verbose logging
python src/optimization/run_optimization.py --mode local --preset quick --data_path ./raw_data --verbose

# Check setup
python src/optimization/setup_optimization.py --test --verbose

# Validate configuration
python -c "from optimization.config_templates import validate_config; print(validate_config('config.json'))"
```

### Performance Optimization

1. **Parallel Trials**: Use Azure ML for parallel execution
2. **GPU Acceleration**: Enable GPU compute for large models
3. **Data Caching**: Cache preprocessed data between trials
4. **Smart Sampling**: Use TPE sampler for efficient exploration

## Integration with Training Pipeline

The optimization pipeline integrates seamlessly with existing training:

```python
# Use optimized hyperparameters
from optimization.optuna_optimizer import EthanolOptimizer

# Load best parameters
optimizer = EthanolOptimizer.load_results("./optimization_results")
best_params = optimizer.get_best_params()

# Apply to training
from train import train_model
model = train_model(data_path, **best_params)
```

## API Reference

### Main Classes

- `EthanolOptimizer`: Core optimization engine
- `OptimizationConfig`: Configuration management
- `AzureMLOptimizer`: Azure ML integration
- `OptimizationRunner`: High-level interface

### Key Methods

- `optimize()`: Run optimization
- `create_visualization_report()`: Generate plots
- `collect_results()`: Retrieve Azure ML results
- `monitor_job()`: Check Azure ML job status

## Best Practices

1. **Start Small**: Use 'quick' preset for initial testing
2. **Data Quality**: Always validate data before optimization
3. **Resource Planning**: Monitor Azure ML costs and quotas
4. **Experiment Tracking**: Use meaningful experiment names
5. **Result Analysis**: Review parameter importance plots
6. **Model Validation**: Cross-validate optimized models
7. **Documentation**: Save configuration and results metadata

## Support

For issues and questions:
1. Check troubleshooting section
2. Review setup validation output
3. Check Azure ML studio for distributed jobs
4. Examine log files in results directory

## Examples

See `src/optimization/examples/` for:
- Custom configuration templates
- Azure ML setup examples
- Integration code samples
- Advanced usage patterns
