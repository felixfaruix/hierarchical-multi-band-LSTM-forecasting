"""
Centralized Import Configuration for Scientific Pipeline Notebook

This module contains all necessary imports organized by category for the 
comprehensive ethanol forecasting pipeline notebook.

Usage in notebook:
    from src.utils.notebook_imports import *
"""

# ==============================================================================
# CORE PYTHON & SYSTEM
# ==============================================================================
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import datetime
import json
import pickle

# Suppress warnings for cleaner notebook output
warnings.filterwarnings('ignore')
import logging
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)


# ==============================================================================
# SCIENTIFIC COMPUTING
# ==============================================================================
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest


# ==============================================================================
# STATISTICAL ANALYSIS & TIME SERIES
# ==============================================================================
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Optional statistical packages
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    pm = None
    PMDARIMA_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    arch_model = None
    ARCH_AVAILABLE = False


# ==============================================================================
# DEEP LEARNING & PYTORCH
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    pl = None
    Trainer = None
    EarlyStopping = None
    ModelCheckpoint = None
    TensorBoardLogger = None
    WandbLogger = None
    PYTORCH_LIGHTNING_AVAILABLE = False


# ==============================================================================
# HYPERPARAMETER OPTIMIZATION
# ==============================================================================
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    PyTorchLightningPruningCallback = None
    OPTUNA_AVAILABLE = False


# ==============================================================================
# EXPERIMENT TRACKING
# ==============================================================================
try:
    import wandb
    from wandb.integration.pytorch import watch
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    watch = None
    WANDB_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


# ==============================================================================
# AZURE ML & CLOUD SERVICES
# ==============================================================================
try:
    from azureml.core import Workspace, Experiment, Run, Dataset
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.train.pytorch import PyTorch
    from azureml.widgets import RunDetails
    AZUREML_AVAILABLE = True
except ImportError:
    Workspace = None
    Experiment = None
    Run = None
    Dataset = None
    ComputeTarget = None
    AmlCompute = None
    PyTorch = None
    RunDetails = None
    AZUREML_AVAILABLE = False


# ==============================================================================
# VISUALIZATION & PLOTTING
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configure plotting styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# ==============================================================================
# ADDITIONAL ML LIBRARIES
# ==============================================================================
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    from hierarchicalforecast import HierarchicalForecast
    from hierarchicalforecast.methods import MinTrace, ERM
    HIERARCHICALFORECAST_AVAILABLE = True
except ImportError:
    HierarchicalForecast = None
    MinTrace = None
    ERM = None
    HIERARCHICALFORECAST_AVAILABLE = False


# ==============================================================================
# NOTEBOOK SPECIFIC UTILITIES
# ==============================================================================
from IPython.display import display, HTML, Markdown, Image
from IPython.core.magic import register_line_magic
import ipywidgets as widgets
from tqdm.auto import tqdm


# ==============================================================================
# PROJECT SPECIFIC IMPORTS
# ==============================================================================
# Note: These will be imported dynamically in the notebook to handle path issues

# Data quality and preprocessing
# from src.utils.data_quality import (
#     DataQualityAssessor, 
#     DataQualityMetrics,
#     FinancialDataPreprocessor,
#     load_and_assess_data,
#     create_mock_data_for_demo,
#     plot_quality_radar,
#     generate_quality_report
# )

# Model architectures
# from src.models.model import HierForecastNet
# from src.models.baseline_models import create_baseline_models_by_scale

# Evaluation framework
# from src.evaluation.evaluation import run_comprehensive_evaluation
# from src.evaluation.metrics import compute_comprehensive_metrics
# from src.evaluation.ts_cross_validation import ImprovedTimeSeriesCrossValidator

# Statistical testing
# from src.evaluation.statistical_testing.stats_evaluate import StatisticalEvaluator
# from src.evaluation.statistical_testing.diebold_mariano import ImprovedDieboldMarianoTest

# Training and utilities
# from src.train.train import train_model, bootstrap_fifos
# from src.utils.evaluation_utils import NeuralModelEvaluator


# ==============================================================================
# AVAILABILITY FLAGS FOR CONDITIONAL EXECUTION
# ==============================================================================
PACKAGE_AVAILABILITY = {
    'pytorch_lightning': PYTORCH_LIGHTNING_AVAILABLE,
    'optuna': OPTUNA_AVAILABLE,
    'wandb': WANDB_AVAILABLE,
    'mlflow': MLFLOW_AVAILABLE,
    'azureml': AZUREML_AVAILABLE,
    'pmdarima': PMDARIMA_AVAILABLE,
    'arch': ARCH_AVAILABLE,
    'lightgbm': LIGHTGBM_AVAILABLE,
    'xgboost': XGBOOST_AVAILABLE,
    'hierarchicalforecast': HIERARCHICALFORECAST_AVAILABLE
}


# ==============================================================================
# UTILITY FUNCTIONS FOR NOTEBOOK
# ==============================================================================
def print_package_status():
    """Print the availability status of optional packages"""
    print("📦 PACKAGE AVAILABILITY STATUS")
    print("=" * 50)
    
    for package, available in PACKAGE_AVAILABILITY.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{package:20s}: {status}")
    
    print("=" * 50)


def setup_notebook_environment():
    """Setup notebook environment with proper configurations"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Configure pandas display options
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)
    
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print("🔧 Notebook environment configured successfully!")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"🔢 NumPy version: {np.__version__}")
    print(f"🐼 Pandas version: {pd.__version__}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    
    # Print GPU availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU (CUDA not available)")


def add_project_path():
    """Add project paths to sys.path for module imports"""
    project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
    src_path = project_root / 'src'
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source path: {src_path}")
    
    return project_root, src_path


# ==============================================================================
# NOTEBOOK MAGIC FUNCTIONS
# ==============================================================================
def create_section_divider(title: str, emoji: str = "📊") -> None:
    """Create a visually appealing section divider"""
    width = 80
    title_with_emoji = f"{emoji} {title}"
    
    print("\n" + "=" * width)
    print(f"{title_with_emoji:^{width}}")
    print("=" * width + "\n")


def show_progress(description: str):
    """Show progress indicator for long-running operations"""
    return tqdm(desc=description, unit="step")


# ==============================================================================
# INITIALIZATION MESSAGE
# ==============================================================================
def print_initialization_message():
    """Print initialization message with package status"""
    print("🚀 SCIENTIFIC PIPELINE NOTEBOOK IMPORTS LOADED")
    print("=" * 60)
    print("📊 Hierarchical Multi-Band LSTM for Ethanol Price Forecasting")
    print("🔬 Advanced Statistical Analysis & Deep Learning Pipeline")
    print("=" * 60)
    
    # Print available optional packages
    available_packages = [pkg for pkg, avail in PACKAGE_AVAILABILITY.items() if avail]
    unavailable_packages = [pkg for pkg, avail in PACKAGE_AVAILABILITY.items() if not avail]
    
    if available_packages:
        print(f"✅ Available packages: {', '.join(available_packages)}")
    
    if unavailable_packages:
        print(f"⚠️  Unavailable packages: {', '.join(unavailable_packages)}")
        print("💡 Install missing packages for full functionality")
    
    print("=" * 60)


# Print initialization message when module is imported
print_initialization_message()
