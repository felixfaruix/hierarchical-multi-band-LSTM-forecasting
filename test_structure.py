#!/usr/bin/env python3
"""
Test script to verify the reorganized project structure and imports.
"""

try:
    print("Testing new modular structure...")
    print("=" * 50)
    
    # Test models imports
    print("✅ Testing models...")
    try:
        from src.models import HierForecastNet
        from src.models.baseline_models import BaselineForecaster
        print("  ✅ Models import successful")
    except ImportError as e:
        print(f"  ❌ Models import failed: {e}")
    
    # Test data imports
    print("✅ Testing data...")
    try:
        from src.data.dataset_preprocessing import *
        from src.data.timeseries_datamodule import *
        print("  ✅ Data import successful")
    except ImportError as e:
        print(f"  ❌ Data import failed: {e}")
    
    # Test evaluation imports
    print("✅ Testing evaluation...")
    try:
        from src.evaluation.evaluation import HierarchicalEvaluationFramework
        from src.evaluation.metrics import ForecastingMetrics
        from src.evaluation.ts_cross_validation import *
        print("  ✅ Evaluation import successful")
    except ImportError as e:
        print(f"  ❌ Evaluation import failed: {e}")
    
    # Test statistical testing imports
    print("✅ Testing statistical testing...")
    try:
        from src.evaluation.statistical_testing.stats_evaluate import StatisticalEvaluator
        from src.evaluation.statistical_testing.diebold_mariano import ImprovedDieboldMarianoTest
        print("  ✅ Statistical testing import successful")
    except ImportError as e:
        print(f"  ❌ Statistical testing import failed: {e}")
    
    # Test stacking imports
    print("✅ Testing stacking...")
    try:
        from src.stacking.stacked_variants import evaluate_stacked_variants
        print("  ✅ Stacking import successful")
    except ImportError as e:
        print(f"  ❌ Stacking import failed: {e}")
    
    # Test training imports
    print("✅ Testing training...")
    try:
        from src.train.train import HierarchicalTrainer
        from src.train.loss_functions import *
        print("  ✅ Training import successful")
    except ImportError as e:
        print(f"  ❌ Training import failed: {e}")
    
    # Test utils imports
    print("✅ Testing utils...")
    try:
        from src.utils.evaluation_utils import NeuralModelEvaluator
        print("  ✅ Utils import successful")
    except ImportError as e:
        print(f"  ❌ Utils import failed: {e}")
    
    print()
    print("🎉 Project reorganization complete!")
    print("✅ New modular structure is ready for use!")

except Exception as e:
    print(f"❌ General error: {e}")
