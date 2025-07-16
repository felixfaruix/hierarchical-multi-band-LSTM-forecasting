"""
Quick Setup and Test Script for Optimization Pipeline

This script helps set up and test the optimization pipeline quickly.
It validates the environment, checks dependencies, and runs a quick test.

Usage:
    python setup_optimization.py
    python setup_optimization.py --test
    python setup_optimization.py --install-deps
    python setup_optimization.py --azure-setup

Author: Scientific Pipeline Framework
Version: 1.0
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizationSetup:
    """Setup and validation for optimization pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self.optimization_path = self.src_path / "optimization"
        
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        import sys
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 8:
            logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"❌ Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required dependencies"""
        required_packages = [
            'torch',
            'pytorch_lightning',
            'optuna', 
            'mlflow',
            'pandas',
            'numpy',
            'scikit-learn',
            'plotly',
            'seaborn',
            'matplotlib'
        ]
        
        optional_packages = [
            'azureml',
            'azureml-core',
            'azureml-train-core'
        ]
        
        status = {'required': {}, 'optional': {}}
        
        for package in required_packages:
            try:
                __import__(package)
                status['required'][package] = True
                logger.info(f"✅ {package} - installed")
            except ImportError:
                status['required'][package] = False
                logger.warning(f"❌ {package} - missing")
        
        for package in optional_packages:
            try:
                __import__(package)
                status['optional'][package] = True
                logger.info(f"✅ {package} - installed (optional)")
            except ImportError:
                status['optional'][package] = False
                logger.info(f"⚠️  {package} - missing (optional)")
        
        return status
    
    def install_dependencies(self, include_azure: bool = False) -> bool:
        """Install required dependencies"""
        logger.info("Installing dependencies...")
        
        base_requirements = [
            "torch>=1.12.0",
            "pytorch-lightning>=1.8.0",
            "optuna>=3.0.0",
            "mlflow>=2.0.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
            "matplotlib>=3.3.0"
        ]
        
        azure_requirements = [
            "azureml-core>=1.40.0",
            "azureml-train-core>=1.40.0",
            "azureml-widgets>=1.40.0"
        ]
        
        requirements = base_requirements
        if include_azure:
            requirements.extend(azure_requirements)
        
        try:
            for req in requirements:
                logger.info(f"Installing {req}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            
            logger.info("✅ Dependencies installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def validate_project_structure(self) -> bool:
        """Validate project structure"""
        required_paths = [
            self.src_path,
            self.src_path / "optimization",
            self.src_path / "utils",
            self.src_path / "model.py",
            self.src_path / "train.py"
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(path)
                logger.error(f"❌ Missing: {path}")
            else:
                logger.info(f"✅ Found: {path}")
        
        if missing_paths:
            logger.error(f"❌ Project structure validation failed. Missing {len(missing_paths)} items.")
            return False
        else:
            logger.info("✅ Project structure validation passed!")
            return True
    
    def test_imports(self) -> bool:
        """Test that all optimization modules can be imported"""
        test_modules = [
            "optimization.optuna_optimizer",
            "optimization.azure_ml_optimizer", 
            "optimization.config_templates",
            "utils.data_quality",
            "utils.advanced_quality_assessment"
        ]
        
        failed_imports = []
        
        for module in test_modules:
            try:
                __import__(module)
                logger.info(f"✅ Import test passed: {module}")
            except ImportError as e:
                failed_imports.append((module, str(e)))
                logger.error(f"❌ Import test failed: {module} - {e}")
        
        if failed_imports:
            logger.error(f"❌ Import validation failed for {len(failed_imports)} modules")
            return False
        else:
            logger.info("✅ All import tests passed!")
            return True
    
    def create_sample_data(self) -> str:
        """Create sample data for testing"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample time series data
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        np.random.seed(42)
        n_samples = len(dates)
        
        # Create synthetic ethanol price data with trends and seasonality
        trend = np.linspace(2.0, 2.5, n_samples)
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        noise = 0.1 * np.random.randn(n_samples)
        price = trend + seasonal + noise
        
        # Create volume data
        volume = 1000 + 200 * np.random.randn(n_samples)
        volume = np.maximum(volume, 100)  # Ensure positive
        
        # Create external factors
        wti_price = 50 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 180) + 5 * np.random.randn(n_samples)
        corn_price = 4 + 1 * np.sin(2 * np.pi * np.arange(n_samples) / 120) + 0.5 * np.random.randn(n_samples)
        usd_brl = 5.0 + 0.5 * np.random.randn(n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'ethanol_price': price,
            'volume': volume,
            'wti_price': wti_price,
            'corn_price': corn_price,
            'usd_brl': usd_brl
        })
        
        df.set_index('date', inplace=True)
        
        # Save to temporary directory
        temp_dir = tempfile.mkdtemp()
        sample_data_path = Path(temp_dir) / "sample_ethanol_data.csv"
        df.to_csv(sample_data_path)
        
        logger.info(f"✅ Sample data created: {sample_data_path}")
        logger.info(f"📊 Data shape: {df.shape}")
        logger.info(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        return str(sample_data_path)
    
    def run_quick_test(self) -> bool:
        """Run a quick optimization test"""
        logger.info("Running quick optimization test...")
        
        try:
            # Create sample data
            sample_data_path = self.create_sample_data()
            
            # Import and test optimizer
            from optimization.optuna_optimizer import EthanolOptimizer, OptimizationConfig
            from optimization.config_templates import OptimizationConfigManager
            
            # Get quick test config
            config_dict = OptimizationConfigManager.get_config('quick')
            config_dict['n_trials'] = 3  # Very quick test
            config_dict['timeout'] = 60   # 1 minute max
            
            config = OptimizationConfig()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Create test output directory
            test_output_dir = Path(tempfile.mkdtemp()) / "test_optimization"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run optimization
            optimizer = EthanolOptimizer(config, sample_data_path, str(test_output_dir))
            study = optimizer.optimize()
            
            # Validate results
            if len(study.trials) > 0:
                logger.info(f"✅ Quick test passed!")
                logger.info(f"📊 Completed {len(study.trials)} trials")
                logger.info(f"🎯 Best value: {study.best_value:.4f}")
                logger.info(f"📁 Test results in: {test_output_dir}")
                return True
            else:
                logger.error("❌ Quick test failed - no trials completed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return False
    
    def setup_azure_ml(self) -> bool:
        """Setup Azure ML workspace connection"""
        logger.info("Setting up Azure ML connection...")
        
        try:
            # Check if Azure ML is available
            from azureml.core import Workspace
            
            # Try to connect to workspace
            config_path = Path.home() / ".azureml" / "config.json"
            
            if config_path.exists():
                ws = Workspace.from_config()
                logger.info(f"✅ Connected to Azure ML workspace: {ws.name}")
                logger.info(f"📍 Location: {ws.location}")
                logger.info(f"📋 Resource group: {ws.resource_group}")
                return True
            else:
                logger.warning("⚠️  No Azure ML config found.")
                logger.info("💡 Run 'az ml folder attach -w <workspace-name> -g <resource-group>' to setup")
                return False
                
        except ImportError:
            logger.error("❌ Azure ML SDK not installed. Run with --install-deps --azure")
            return False
        except Exception as e:
            logger.error(f"❌ Azure ML setup failed: {e}")
            return False
    
    def create_config_examples(self) -> None:
        """Create example configuration files"""
        examples_dir = self.optimization_path / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Create Azure ML example config
        azure_config = {
            "name": "azure_ml_production",
            "optimization": {
                "n_trials": 500,
                "timeout": 14400,
                "cv_folds": 5
            },
            "azure_ml": {
                "use_azure_ml": True,
                "subscription_id": "your-subscription-id-here",
                "resource_group": "your-resource-group-here",
                "workspace_name": "your-workspace-name-here",
                "compute_cluster": "cpu-cluster",
                "instance_count": 8,
                "max_concurrent_trials": 16,
                "experiment_name": "ethanol-optimization"
            }
        }
        
        azure_config_path = examples_dir / "azure_ml_config.json"
        with open(azure_config_path, 'w') as f:
            json.dump(azure_config, f, indent=2)
        
        logger.info(f"✅ Created Azure ML config example: {azure_config_path}")
    
    def print_summary(self, results: Dict) -> None:
        """Print setup summary"""
        print("\n" + "="*80)
        print("🚀 OPTIMIZATION PIPELINE SETUP SUMMARY")
        print("="*80)
        
        # Python version
        if results.get('python_ok'):
            print("✅ Python version: Compatible")
        else:
            print("❌ Python version: Incompatible")
        
        # Dependencies
        deps = results.get('dependencies', {})
        required_ok = all(deps.get('required', {}).values())
        if required_ok:
            print("✅ Required dependencies: All installed")
        else:
            missing = [k for k, v in deps.get('required', {}).items() if not v]
            print(f"❌ Required dependencies: Missing {missing}")
        
        # Project structure
        if results.get('structure_ok'):
            print("✅ Project structure: Valid")
        else:
            print("❌ Project structure: Invalid")
        
        # Imports
        if results.get('imports_ok'):
            print("✅ Module imports: All working")
        else:
            print("❌ Module imports: Some failed")
        
        # Quick test
        if results.get('test_ok'):
            print("✅ Quick test: Passed")
        else:
            print("❌ Quick test: Failed")
        
        # Azure ML
        if results.get('azure_ok'):
            print("✅ Azure ML: Connected")
        elif 'azure_ok' in results:
            print("⚠️  Azure ML: Not configured")
        
        print("\n💡 NEXT STEPS:")
        
        if required_ok and results.get('structure_ok') and results.get('imports_ok'):
            print("🎉 Your optimization pipeline is ready!")
            print("\n📋 Quick start commands:")
            print("  # Run local optimization:")
            print("  python src/optimization/run_optimization.py --mode local --preset quick --data_path ./raw_data")
            print("\n  # Run Azure ML optimization:")
            print("  python src/optimization/run_optimization.py --mode azure --preset production --data_path ./raw_data")
            print("\n  # Monitor Azure ML job:")
            print("  python src/optimization/run_optimization.py --mode monitor --job_name <job-name>")
        else:
            print("🔧 Please fix the issues above before running optimization.")
            
            if not required_ok:
                print("  # Install missing dependencies:")
                print("  python src/optimization/setup_optimization.py --install-deps")
            
            if not results.get('structure_ok'):
                print("  # Check project structure and missing files")
            
            if not results.get('imports_ok'):
                print("  # Fix import issues in the modules")
        
        print("\n📖 For more information, see the optimization documentation.")
        print("="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Setup and test optimization pipeline")
    
    parser.add_argument("--test", action="store_true", help="Run quick optimization test")
    parser.add_argument("--install-deps", action="store_true", help="Install required dependencies")
    parser.add_argument("--azure", action="store_true", help="Include Azure ML dependencies")
    parser.add_argument("--azure-setup", action="store_true", help="Setup Azure ML connection")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = OptimizationSetup()
    results = {}
    
    print("🔍 Validating optimization pipeline setup...")
    
    # Check Python version
    results['python_ok'] = setup.check_python_version()
    
    # Install dependencies if requested
    if args.install_deps:
        results['install_ok'] = setup.install_dependencies(include_azure=args.azure)
    
    # Check dependencies
    results['dependencies'] = setup.check_dependencies()
    
    # Validate project structure
    results['structure_ok'] = setup.validate_project_structure()
    
    # Test imports
    results['imports_ok'] = setup.test_imports()
    
    # Create example configs
    setup.create_config_examples()
    
    # Run quick test if requested
    if args.test:
        results['test_ok'] = setup.run_quick_test()
    
    # Setup Azure ML if requested
    if args.azure_setup:
        results['azure_ok'] = setup.setup_azure_ml()
    
    # Print summary
    setup.print_summary(results)


if __name__ == "__main__":
    main()
