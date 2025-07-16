"""
Optimization Runner Script

This script provides a unified interface for running hyperparameter optimization
with support for both local and Azure ML execution. It integrates all components
and provides comprehensive logging and monitoring.

Usage:
    python run_optimization.py --mode local --preset quick --data_path ./data
    python run_optimization.py --mode azure --preset production --data_path ./data
    python run_optimization.py --mode monitor --job_name <job_name>

Author: Scientific Pipeline Framework
Version: 1.0
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import subprocess
import time

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

# Project imports
from optimization.optuna_optimizer import EthanolOptimizer, OptimizationConfig
from optimization.azure_ml_optimizer import AzureMLOptimizer, AzureMLOptimizationConfig
from optimization.config_templates import OptimizationConfigManager, validate_config
from utils.data_quality import DataQualityAssessor
from utils.advanced_quality_assessment import AdvancedQualityAssessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationRunner:
    """Main runner for hyperparameter optimization"""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """Initialize runner with configuration"""
        
        if config_path:
            self.config = OptimizationConfigManager.load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            # Use default production config
            self.config = OptimizationConfigManager.get_config('production')
        
        # Validate configuration
        errors = validate_config(self.config)
        if errors:
            raise ValueError(f"Configuration errors: {errors}")
        
        self.use_azure_ml = self.config.get('use_azure_ml', False)
        self.results_dir = Path(self.config.get('output_dir', './optimization_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.quality_assessor = DataQualityAssessor()
        self.advanced_quality_assessor = AdvancedQualityAssessor()
        
        logger.info(f"Optimization runner initialized")
        logger.info(f"Mode: {'Azure ML' if self.use_azure_ml else 'Local'}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def validate_data(self, data_path: str) -> Dict[str, Any]:
        """Validate data quality before optimization"""
        logger.info("Validating data quality...")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Load data files
        data_files = list(data_path.glob("*.csv"))
        if not data_files:
            raise ValueError(f"No CSV files found in {data_path}")
        
        quality_results = {}
        for file_path in data_files:
            try:
                import pandas as pd
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Basic quality assessment
                basic_metrics = self.quality_assessor.comprehensive_assessment(df)
                
                # Advanced quality assessment
                advanced_metrics = self.advanced_quality_assessor.comprehensive_advanced_assessment(df)
                
                quality_results[file_path.stem] = {
                    'basic': basic_metrics.__dict__,
                    'advanced': advanced_metrics.__dict__,
                    'shape': df.shape,
                    'date_range': [str(df.index.min()), str(df.index.max())],
                    'columns': list(df.columns)
                }
                
                logger.info(f"Validated {file_path.stem}: {df.shape} samples, "
                           f"quality score: {basic_metrics.completeness:.3f}")
                
            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")
                quality_results[file_path.stem] = {'error': str(e)}
        
        # Save validation results
        validation_report_path = self.results_dir / "data_validation_report.json"
        with open(validation_report_path, 'w') as f:
            json.dump(quality_results, f, indent=2, default=str)
        
        logger.info(f"Data validation completed. Report saved to {validation_report_path}")
        return quality_results
    
    def run_local_optimization(self, data_path: str) -> Dict[str, Any]:
        """Run optimization locally"""
        logger.info("Starting local optimization...")
        
        # Validate data first
        quality_results = self.validate_data(data_path)
        
        # Create local optimizer
        local_config = OptimizationConfig()
        for key, value in self.config.items():
            if hasattr(local_config, key):
                setattr(local_config, key, value)
        
        optimizer = EthanolOptimizer(local_config, data_path, str(self.results_dir))
        
        # Run optimization
        start_time = datetime.now()
        try:
            study = optimizer.optimize()
            
            # Generate results
            results = {
                'status': 'completed',
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
                'n_trials': len(study.trials),
                'best_value': study.best_value if not local_config.optimize_multiple_metrics else None,
                'best_params': study.best_params if not local_config.optimize_multiple_metrics else None,
                'data_validation': quality_results
            }
            
            if local_config.optimize_multiple_metrics:
                results['pareto_front'] = [
                    {'values': trial.values, 'params': trial.params} 
                    for trial in study.best_trials
                ]
            
            # Save results
            results_path = self.results_dir / "optimization_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Create visualization report
            viz_path = optimizer.create_visualization_report()
            results['visualization_path'] = viz_path
            
            logger.info(f"Local optimization completed successfully!")
            logger.info(f"Best value: {results.get('best_value', 'N/A')}")
            logger.info(f"Total trials: {results['n_trials']}")
            logger.info(f"Duration: {results['duration_minutes']:.1f} minutes")
            
            return results
            
        except Exception as e:
            logger.error(f"Local optimization failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
    
    def run_azure_optimization(self, data_path: str) -> Dict[str, Any]:
        """Run optimization on Azure ML"""
        logger.info("Starting Azure ML optimization...")
        
        try:
            # Validate data first
            quality_results = self.validate_data(data_path)
            
            # Create Azure ML optimizer
            azure_config = AzureMLOptimizationConfig()
            for key, value in self.config.items():
                if hasattr(azure_config, key):
                    setattr(azure_config, key, value)
            
            optimizer = AzureMLOptimizer(azure_config)
            
            # Submit optimization job
            job = optimizer.run_distributed_optimization(data_path, str(self.results_dir))
            
            results = {
                'status': 'submitted',
                'job_name': job.name,
                'job_url': job.studio_url,
                'compute_cluster': azure_config.compute_cluster,
                'instance_count': azure_config.instance_count,
                'submit_time': datetime.now().isoformat(),
                'data_validation': quality_results
            }
            
            # Save job info
            job_info_path = self.results_dir / "azure_job_info.json"
            with open(job_info_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Azure ML optimization job submitted!")
            logger.info(f"Job name: {job.name}")
            logger.info(f"Studio URL: {job.studio_url}")
            logger.info(f"Monitor with: python run_optimization.py --mode monitor --job_name {job.name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Azure ML optimization failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'submit_time': datetime.now().isoformat()
            }
    
    def monitor_azure_job(self, job_name: str) -> Dict[str, Any]:
        """Monitor Azure ML optimization job"""
        logger.info(f"Monitoring Azure ML job: {job_name}")
        
        try:
            azure_config = AzureMLOptimizationConfig()
            optimizer = AzureMLOptimizer(azure_config)
            
            status = optimizer.monitor_job(job_name)
            
            logger.info(f"Job Status: {status['status']}")
            logger.info(f"Compute: {status['compute']}")
            logger.info(f"Experiment: {status['experiment']}")
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to monitor job {job_name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def collect_azure_results(self, job_name: str) -> Dict[str, Any]:
        """Collect results from completed Azure ML job"""
        logger.info(f"Collecting results from Azure ML job: {job_name}")
        
        try:
            azure_config = AzureMLOptimizationConfig()
            optimizer = AzureMLOptimizer(azure_config)
            
            results = optimizer.collect_results(job_name, str(self.results_dir))
            
            logger.info(f"Results collected successfully!")
            logger.info(f"Total trials: {results.get('total_trials', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to collect results from {job_name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def create_summary_report(self) -> str:
        """Create comprehensive summary report"""
        logger.info("Creating summary report...")
        
        # Collect all result files
        result_files = {
            'optimization': self.results_dir / "optimization_results.json",
            'data_validation': self.results_dir / "data_validation_report.json",
            'azure_job': self.results_dir / "azure_job_info.json",
            'combined_results': self.results_dir / "combined_results.json"
        }
        
        report_data = {}
        for result_type, file_path in result_files.items():
            if file_path.exists():
                with open(file_path, 'r') as f:
                    report_data[result_type] = json.load(f)
        
        # Create HTML report
        html_report = self._generate_html_report(report_data)
        
        report_path = self.results_dir / "optimization_summary_report.html"
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        logger.info(f"Summary report created: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, data: Dict) -> str:
        """Generate HTML summary report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ethanol Forecasting Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; }}
                .success {{ color: #27ae60; }}
                .error {{ color: #e74c3c; }}
                .warning {{ color: #f39c12; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Ethanol Forecasting Optimization Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add optimization results section
        if 'optimization' in data:
            opt_data = data['optimization']
            status_class = 'success' if opt_data.get('status') == 'completed' else 'error'
            
            html += f"""
            <div class="section">
                <h2>Optimization Results</h2>
                <div class="metric">
                    <strong>Status:</strong> 
                    <span class="{status_class}">{opt_data.get('status', 'Unknown')}</span>
                </div>
                <div class="metric">
                    <strong>Total Trials:</strong> {opt_data.get('n_trials', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Duration:</strong> {opt_data.get('duration_minutes', 'N/A')} minutes
                </div>
                <div class="metric">
                    <strong>Best Value:</strong> {opt_data.get('best_value', 'N/A')}
                </div>
            </div>
            """
        
        # Add data validation section
        if 'data_validation' in data:
            html += """
            <div class="section">
                <h2>Data Quality Assessment</h2>
                <table>
                    <tr><th>Dataset</th><th>Shape</th><th>Completeness</th><th>Accuracy</th><th>Date Range</th></tr>
            """
            
            for dataset, info in data['data_validation'].items():
                if 'basic' in info:
                    html += f"""
                    <tr>
                        <td>{dataset}</td>
                        <td>{info.get('shape', 'N/A')}</td>
                        <td>{info['basic'].get('completeness', 'N/A'):.3f}</td>
                        <td>{info['basic'].get('accuracy', 'N/A'):.3f}</td>
                        <td>{' to '.join(info.get('date_range', ['N/A', 'N/A']))}</td>
                    </tr>
                    """
            
            html += "</table></div>"
        
        # Add Azure ML job info if available
        if 'azure_job' in data:
            job_data = data['azure_job']
            html += f"""
            <div class="section">
                <h2>Azure ML Job Information</h2>
                <div class="metric">
                    <strong>Job Name:</strong> {job_data.get('job_name', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Compute Cluster:</strong> {job_data.get('compute_cluster', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Instance Count:</strong> {job_data.get('instance_count', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Studio URL:</strong> 
                    <a href="{job_data.get('job_url', '#')}" target="_blank">View in Azure ML Studio</a>
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Ethanol Forecasting Hyperparameter Optimization Runner")
    
    # Mode selection
    parser.add_argument("--mode", choices=['local', 'azure', 'monitor', 'collect', 'report'],
                       required=True, help="Optimization mode")
    
    # Configuration
    parser.add_argument("--preset", choices=['quick', 'research', 'production', 'multi_objective'],
                       default='production', help="Configuration preset")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    
    # Data and output
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./optimization_results",
                       help="Output directory for results")
    
    # Azure ML specific
    parser.add_argument("--job_name", type=str, help="Azure ML job name (for monitor/collect modes)")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, help="Azure resource group")
    parser.add_argument("--workspace_name", type=str, help="Azure ML workspace name")
    
    # Overrides
    parser.add_argument("--n_trials", type=int, help="Override number of trials")
    parser.add_argument("--timeout", type=int, help="Override timeout (seconds)")
    parser.add_argument("--cv_folds", type=int, help="Override CV folds")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = OptimizationConfigManager.load_config(args.config)
    else:
        # Create config from preset
        overrides = {}
        if args.n_trials:
            overrides['n_trials'] = args.n_trials
        if args.timeout:
            overrides['timeout'] = args.timeout
        if args.cv_folds:
            overrides['cv_folds'] = args.cv_folds
        if args.output_dir:
            overrides['output_dir'] = args.output_dir
        
        # Add Azure ML settings if provided
        if args.mode == 'azure':
            azure_overrides = {}
            if args.subscription_id:
                azure_overrides['subscription_id'] = args.subscription_id
            if args.resource_group:
                azure_overrides['resource_group'] = args.resource_group
            if args.workspace_name:
                azure_overrides['workspace_name'] = args.workspace_name
            
            config = OptimizationConfigManager.create_azure_config(args.preset, **azure_overrides)
            config.update(overrides)
        else:
            config = OptimizationConfigManager.get_config(args.preset, **overrides)
    
    # Create runner
    runner = OptimizationRunner(config_dict=config)
    
    # Execute based on mode
    if args.mode == 'local':
        if not args.data_path:
            parser.error("--data_path is required for local mode")
        
        results = runner.run_local_optimization(args.data_path)
        
        if results['status'] == 'completed':
            print("\n🎉 Local optimization completed successfully!")
            print(f"📊 Best value: {results.get('best_value', 'N/A')}")
            print(f"📈 Total trials: {results['n_trials']}")
            print(f"⏱️  Duration: {results['duration_minutes']:.1f} minutes")
            print(f"📁 Results saved to: {args.output_dir}")
        else:
            print(f"\n❌ Optimization failed: {results.get('error', 'Unknown error')}")
    
    elif args.mode == 'azure':
        if not args.data_path:
            parser.error("--data_path is required for azure mode")
        
        results = runner.run_azure_optimization(args.data_path)
        
        if results['status'] == 'submitted':
            print("\n🚀 Azure ML optimization job submitted!")
            print(f"📝 Job name: {results['job_name']}")
            print(f"🌐 Studio URL: {results['job_url']}")
            print(f"💻 Compute: {results['compute_cluster']} ({results['instance_count']} instances)")
            print(f"\n📊 Monitor with: python run_optimization.py --mode monitor --job_name {results['job_name']}")
        else:
            print(f"\n❌ Azure ML submission failed: {results.get('error', 'Unknown error')}")
    
    elif args.mode == 'monitor':
        if not args.job_name:
            parser.error("--job_name is required for monitor mode")
        
        status = runner.monitor_azure_job(args.job_name)
        print(f"\n📊 Job Status: {status.get('status', 'Unknown')}")
        
        if 'studio_url' in status:
            print(f"🌐 Studio URL: {status['studio_url']}")
    
    elif args.mode == 'collect':
        if not args.job_name:
            parser.error("--job_name is required for collect mode")
        
        results = runner.collect_azure_results(args.job_name)
        
        if results.get('status') != 'error':
            print("\n📥 Results collected successfully!")
            print(f"📊 Total trials: {results.get('total_trials', 'N/A')}")
            print(f"📁 Results saved to: {args.output_dir}")
        else:
            print(f"\n❌ Collection failed: {results.get('error', 'Unknown error')}")
    
    elif args.mode == 'report':
        report_path = runner.create_summary_report()
        print(f"\n📋 Summary report created: {report_path}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
