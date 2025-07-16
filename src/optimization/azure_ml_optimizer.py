"""
Azure ML Integration for Optuna Hyperparameter Optimization

This module provides Azure ML specific integration for distributed hyperparameter optimization:
- AzureML compute cluster integration
- Distributed Optuna optimization across multiple nodes
- Azure ML logging and experiment tracking
- Model registration and deployment preparation
- Integration with Azure ML pipelines

Author: Scientific Pipeline Framework  
Version: 1.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import tempfile
import shutil

# Azure ML imports
try:
    from azure.ai.ml import MLClient, Input, Output
    from azure.ai.ml.entities import (
        Job, Environment, Command, Model, Data, 
        ComputeInstance, AmlCompute, ManagedIdentity
    )
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential
    import mlflow
    import mlflow.pytorch
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    logging.warning("Azure ML SDK not available. Install with: pip install azure-ai-ml")

# Optuna with storage backend
import optuna
from optuna.storages import RDBStorage
from optuna.integration.mlflow import MLflowCallback

# Project imports
from .optuna_optimizer import EthanolOptimizer, OptimizationConfig, HyperparameterSpace
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureMLOptimizationConfig(OptimizationConfig):
    """Extended configuration for Azure ML optimization"""
    
    def __init__(self):
        super().__init__()
        
        # Azure ML specific settings
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_ML_WORKSPACE")
        self.compute_cluster = os.getenv("AZURE_ML_COMPUTE_CLUSTER", "ethanol-optimization-cluster")
        
        # Distributed optimization settings
        self.n_parallel_jobs = 4  # Number of parallel optimization jobs
        self.storage_backend = "sqlite"  # Can be "sqlite", "mysql", "postgresql"
        self.storage_url = None  # Will be set based on backend
        
        # MLflow tracking
        self.use_mlflow = True
        self.mlflow_experiment_name = "ethanol_hierarchical_optimization"
        
        # Model registration
        self.register_best_model = True
        self.model_name = "ethanol_hierarchical_forecaster"
        self.model_version = None  # Auto-increment
        
        # Data management
        self.data_asset_name = "ethanol_training_data"
        self.data_version = "latest"
        
        # Environment settings
        self.environment_name = "ethanol-forecasting-env"
        self.environment_version = "latest"
        
        # Job settings
        self.job_timeout_minutes = 480  # 8 hours
        self.instance_type = "Standard_DS4_v2"  # Can be upgraded for GPU
        self.instance_count = 4  # Number of compute instances


class AzureMLOptimizer:
    """Azure ML integrated hyperparameter optimizer"""
    
    def __init__(self, config: AzureMLOptimizationConfig):
        if not AZURE_ML_AVAILABLE:
            raise ImportError("Azure ML SDK is required for AzureMLOptimizer")
        
        self.config = config
        self.ml_client = None
        self.workspace = None
        self._initialize_azure_ml()
    
    def _initialize_azure_ml(self):
        """Initialize Azure ML client and workspace"""
        try:
            # Initialize ML client
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=self.config.subscription_id,
                resource_group_name=self.config.resource_group,
                workspace_name=self.config.workspace_name
            )
            
            logger.info(f"Connected to Azure ML workspace: {self.config.workspace_name}")
            
            # Initialize MLflow
            if self.config.use_mlflow:
                mlflow.set_tracking_uri(self.ml_client.workspaces.get().mlflow_tracking_uri)
                mlflow.set_experiment(self.config.mlflow_experiment_name)
                logger.info(f"MLflow tracking initialized for experiment: {self.config.mlflow_experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML: {e}")
            raise
    
    def setup_environment(self) -> Environment:
        """Setup or get the Azure ML environment for optimization"""
        try:
            # Try to get existing environment
            environment = self.ml_client.environments.get(
                name=self.config.environment_name,
                version=self.config.environment_version
            )
            logger.info(f"Using existing environment: {self.config.environment_name}")
            
        except Exception:
            # Create new environment
            logger.info(f"Creating new environment: {self.config.environment_name}")
            
            # Environment definition
            environment = Environment(
                name=self.config.environment_name,
                description="Ethanol forecasting optimization environment with Optuna and PyTorch",
                conda_file="environment.yml",  # Will be created
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            )
            
            # Create conda environment file
            conda_config = {
                "name": "ethanol-forecasting",
                "channels": ["conda-forge", "pytorch", "nvidia"],
                "dependencies": [
                    "python=3.9",
                    "pytorch",
                    "torchvision", 
                    "torchaudio",
                    "pytorch-lightning",
                    "optuna",
                    "pandas",
                    "numpy",
                    "scikit-learn",
                    "plotly",
                    "seaborn",
                    "matplotlib",
                    "statsmodels",
                    "scipy",
                    {
                        "pip": [
                            "azure-ai-ml",
                            "mlflow",
                            "wandb",
                            "optuna[integration]",
                            "pytorch-lightning[extra]"
                        ]
                    }
                ]
            }
            
            # Save conda file temporarily
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                import yaml
                yaml.dump(conda_config, f)
                environment.conda_file = f.name
            
            # Create environment
            environment = self.ml_client.environments.create_or_update(environment)
            logger.info(f"Created environment: {environment.name}:{environment.version}")
        
        return environment
    
    def setup_compute(self) -> AmlCompute:
        """Setup or get compute cluster for optimization"""
        try:
            # Try to get existing compute
            compute = self.ml_client.compute.get(self.config.compute_cluster)
            logger.info(f"Using existing compute cluster: {self.config.compute_cluster}")
            
        except Exception:
            # Create new compute cluster
            logger.info(f"Creating new compute cluster: {self.config.compute_cluster}")
            
            compute = AmlCompute(
                name=self.config.compute_cluster,
                type="amlcompute",
                size=self.config.instance_type,
                min_instances=0,
                max_instances=self.config.instance_count,
                idle_time_before_scale_down=300,  # 5 minutes
                tier="Dedicated",
                description="Compute cluster for ethanol forecasting optimization"
            )
            
            compute = self.ml_client.compute.begin_create_or_update(compute).result()
            logger.info(f"Created compute cluster: {compute.name}")
        
        return compute
    
    def setup_data_asset(self, data_path: str) -> Data:
        """Setup data asset in Azure ML"""
        try:
            # Try to get existing data asset
            data_asset = self.ml_client.data.get(
                name=self.config.data_asset_name,
                version=self.config.data_version
            )
            logger.info(f"Using existing data asset: {self.config.data_asset_name}")
            
        except Exception:
            # Create new data asset
            logger.info(f"Creating new data asset: {self.config.data_asset_name}")
            
            data_asset = Data(
                path=data_path,
                type=AssetTypes.URI_FOLDER,
                name=self.config.data_asset_name,
                description="Ethanol forecasting training data"
            )
            
            data_asset = self.ml_client.data.create_or_update(data_asset)
            logger.info(f"Created data asset: {data_asset.name}:{data_asset.version}")
        
        return data_asset
    
    def create_optimization_script(self, output_dir: str) -> str:
        """Create the optimization script for Azure ML job"""
        script_content = f'''
import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project path
sys.path.append("/tmp/src")

# Import optimization modules
from optimization.optuna_optimizer import EthanolOptimizer, OptimizationConfig
import optuna
from optuna.storages import RDBStorage
import mlflow

def main():
    """Main optimization function for Azure ML"""
    
    # Get job arguments
    data_path = os.environ.get("AZUREML_INPUT_data", "/tmp/data")
    output_path = os.environ.get("AZUREML_OUTPUT_results", "/tmp/outputs")
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    logger.info(f"Starting optimization - Node {{node_rank}}/{{world_size}}")
    logger.info(f"Data path: {{data_path}}")
    logger.info(f"Output path: {{output_path}}")
    
    # Setup configuration
    config = OptimizationConfig()
    config.n_trials = {self.config.n_trials} // world_size  # Distribute trials across nodes
    config.experiment_name = "{self.config.mlflow_experiment_name}"
    
    # Setup distributed storage for Optuna
    storage_url = "sqlite:///optuna_study.db"
    if world_size > 1:
        # Use shared storage for distributed optimization
        storage_url = os.environ.get("OPTUNA_STORAGE_URL", storage_url)
    
    # Create study with shared storage
    study = optuna.create_study(
        study_name=f"ethanol_optimization_{{node_rank}}",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize"
    )
    
    # Setup MLflow
    mlflow.set_experiment(config.experiment_name)
    
    # Create optimizer
    optimizer = EthanolOptimizer(config, data_path, output_path)
    
    # Run optimization
    with mlflow.start_run(run_name=f"optimization_node_{{node_rank}}"):
        study = optimizer.optimize()
        
        # Log results to MLflow
        if config.optimize_multiple_metrics:
            for i, trial in enumerate(study.best_trials):
                mlflow.log_metrics({{
                    f"pareto_{{i}}_{{config.primary_metric}}": trial.values[0],
                    f"pareto_{{i}}_{{config.secondary_metric}}": trial.values[1]
                }})
        else:
            mlflow.log_metrics({{
                "best_score": study.best_value,
                "n_trials": len(study.trials)
            }})
            mlflow.log_params(study.best_params)
    
    # Save node-specific results
    results = {{
        "node_rank": node_rank,
        "n_trials": len(study.trials),
        "best_value": study.best_value if not config.optimize_multiple_metrics else None,
        "best_params": study.best_params if not config.optimize_multiple_metrics else None
    }}
    
    with open(Path(output_path) / f"node_{{node_rank}}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Node {{node_rank}} optimization completed")

if __name__ == "__main__":
    main()
'''
        
        # Save script
        script_path = Path(output_dir) / "azure_optimization_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def run_distributed_optimization(self, data_path: str, output_dir: str) -> Job:
        """Run distributed hyperparameter optimization on Azure ML"""
        
        # Setup prerequisites
        environment = self.setup_environment()
        compute = self.setup_compute()
        data_asset = self.setup_data_asset(data_path)
        
        # Create optimization script
        script_path = self.create_optimization_script(output_dir)
        
        # Setup job
        job = Command(
            command=f"python azure_optimization_script.py",
            code="./src",  # Upload entire src directory
            environment=f"{environment.name}:{environment.version}",
            compute=self.config.compute_cluster,
            distribution={
                "type": "PyTorch",
                "process_count_per_instance": 1
            },
            instance_count=self.config.instance_count,
            inputs={
                "data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml://data/{data_asset.name}/versions/{data_asset.version}"
                )
            },
            outputs={
                "results": Output(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml://datastores/workspaceblobstore/paths/optimization_results/"
                )
            },
            display_name="Ethanol Hierarchical Forecasting Optimization",
            description="Distributed hyperparameter optimization for hierarchical multi-band ethanol forecasting",
            experiment_name=self.config.mlflow_experiment_name,
            tags={
                "project": "ethanol-forecasting",
                "type": "hyperparameter-optimization",
                "framework": "optuna",
                "model": "hierarchical-lstm"
            }
        )
        
        # Submit job
        logger.info("Submitting optimization job to Azure ML...")
        submitted_job = self.ml_client.jobs.create_or_update(job)
        
        logger.info(f"Job submitted successfully!")
        logger.info(f"Job name: {submitted_job.name}")
        logger.info(f"Job URL: {submitted_job.studio_url}")
        
        return submitted_job
    
    def monitor_job(self, job_name: str) -> Dict[str, Any]:
        """Monitor optimization job progress"""
        job = self.ml_client.jobs.get(job_name)
        
        return {
            "status": job.status,
            "start_time": job.creation_context.created_at,
            "compute": job.compute,
            "experiment": job.experiment_name,
            "studio_url": job.studio_url
        }
    
    def collect_results(self, job_name: str, local_output_dir: str) -> Dict[str, Any]:
        """Collect optimization results from completed job"""
        job = self.ml_client.jobs.get(job_name)
        
        if job.status != "Completed":
            logger.warning(f"Job {job_name} is not completed. Status: {job.status}")
            return {"status": job.status}
        
        # Download outputs
        self.ml_client.jobs.download(
            name=job_name,
            download_path=local_output_dir,
            output_name="results"
        )
        
        # Aggregate results from all nodes
        results_dir = Path(local_output_dir) / "results"
        all_results = []
        
        for result_file in results_dir.glob("node_*_results.json"):
            with open(result_file, 'r') as f:
                node_results = json.load(f)
                all_results.append(node_results)
        
        # Combine results
        combined_results = {
            "total_trials": sum(r["n_trials"] for r in all_results),
            "node_results": all_results,
            "job_name": job_name,
            "job_status": job.status
        }
        
        # Save combined results
        with open(Path(local_output_dir) / "combined_results.json", 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        logger.info(f"Results collected and saved to {local_output_dir}")
        return combined_results
    
    def register_best_model(self, job_name: str, model_path: str) -> Model:
        """Register the best model from optimization"""
        if not self.config.register_best_model:
            logger.info("Model registration disabled")
            return None
        
        # Create model asset
        model = Model(
            path=model_path,
            name=self.config.model_name,
            description="Best hierarchical multi-band ethanol forecasting model from optimization",
            tags={
                "optimization_job": job_name,
                "framework": "pytorch",
                "type": "hierarchical-forecasting",
                "commodity": "ethanol"
            }
        )
        
        # Register model
        registered_model = self.ml_client.models.create_or_update(model)
        
        logger.info(f"Model registered: {registered_model.name}:{registered_model.version}")
        return registered_model


def create_azure_optimization_pipeline():
    """Create an Azure ML pipeline for optimization"""
    
    # This would create a more complex pipeline with multiple steps
    # For now, we'll use the single job approach above
    pass


def main():
    """Main function for Azure ML optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure ML Ethanol Forecasting Optimization")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, help="Azure resource group")
    parser.add_argument("--workspace_name", type=str, help="Azure ML workspace name")
    parser.add_argument("--compute_cluster", type=str, help="Compute cluster name")
    parser.add_argument("--n_trials", type=int, default=100, help="Total number of trials")
    parser.add_argument("--instance_count", type=int, default=4, help="Number of compute instances")
    parser.add_argument("--monitor_only", type=str, help="Monitor existing job by name")
    parser.add_argument("--collect_results", type=str, help="Collect results from job by name")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = AzureMLOptimizationConfig()
    if args.subscription_id:
        config.subscription_id = args.subscription_id
    if args.resource_group:
        config.resource_group = args.resource_group
    if args.workspace_name:
        config.workspace_name = args.workspace_name
    if args.compute_cluster:
        config.compute_cluster = args.compute_cluster
    
    config.n_trials = args.n_trials
    config.instance_count = args.instance_count
    
    # Create optimizer
    optimizer = AzureMLOptimizer(config)
    
    # Handle different modes
    if args.monitor_only:
        # Monitor existing job
        status = optimizer.monitor_job(args.monitor_only)
        print(f"Job Status: {status}")
        
    elif args.collect_results:
        # Collect results from completed job
        results = optimizer.collect_results(args.collect_results, args.output_dir)
        print(f"Results collected: {results}")
        
    else:
        # Run new optimization
        job = optimizer.run_distributed_optimization(args.data_path, args.output_dir)
        print(f"Optimization job submitted: {job.name}")
        print(f"Monitor at: {job.studio_url}")


if __name__ == "__main__":
    main()
