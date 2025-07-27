"""
MLflow model registry for Market Master.
"""

import mlflow
import mlflow.sklearn
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow model registry for Market Master."""
    
    def __init__(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            tracking_uri: MLflow tracking URI
            registry_uri: MLflow registry URI
        """
        # Set MLflow tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv('MLFLOW_TRACKING_URI'):
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Set registry URI
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        elif os.getenv('MLFLOW_REGISTRY_URI'):
            mlflow.set_registry_uri(os.getenv('MLFLOW_REGISTRY_URI'))
        
        # Registry configuration
        self.experiment_name = "market_master"
        self.model_name = "action_predictor"
        
        # Ensure experiment exists
        mlflow.set_experiment(self.experiment_name)
        
        logger.info("Model registry initialized", 
                   tracking_uri=mlflow.get_tracking_uri(),
                   registry_uri=mlflow.get_registry_uri(),
                   experiment_name=self.experiment_name)
    
    def register_model(self, model, model_name: str, metrics: Dict[str, float], 
                      parameters: Dict[str, Any] = None, tags: Dict[str, str] = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Model performance metrics
            parameters: Model parameters
            tags: Additional tags
            
        Returns:
            Model version URI
        """
        logger.info(f"Registering model: {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            if parameters:
                mlflow.log_params(parameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id
            
            # Register model
            model_uri = f"runs:/{run_id}/{model_name}"
            model_version = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Model registered successfully", 
                       model_name=model_name,
                       version=model_version.version,
                       run_id=run_id)
            
            return model_uri
    
    def load_model(self, model_name: str, version: Optional[int] = None, stage: str = "Production") -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Model version (if None, loads latest)
            stage: Model stage (Production, Staging, Archived)
            
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            model = mlflow.sklearn.load_model(model_uri)
            
            logger.info(f"Model loaded successfully", 
                       model_name=model_name,
                       version=version,
                       stage=stage)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information
        """
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.list_registered_models()
            
            model_info = []
            for model in models:
                model_info.append({
                    'name': model.name,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'description': model.description,
                    'tags': model.tags
                })
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List versions of a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model versions
        """
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            
            version_info = []
            for version in versions:
                version_info.append({
                    'version': version.version,
                    'run_id': version.run_id,
                    'status': version.status,
                    'stage': version.current_stage,
                    'creation_timestamp': version.creation_timestamp,
                    'last_updated_timestamp': version.last_updated_timestamp
                })
            
            return version_info
            
        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []
    
    def transition_model_stage(self, model_name: str, version: int, stage: str) -> bool:
        """
        Transition a model to a different stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage
            
        Returns:
            True if successful
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(model_name, version, stage)
            
            logger.info(f"Model stage transitioned", 
                       model_name=model_name,
                       version=version,
                       stage=stage)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def delete_model_version(self, model_name: str, version: int) -> bool:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            True if successful
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.delete_model_version(model_name, version)
            
            logger.info(f"Model version deleted", 
                       model_name=model_name,
                       version=version)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False
    
    def get_model_metadata(self, model_name: str, version: int) -> Dict[str, Any]:
        """
        Get metadata for a specific model version.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Model metadata
        """
        try:
            client = mlflow.tracking.MlflowClient()
            version_info = client.get_model_version(model_name, version)
            
            # Get run info
            run = client.get_run(version_info.run_id)
            
            metadata = {
                'model_name': model_name,
                'version': version,
                'run_id': version_info.run_id,
                'status': version_info.status,
                'stage': version_info.current_stage,
                'creation_timestamp': version_info.creation_timestamp,
                'last_updated_timestamp': version_info.last_updated_timestamp,
                'parameters': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            return {}
    
    def compare_models(self, model_name: str, version1: int, version2: int) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        try:
            metadata1 = self.get_model_metadata(model_name, version1)
            metadata2 = self.get_model_metadata(model_name, version2)
            
            comparison = {
                'model_name': model_name,
                'version1': {
                    'version': version1,
                    'metrics': metadata1.get('metrics', {}),
                    'parameters': metadata1.get('parameters', {})
                },
                'version2': {
                    'version': version2,
                    'metrics': metadata2.get('metrics', {}),
                    'parameters': metadata2.get('parameters', {})
                },
                'metrics_diff': {},
                'parameters_diff': {}
            }
            
            # Compare metrics
            metrics1 = metadata1.get('metrics', {})
            metrics2 = metadata2.get('metrics', {})
            
            for metric in set(metrics1.keys()) | set(metrics2.keys()):
                val1 = metrics1.get(metric, 0)
                val2 = metrics2.get(metric, 0)
                comparison['metrics_diff'][metric] = val2 - val1
            
            # Compare parameters
            params1 = metadata1.get('parameters', {})
            params2 = metadata2.get('parameters', {})
            
            for param in set(params1.keys()) | set(params2.keys()):
                val1 = params1.get(param, '')
                val2 = params2.get(param, '')
                comparison['parameters_diff'][param] = {
                    'version1': val1,
                    'version2': val2,
                    'changed': val1 != val2
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}


# Convenience functions

def register_model(model, model_name: str, metrics: Dict[str, float], 
                  parameters: Dict[str, Any] = None, tags: Dict[str, str] = None) -> str:
    """
    Register a model using the global registry.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Model performance metrics
        parameters: Model parameters
        tags: Additional tags
        
    Returns:
        Model version URI
    """
    registry = ModelRegistry()
    return registry.register_model(model, model_name, metrics, parameters, tags)


def load_model(model_name: str, version: Optional[int] = None, stage: str = "Production") -> Any:
    """
    Load a model using the global registry.
    
    Args:
        model_name: Name of the model
        version: Model version (if None, loads latest)
        stage: Model stage
        
    Returns:
        Loaded model
    """
    registry = ModelRegistry()
    return registry.load_model(model_name, version, stage) 