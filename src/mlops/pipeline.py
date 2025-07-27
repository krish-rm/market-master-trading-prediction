"""
MLOps pipeline module for Market Master.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import mlflow
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from prefect.filesystems import LocalFileSystem
from prefect.server.schemas.schedules import CronSchedule
from ..data import generate_training_data, AssetClass
from ..models import ActionPredictor, train_action_predictor, evaluate_model
from .model_registry import ModelRegistry
from .monitoring import ComprehensiveMonitor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MLPipeline:
    """Base class for ML pipelines."""
    
    def __init__(self, name: str):
        """
        Initialize the pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.registry = ModelRegistry()
        self.monitor = None
    
    def run(self) -> Dict[str, Any]:
        """Run the pipeline."""
        raise NotImplementedError


class TrainingPipeline(MLPipeline):
    """Training pipeline for Market Master."""
    
    def __init__(self, asset_class: AssetClass = AssetClass.EQUITY, 
                 instrument: str = "AAPL"):
        """
        Initialize training pipeline.
        
        Args:
            asset_class: Asset class to train on
            instrument: Instrument to train on
        """
        super().__init__(f"training_{asset_class.value}_{instrument}")
        self.asset_class = asset_class
        self.instrument = instrument
    
    def run(self, n_samples: int = 10000, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run training pipeline.
        
        Args:
            n_samples: Number of training samples
            test_size: Test set size ratio
            
        Returns:
            Training results
        """
        logger.info(f"Starting training pipeline for {self.asset_class.value}/{self.instrument}")
        
        # Generate training data
        data = generate_training_data(self.asset_class, self.instrument, n_samples)
        
        # Split data
        train_data, test_data = self._split_data(data, test_size)
        
        # Prepare features and labels
        X_train = train_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        y_train = train_data['action']
        X_test = test_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        y_test = test_data['action']
        
        # Train model
        model, train_metrics = train_action_predictor(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        eval_results = evaluate_model(model, X_test, y_test)
        
        # Register model
        model_uri = self.registry.register_model(
            model.model,
            f"action_predictor_{self.asset_class.value}_{self.instrument}",
            train_metrics,
            parameters=model.model_config,
            tags={
                'asset_class': self.asset_class.value,
                'instrument': self.instrument,
                'training_date': datetime.now().isoformat()
            }
        )
        
        # Save model locally
        model_path = f"models/action_predictor_{self.asset_class.value}_{self.instrument}.joblib"
        model.save_model(model_path)
        
        results = {
            'model_uri': model_uri,
            'model_path': model_path,
            'training_metrics': train_metrics,
            'evaluation_results': eval_results,
            'asset_class': self.asset_class.value,
            'instrument': self.instrument,
            'training_date': datetime.now().isoformat()
        }
        
        logger.info("Training pipeline completed successfully", results=results)
        return results
    
    def _split_data(self, data: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        # Ensure chronological split
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        return train_data, test_data


class InferencePipeline(MLPipeline):
    """Inference pipeline for Market Master."""
    
    def __init__(self, model_name: str, model_version: Optional[int] = None):
        """
        Initialize inference pipeline.
        
        Args:
            model_name: Name of the model to load
            model_version: Model version (if None, loads latest)
        """
        super().__init__(f"inference_{model_name}")
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
    
    def load_model(self):
        """Load the model from registry."""
        try:
            self.model = self.registry.load_model(self.model_name, self.model_version)
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run inference pipeline.
        
        Args:
            data: Input data for inference
            
        Returns:
            Inference results
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Running inference on {len(data)} samples")
        
        # Prepare features
        features = data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        
        # Make predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Add predictions to data
        results_data = data.copy()
        results_data['prediction'] = predictions
        results_data['confidence'] = np.max(probabilities, axis=1)
        
        # Calculate prediction distribution
        prediction_dist = pd.Series(predictions).value_counts().to_dict()
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'results_data': results_data,
            'prediction_distribution': prediction_dist,
            'inference_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version
        }
        
        logger.info("Inference completed", prediction_distribution=prediction_dist)
        return results


# Prefect Tasks

@task(name="generate_training_data", cache_key_fn=task_input_hash)
def generate_training_data_task(asset_class: str, instrument: str, n_samples: int) -> pd.DataFrame:
    """Generate training data task."""
    logger = get_run_logger()
    logger.info(f"Generating training data for {asset_class}/{instrument}")
    
    asset_enum = AssetClass(asset_class)
    data = generate_training_data(asset_enum, instrument, n_samples)
    
    logger.info(f"Generated {len(data)} training samples")
    return data


@task(name="train_model")
def train_model_task(X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[ActionPredictor, Dict[str, float]]:
    """Train model task."""
    logger = get_run_logger()
    logger.info("Training Action Predictor model")
    
    model, metrics = train_action_predictor(X_train, y_train, X_val, y_val)
    
    logger.info("Model training completed", metrics=metrics)
    return model, metrics


@task(name="evaluate_model")
def evaluate_model_task(model: ActionPredictor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate model task."""
    logger = get_run_logger()
    logger.info("Evaluating model performance")
    
    results = evaluate_model(model, X_test, y_test)
    
    logger.info("Model evaluation completed", accuracy=results['metrics']['accuracy'])
    return results


@task(name="register_model")
def register_model_task(model: ActionPredictor, model_name: str, metrics: Dict[str, float]) -> str:
    """Register model task."""
    logger = get_run_logger()
    logger.info(f"Registering model: {model_name}")
    
    registry = ModelRegistry()
    model_uri = registry.register_model(
        model.model,
        model_name,
        metrics,
        parameters=model.model_config
    )
    
    logger.info(f"Model registered successfully: {model_uri}")
    return model_uri


@task(name="monitor_model")
def monitor_model_task(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                      predictions: np.ndarray = None) -> Dict[str, Any]:
    """Monitor model task."""
    logger = get_run_logger()
    logger.info("Running model monitoring")
    
    monitor = ComprehensiveMonitor(reference_data)
    results = monitor.run_monitoring(current_data, predictions)
    
    logger.info("Monitoring completed", status=results['overall_status'])
    return results


# Prefect Flows

@flow(name="Market Master Training Pipeline")
def market_master_training_flow(asset_class: str = "equity", instrument: str = "AAPL", 
                               n_samples: int = 10000) -> Dict[str, Any]:
    """
    Complete training pipeline flow.
    
    Args:
        asset_class: Asset class to train on
        instrument: Instrument to train on
        n_samples: Number of training samples
        
    Returns:
        Training results
    """
    logger = get_run_logger()
    logger.info("Starting Market Master training pipeline")
    
    # Generate training data
    data = generate_training_data_task(asset_class, instrument, n_samples)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:int(len(data) * 0.9)]
    test_data = data.iloc[int(len(data) * 0.9):]
    
    # Prepare features and labels
    X_train = train_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    y_train = train_data['action']
    X_val = val_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    y_val = val_data['action']
    X_test = test_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    y_test = test_data['action']
    
    # Train model
    model, train_metrics = train_model_task(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    eval_results = evaluate_model_task(model, X_test, y_test)
    
    # Register model
    model_name = f"action_predictor_{asset_class}_{instrument}"
    model_uri = register_model_task(model, model_name, train_metrics)
    
    # Run monitoring
    monitor_results = monitor_model_task(train_data, test_data, eval_results['predictions'])
    
    results = {
        'model_uri': model_uri,
        'training_metrics': train_metrics,
        'evaluation_results': eval_results,
        'monitoring_results': monitor_results,
        'asset_class': asset_class,
        'instrument': instrument,
        'training_date': datetime.now().isoformat()
    }
    
    logger.info("Market Master training pipeline completed successfully")
    return results


@flow(name="Market Master Inference Pipeline")
def market_master_inference_flow(model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete inference pipeline flow.
    
    Args:
        model_name: Name of the model to use
        data: Input data for inference
        
    Returns:
        Inference results
    """
    logger = get_run_logger()
    logger.info("Starting Market Master inference pipeline")
    
    # Load model
    registry = ModelRegistry()
    model = registry.load_model(model_name)
    
    # Prepare features
    features = data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    
    # Make predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    
    # Run monitoring
    monitor_results = monitor_model_task(data, data, predictions)
    
    results = {
        'predictions': predictions,
        'probabilities': probabilities,
        'monitoring_results': monitor_results,
        'model_name': model_name,
        'inference_timestamp': datetime.now().isoformat()
    }
    
    logger.info("Market Master inference pipeline completed successfully")
    return results


# Deployment functions

def deploy_training_pipeline():
    """Deploy the training pipeline."""
    # Use the new Prefect API with flow.serve()
    market_master_training_flow.serve(
        name="market-master-training",
        version="1.0.0",
        work_queue_name="mlops",
        schedule=CronSchedule(cron="0 2 * * *")  # Daily at 2 AM
    )
    
    logger.info("Training pipeline deployed successfully")


def deploy_inference_pipeline():
    """Deploy the inference pipeline."""
    # Use the new Prefect API with flow.serve()
    market_master_inference_flow.serve(
        name="market-master-inference",
        version="1.0.0",
        work_queue_name="mlops"
    )
    
    logger.info("Inference pipeline deployed successfully") 