"""
Market Master Pipeline using Prefect for Financial Market Prediction System.

This module defines the complete MLOps workflow including:
- Data ingestion and preprocessing
- Model training and validation
- Model deployment and registry management
- Monitoring and alerting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from prefect.blocks.system import Secret
from prefect.filesystems import LocalFileSystem

from src.data.data_generator import MarketDataGenerator
from src.models.action_predictor import ActionPredictor
from src.mlops.model_registry import ModelRegistry
from src.mlops.monitoring import ModelMonitor
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prefect configuration
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def get_data_generator() -> MarketDataGenerator:
    """Get configured data generator instance."""
    return MarketDataGenerator()

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def get_model_registry() -> ModelRegistry:
    """Get configured model registry instance."""
    return ModelRegistry()

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def get_model_monitor() -> ModelMonitor:
    """Get configured model monitor instance."""
    return ModelMonitor()

@task(retries=3, retry_delay_seconds=60)
def ingest_financial_data(
    asset_class: str = "crypto",
    symbol: str = "BTC/USD",
    num_samples: int = 10000
) -> pd.DataFrame:
    """
    Ingest financial market data for training.
    
    Args:
        asset_class: Type of asset (crypto, forex, equity, commodity)
        symbol: Trading symbol
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with market data
    """
    logger = get_run_logger()
    logger.info(f"Ingesting {num_samples} samples for {symbol}")
    
    data_gen = get_data_generator()
    data = data_gen.generate_training_data(
        asset_class=asset_class,
        symbol=symbol,
        num_samples=num_samples
    )
    
    logger.info(f"Successfully ingested {len(data)} samples")
    return data

@task(retries=2, retry_delay_seconds=30)
def preprocess_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from raw market data.
    
    Args:
        data: Raw market data DataFrame
        
    Returns:
        DataFrame with technical indicators
    """
    logger = get_run_logger()
    logger.info("Calculating technical indicators")
    
    # Calculate technical indicators
    data_gen = get_data_generator()
    processed_data = data_gen.calculate_technical_indicators(data)
    
    logger.info(f"Calculated {len(processed_data.columns)} technical indicators")
    return processed_data

@task(retries=2, retry_delay_seconds=30)
def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        data: DataFrame with technical indicators
        
    Returns:
        Tuple of features and labels
    """
    logger = get_run_logger()
    logger.info("Preparing training data")
    
    # Remove rows with missing values
    data_clean = data.dropna()
    
    # Separate features and labels
    feature_columns = [col for col in data_clean.columns if col not in ['action', 'timestamp']]
    features = data_clean[feature_columns]
    labels = data_clean['action']
    
    logger.info(f"Prepared {len(features)} samples with {len(feature_columns)} features")
    return features, labels

@task(retries=3, retry_delay_seconds=60)
def train_financial_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model_params: Optional[Dict[str, Any]] = None
) -> ActionPredictor:
    """
    Train the financial prediction model.
    
    Args:
        features: Training features
        labels: Training labels
        model_params: Model hyperparameters
        
    Returns:
        Trained ActionPredictor model
    """
    logger = get_run_logger()
    logger.info("Training financial prediction model")
    
    # Default model parameters
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    
    # Initialize and train model
    model = ActionPredictor(**model_params)
    model.fit(features, labels)
    
    # Evaluate model
    accuracy = model.score(features, labels)
    logger.info(f"Model training completed with accuracy: {accuracy:.3f}")
    
    return model

@task(retries=2, retry_delay_seconds=30)
def validate_model_performance(
    model: ActionPredictor,
    features: pd.DataFrame,
    labels: pd.Series
) -> Dict[str, float]:
    """
    Validate model performance and return metrics.
    
    Args:
        model: Trained model
        features: Validation features
        labels: Validation labels
        
    Returns:
        Dictionary of performance metrics
    """
    logger = get_run_logger()
    logger.info("Validating model performance")
    
    # Make predictions
    predictions = model.predict(features)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1_score': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'training_samples': len(features),
        'feature_count': len(features.columns)
    }
    
    logger.info(f"Validation metrics: {metrics}")
    return metrics

@task(retries=2, retry_delay_seconds=30)
def register_model(
    model: ActionPredictor,
    metrics: Dict[str, float],
    model_name: str = "financial_predictor"
) -> str:
    """
    Register model in MLflow registry.
    
    Args:
        model: Trained model
        metrics: Model performance metrics
        model_name: Name for the model
        
    Returns:
        Model version string
    """
    logger = get_run_logger()
    logger.info(f"Registering model: {model_name}")
    
    registry = get_model_registry()
    model_version = registry.register_model(
        model=model,
        model_name=model_name,
        metrics=metrics
    )
    
    logger.info(f"Model registered with version: {model_version}")
    return model_version

@task(retries=2, retry_delay_seconds=30)
def deploy_model(
    model_version: str,
    model_name: str = "financial_predictor"
) -> bool:
    """
    Deploy model to production.
    
    Args:
        model_version: Model version to deploy
        model_name: Name of the model
        
    Returns:
        True if deployment successful
    """
    logger = get_run_logger()
    logger.info(f"Deploying model version: {model_version}")
    
    registry = get_model_registry()
    success = registry.transition_model(
        model_name=model_name,
        version=model_version,
        stage="Production"
    )
    
    if success:
        logger.info(f"Model {model_version} deployed successfully")
    else:
        logger.error(f"Failed to deploy model {model_version}")
    
    return success

@task(retries=2, retry_delay_seconds=30)
def monitor_model_performance(
    model_name: str = "financial_predictor",
    window_size: int = 100
) -> Dict[str, Any]:
    """
    Monitor model performance in production.
    
    Args:
        model_name: Name of the model to monitor
        window_size: Number of recent predictions to analyze
        
    Returns:
        Dictionary with monitoring results
    """
    logger = get_run_logger()
    logger.info(f"Monitoring model: {model_name}")
    
    monitor = get_model_monitor()
    monitoring_results = monitor.monitor_model_performance(
        model_name=model_name,
        window_size=window_size
    )
    
    logger.info(f"Monitoring results: {monitoring_results}")
    return monitoring_results

@task(retries=2, retry_delay_seconds=30)
def check_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Check for data drift between reference and current data.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        
    Returns:
        Dictionary with drift detection results
    """
    logger = get_run_logger()
    logger.info("Checking for data drift")
    
    monitor = get_model_monitor()
    drift_results = monitor.detect_data_drift(
        reference_data=reference_data,
        current_data=current_data
    )
    
    logger.info(f"Drift detection results: {drift_results}")
    return drift_results

@task
def trigger_retraining(
    drift_detected: bool,
    performance_degraded: bool,
    model_name: str = "financial_predictor"
) -> bool:
    """
    Trigger model retraining based on monitoring results.
    
    Args:
        drift_detected: Whether data drift was detected
        performance_degraded: Whether performance degraded
        model_name: Name of the model
        
    Returns:
        True if retraining should be triggered
    """
    logger = get_run_logger()
    
    should_retrain = drift_detected or performance_degraded
    
    if should_retrain:
        logger.warning(f"Retraining triggered for {model_name}")
        logger.info(f"Drift detected: {drift_detected}")
        logger.info(f"Performance degraded: {performance_degraded}")
    else:
        logger.info(f"No retraining needed for {model_name}")
    
    return should_retrain

@flow(name="market-master-training-pipeline")
def market_master_training_pipeline(
    asset_class: str = "crypto",
    symbol: str = "BTC/USD",
    num_samples: int = 10000,
    model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete MLOps training pipeline.
    
    Args:
        asset_class: Type of asset to train on
        symbol: Trading symbol
        num_samples: Number of training samples
        model_params: Model hyperparameters
        
    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    logger.info("Starting MLOps training pipeline")
    
    # Step 1: Data ingestion
    raw_data = ingest_financial_data(asset_class, symbol, num_samples)
    
    # Step 2: Feature engineering
    processed_data = preprocess_technical_indicators(raw_data)
    
    # Step 3: Data preparation
    features, labels = prepare_training_data(processed_data)
    
    # Step 4: Model training
    model = train_financial_model(features, labels, model_params)
    
    # Step 5: Model validation
    metrics = validate_model_performance(model, features, labels)
    
    # Step 6: Model registration
    model_version = register_model(model, metrics)
    
    # Step 7: Model deployment (if metrics are good)
    if metrics['accuracy'] > 0.4:  # Minimum accuracy threshold
        deployment_success = deploy_model(model_version)
    else:
        deployment_success = False
        logger.warning("Model accuracy below threshold, skipping deployment")
    
    results = {
        'model_version': model_version,
        'metrics': metrics,
        'deployment_success': deployment_success,
        'asset_class': asset_class,
        'symbol': symbol,
        'training_samples': num_samples
    }
    
    logger.info("MLOps training pipeline completed successfully")
    return results

@flow(name="market-master-monitoring-pipeline")
def market_master_monitoring_pipeline(
    model_name: str = "financial_predictor",
    window_size: int = 100
) -> Dict[str, Any]:
    """
    MLOps monitoring pipeline.
    
    Args:
        model_name: Name of the model to monitor
        window_size: Monitoring window size
        
    Returns:
        Dictionary with monitoring results
    """
    logger = get_run_logger()
    logger.info("Starting MLOps monitoring pipeline")
    
    # Step 1: Performance monitoring
    performance_results = monitor_model_performance(model_name, window_size)
    
    # Step 2: Data drift detection (simulated)
    # In a real scenario, you would compare current data with reference data
    drift_detected = performance_results.get('drift_detected', False)
    performance_degraded = performance_results.get('performance_degraded', False)
    
    # Step 3: Retraining decision
    retraining_needed = trigger_retraining(drift_detected, performance_degraded, model_name)
    
    results = {
        'performance_results': performance_results,
        'drift_detected': drift_detected,
        'performance_degraded': performance_degraded,
        'retraining_needed': retraining_needed
    }
    
    logger.info("MLOps monitoring pipeline completed")
    return results

@flow(name="market-master-complete-pipeline")
def market_master_complete_pipeline(
    asset_class: str = "crypto",
    symbol: str = "BTC/USD",
    num_samples: int = 10000,
    enable_monitoring: bool = True
) -> Dict[str, Any]:
    """
    Complete MLOps pipeline including training and monitoring.
    
    Args:
        asset_class: Type of asset to train on
        symbol: Trading symbol
        num_samples: Number of training samples
        enable_monitoring: Whether to enable monitoring
        
    Returns:
        Dictionary with complete pipeline results
    """
    logger = get_run_logger()
    logger.info("Starting complete MLOps pipeline")
    
    # Training pipeline
    training_results = market_master_training_pipeline(asset_class, symbol, num_samples)
    
    # Monitoring pipeline (if enabled)
    monitoring_results = None
    if enable_monitoring:
        monitoring_results = market_master_monitoring_pipeline()
    
    complete_results = {
        'training': training_results,
        'monitoring': monitoring_results,
        'pipeline_status': 'completed'
    }
    
    logger.info("Complete MLOps pipeline finished successfully")
    return complete_results

if __name__ == "__main__":
    # Run the complete pipeline
    results = market_master_complete_pipeline(
        asset_class="crypto",
        symbol="BTC/USD",
        num_samples=5000,
        enable_monitoring=True
    )
    print("Pipeline Results:", results) 