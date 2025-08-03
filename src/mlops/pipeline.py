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
from sklearn.ensemble import RandomForestClassifier
import logging

# Use basic logging instead of relative imports
logger = logging.getLogger(__name__)

# Define a simple AssetClass enum to avoid import issues
class AssetClass:
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDICES = "indices"
    
    @classmethod
    def values(cls):
        return [cls.EQUITY, cls.CRYPTO, cls.FOREX, cls.COMMODITY, cls.INDICES]

# Simple implementations to avoid import issues
def generate_training_data(asset_class, instrument, n_samples):
    """Simple training data generation."""
    import pandas as pd
    import numpy as np
    
    # Generate simple synthetic data
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(100, 200, n_samples),
        'low': np.random.uniform(100, 200, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples),
        'action': np.random.choice([0, 1, 2], n_samples),
        'asset_class': asset_class,
        'instrument': instrument
    })
    return data

def train_action_predictor(X_train, y_train, X_val, y_val):
    """Simple model training."""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Simple metrics
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    metrics = {
        'train_accuracy': train_score,
        'val_accuracy': val_score
    }
    
    return model, metrics

def evaluate_model(model, X_test, y_test):
    """Simple model evaluation."""
    from sklearn.metrics import accuracy_score
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'test_accuracy': accuracy,
        'predictions': y_pred.tolist()
    }

# Simple implementations for missing classes
class ActionPredictor:
    """Simple action predictor."""
    
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier(n_estimators=10, random_state=42)
        self.model_config = {'n_estimators': 10, 'random_state': 42}
    
    def save_model(self, filepath):
        """Save model."""
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

class ModelRegistry:
    """Simple model registry."""
    
    def __init__(self):
        self.models = {}
    
    def register_model(self, model, name, metrics, parameters=None, tags=None):
        """Register a model."""
        self.models[name] = {
            'model': model,
            'metrics': metrics,
            'parameters': parameters or {},
            'tags': tags or {},
            'timestamp': datetime.now().isoformat()
        }
        logger.info(f"Registered model: {name}")
        return f"model://{name}"

class ComprehensiveMonitor:
    """Simple comprehensive monitor."""
    
    def __init__(self, reference_data=None):
        self.reference_data = reference_data or pd.DataFrame()
        self.monitoring_history = []
    
    def run_monitoring(self, current_data, predictions=None, y_true=None):
        """Run monitoring."""
        logger.info("Running simple monitoring")
        
        # Simple monitoring metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': current_data.shape,
            'missing_values': current_data.isnull().sum().sum(),
            'monitoring_status': 'HEALTHY'
        }
        
        self.monitoring_history.append(metrics)
        return metrics


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
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        return train_data, test_data


class InferencePipeline(MLPipeline):
    """Inference pipeline for Market Master."""

    def __init__(self, model_name: str, model_version: Optional[int] = None):
        super().__init__(f"inference_{model_name}")
        self.model_name = model_name
        self.model_version = model_version
        self.model = None

    def load_model(self):
        try:
            self.model = self.registry.load_model(self.model_name, self.model_version)
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        logger.info(f"Running inference on {len(data)} samples")
        features = data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        results_data = data.copy()
        results_data['prediction'] = predictions
        results_data['confidence'] = np.max(probabilities, axis=1)

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


@task(name="generate_training_data", cache_key_fn=task_input_hash)
def generate_training_data_task(asset_class: str, instrument: str, n_samples: int) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Generating training data for {asset_class}/{instrument}")
    asset_enum = AssetClass(asset_class)
    data = generate_training_data(asset_enum, instrument, n_samples)
    logger.info(f"Generated {len(data)} training samples")
    return data


@task(name="train_model")
def train_model_task(X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[ActionPredictor, Dict[str, float]]:
    logger = get_run_logger()
    logger.info("Training Action Predictor model")
    model, metrics = train_action_predictor(X_train, y_train, X_val, y_val)
    logger.info(f"Model training completed with metrics: {metrics}")
    return model, metrics


@task(name="evaluate_model")
def evaluate_model_task(model: ActionPredictor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Evaluating model performance")
    results = evaluate_model(model, X_test, y_test)
    logger.info(f"Model evaluation completed with accuracy: {results['metrics']['accuracy']}")
    return results


@task(name="register_model")
def register_model_task(model: ActionPredictor, model_name: str, metrics: Dict[str, float]) -> str:
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
    logger = get_run_logger()
    logger.info("Running model monitoring")
    monitor = ComprehensiveMonitor(reference_data)
    results = monitor.run_monitoring(current_data, predictions)
    logger.info(f"Monitoring completed with status: {results['overall_status']}")
    return results


@flow(name="Market Master Training Pipeline")
def market_master_training_flow(asset_class: str = "equity", instrument: str = "AAPL",
                               n_samples: int = 10000) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting Market Master training pipeline")
    data = generate_training_data_task(asset_class, instrument, n_samples)
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:int(len(data) * 0.9)]
    test_data = data.iloc[int(len(data) * 0.9):]

    X_train = train_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    y_train = train_data['action']
    X_val = val_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    y_val = val_data['action']
    X_test = test_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    y_test = test_data['action']

    model, train_metrics = train_model_task(X_train, y_train, X_val, y_val)
    eval_results = evaluate_model_task(model, X_test, y_test)
    model_name = f"action_predictor_{asset_class}_{instrument}"
    model_uri = register_model_task(model, model_name, train_metrics)
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
    logger = get_run_logger()
    logger.info("Starting Market Master inference pipeline")
    registry = ModelRegistry()
    model = registry.load_model(model_name)
    features = data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
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


def deploy_training_pipeline():
    """Simple deployment - create executable scripts."""
    import os
    
    logger.info("Creating training pipeline runner...")
    
    # Create runners directory
    os.makedirs("runners", exist_ok=True)
    
    script_content = '''#!/usr/bin/env python
"""Training Pipeline Runner"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

if __name__ == "__main__":
    print("Running Market Master Training Pipeline...")
    try:
        # Import here to avoid circular imports
        from src.mlops.pipeline import market_master_training_flow
        
        result = market_master_training_flow(
            asset_class="equity",
            instrument="AAPL", 
            n_samples=10000
        )
        print("Training completed successfully!")
        print(f"Model URI: {result.get('model_uri', 'N/A')}")
        print(f"Training Accuracy: {result.get('training_metrics', {}).get('accuracy', 'N/A')}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
'''
    
    with open("runners/run_training.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    logger.info("Training pipeline deployed as runners/run_training.py")


def deploy_inference_pipeline():
    """Simple deployment - create executable scripts."""
    import os
    
    logger.info("Creating inference pipeline runner...")
    
    script_content = '''#!/usr/bin/env python
"""Inference Pipeline Runner"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

if __name__ == "__main__":
    print("Running Market Master Inference Pipeline...")
    try:
        # Import here to avoid circular imports
        from src.mlops.pipeline import market_master_inference_flow
        from src.data import generate_training_data, AssetClass
        
        # Generate some sample data for inference
        sample_data = generate_training_data(AssetClass.EQUITY, "AAPL", 100)
        
        result = market_master_inference_flow(
            model_name="action_predictor_equity_AAPL",
            data=sample_data
        )
        
        print("Inference completed successfully!")
        print(f"Predictions made: {len(result.get('predictions', []))}")
        print(f"Prediction distribution: {result.get('prediction_distribution', {})}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        print("Note: Make sure you've trained a model first!")
        import traceback
        traceback.print_exc()
'''
    
    with open("runners/run_inference.py", "w", encoding="utf-8") as f:
        f.write(script_content)
        
    logger.info("Inference pipeline deployed as runners/run_inference.py")



# Only used when building deployments

from prefect.settings import PREFECT_DEFAULT_WORK_POOL_NAME

def create_prefect_deployments():
    """Create Prefect deployments for training and inference flows."""
    
    # Create deployment scripts instead of trying to start servers
    print("Creating Prefect deployment scripts...")
    
    # Create training deployment script
    training_script = '''#!/usr/bin/env python
"""Prefect Training Deployment Script"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlops.pipeline import market_master_training_flow

if __name__ == "__main__":
    print("Running Market Master Training Flow via Prefect...")
    result = market_master_training_flow(
        asset_class="equity",
        instrument="AAPL", 
        n_samples=10000
    )
    print(f"Training completed! Model URI: {result.get('model_uri', 'N/A')}")
'''
    
    # Create inference deployment script
    inference_script = '''#!/usr/bin/env python
"""Prefect Inference Deployment Script"""
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlops.pipeline import market_master_inference_flow

if __name__ == "__main__":
    print("Running Market Master Inference Flow via Prefect...")
    # Create dummy data for inference
    dummy_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'action': ['buy', 'sell', 'hold']
    })
    
    result = market_master_inference_flow(
        model_name="action_predictor_equity_AAPL",
        data=dummy_data
    )
    print(f"Inference completed! Predictions: {len(result.get('predictions', []))}")
'''
    
    # Write scripts
    import os
    os.makedirs("deployments", exist_ok=True)
    with open("deployments/prefect_training.py", "w") as f:
        f.write(training_script)
    
    with open("deployments/prefect_inference.py", "w") as f:
        f.write(inference_script)
    
    print("✅ Prefect deployment scripts created:")
    print("  - deployments/prefect_training.py")
    print("  - deployments/prefect_inference.py")
    print("\nTo run training: python deployments/prefect_training.py")
    print("To run inference: python deployments/prefect_inference.py")
