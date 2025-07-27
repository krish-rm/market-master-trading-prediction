# MLOps Pipeline Guide

## Overview

Market Master implements a comprehensive MLOps pipeline using Prefect for workflow orchestration, MLflow for experiment tracking, and automated monitoring for continuous model improvement. This guide covers pipeline design, implementation, deployment, and management.

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Prefect Workflow Orchestration](#prefect-workflow-orchestration)
- [Data Pipeline](#data-pipeline)
- [Training Pipeline](#training-pipeline)
- [Deployment Pipeline](#deployment-pipeline)
- [Monitoring Pipeline](#monitoring-pipeline)
- [Model Registry Management](#model-registry-management)
- [Pipeline Deployment](#pipeline-deployment)
- [Scheduling & Automation](#scheduling--automation)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Pipeline Architecture

### Overview

Market Master uses a modular pipeline architecture with Prefect for orchestration:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │ Training Pipeline│    │ Deployment      │
│                 │    │                 │    │ Pipeline        │
│ • Data Ingestion│───▶│ • Model Training│───▶│ • Model Registry│
│ • Preprocessing │    │ • Validation    │    │ • Versioning    │
│ • Feature Eng.  │    │ • Evaluation   │    │ • Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Monitoring      │    │ Orchestration   │    │ CI/CD Pipeline  │
│ Pipeline        │    │ (Prefect)       │    │ (GitHub Actions)│
│                 │    │                 │    │                 │
│ • Performance   │    │ • Task Scheduling│    │ • Automated     │
│ • Drift Detection│    │ • Error Handling│    │   Testing       │
│ • Alerting      │    │ • Retry Logic   │    │ • Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Pipeline Components

- **Data Pipeline**: Ingestion, preprocessing, and feature engineering
- **Training Pipeline**: Model training, validation, and evaluation
- **Deployment Pipeline**: Model registry, versioning, and deployment
- **Monitoring Pipeline**: Performance tracking and drift detection
- **Orchestration**: Prefect workflows with error handling and retries
- **CI/CD**: GitHub Actions for automated testing and deployment

## Prefect Workflow Orchestration

### Overview

Prefect provides robust workflow orchestration with built-in error handling, retries, and monitoring.

### Task Definition

```python
from prefect import task, flow, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    name="ingest_financial_data",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
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
    
    data_gen = MarketDataGenerator()
    data = data_gen.generate_training_data(
        asset_class=asset_class,
        symbol=symbol,
        num_samples=num_samples
    )
    
    logger.info(f"Successfully ingested {len(data)} samples")
    return data
```

### Flow Definition

```python
@flow(name="market-master-training-pipeline")
def market_master_training_pipeline(
    asset_class: str = "crypto",
    symbol: str = "BTC/USD",
    num_samples: int = 10000,
    model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete training pipeline for Market Master.
    
    Args:
        asset_class: Asset class for training
        symbol: Trading symbol
        num_samples: Number of training samples
        model_params: Model hyperparameters
        
    Returns:
        Training results and model metadata
    """
    logger = get_run_logger()
    logger.info(f"Starting training pipeline for {asset_class}/{symbol}")
    
    # Data pipeline
    raw_data = ingest_financial_data(asset_class, symbol, num_samples)
    processed_data = preprocess_technical_indicators(raw_data)
    features, labels = prepare_training_data(processed_data)
    
    # Training pipeline
    model = train_financial_model(features, labels, model_params)
    metrics = validate_model_performance(model, features, labels)
    
    # Deployment pipeline
    model_uri = register_model(model, metrics)
    deployment_success = deploy_model(model_uri)
    
    # Monitoring pipeline
    if deployment_success:
        monitor_model_performance()
    
    return {
        'model_uri': model_uri,
        'metrics': metrics,
        'deployment_success': deployment_success,
        'pipeline_status': 'completed'
    }
```

### Error Handling & Retries

```python
@task(
    retries=3,
    retry_delay_seconds=30,
    retry_jitter_factor=0.1
)
def train_financial_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model_params: Optional[Dict[str, Any]] = None
) -> ActionPredictor:
    """
    Train financial prediction model with error handling.
    """
    try:
        logger = get_run_logger()
        logger.info("Starting model training")
        
        # Initialize model
        model = ActionPredictor()
        if model_params:
            model.set_hyperparameters(model_params)
        
        # Train model
        model.train(features, labels)
        
        logger.info("Model training completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
```

## Data Pipeline

### Data Ingestion

```python
@task(retries=3, retry_delay_seconds=60)
def ingest_financial_data(
    asset_class: str = "crypto",
    symbol: str = "BTC/USD",
    num_samples: int = 10000
) -> pd.DataFrame:
    """Ingest financial market data for training."""
    
    logger = get_run_logger()
    logger.info(f"Ingesting {num_samples} samples for {symbol}")
    
    data_gen = MarketDataGenerator()
    data = data_gen.generate_training_data(
        asset_class=asset_class,
        symbol=symbol,
        num_samples=num_samples
    )
    
    logger.info(f"Successfully ingested {len(data)} samples")
    return data
```

### Data Preprocessing

```python
@task(retries=2, retry_delay_seconds=30)
def preprocess_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators from raw market data."""
    
    logger = get_run_logger()
    logger.info("Calculating technical indicators")
    
    # Calculate technical indicators
    data_gen = MarketDataGenerator()
    processed_data = data_gen.calculate_technical_indicators(data)
    
    logger.info(f"Calculated {len(processed_data.columns)} technical indicators")
    return processed_data
```

### Feature Engineering

```python
@task(retries=2, retry_delay_seconds=30)
def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels for training."""
    
    logger = get_run_logger()
    logger.info("Preparing training data")
    
    # Separate features and labels
    features = data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
    labels = data['action']
    
    # Handle missing values
    features = features.fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining rows with missing values
    valid_mask = ~(features.isnull().any(axis=1) | labels.isnull())
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    logger.info(f"Prepared {len(features)} training samples with {len(features.columns)} features")
    return features, labels
```

### Data Validation

```python
@task
def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality before training."""
    
    logger = get_run_logger()
    logger.info("Validating data quality")
    
    # Calculate quality metrics
    missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    duplicate_ratio = data.duplicated().sum() / len(data)
    
    # Check for required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'action']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    quality_report = {
        'missing_ratio': missing_ratio,
        'duplicate_ratio': duplicate_ratio,
        'missing_columns': missing_columns,
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'quality_score': 1.0 - (missing_ratio + duplicate_ratio)
    }
    
    # Raise error if quality is too low
    if quality_report['quality_score'] < 0.8:
        raise ValueError(f"Data quality too low: {quality_report['quality_score']:.2%}")
    
    logger.info(f"Data quality validation passed: {quality_report['quality_score']:.2%}")
    return quality_report
```

## Training Pipeline

### Model Training

```python
@task(retries=3, retry_delay_seconds=60)
def train_financial_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model_params: Optional[Dict[str, Any]] = None
) -> ActionPredictor:
    """Train financial prediction model."""
    
    logger = get_run_logger()
    logger.info("Starting model training")
    
    # Initialize model
    model = ActionPredictor()
    
    # Set hyperparameters if provided
    if model_params:
        model.set_hyperparameters(model_params)
        logger.info(f"Using custom hyperparameters: {model_params}")
    
    # Train model
    start_time = time.time()
    model.train(features, labels)
    training_time = time.time() - start_time
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    return model
```

### Model Validation

```python
@task(retries=2, retry_delay_seconds=30)
def validate_model_performance(
    model: ActionPredictor,
    features: pd.DataFrame,
    labels: pd.Series
) -> Dict[str, float]:
    """Validate model performance."""
    
    logger = get_run_logger()
    logger.info("Validating model performance")
    
    # Make predictions
    predictions = model.predict(features)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1_score': f1_score(labels, predictions, average='weighted')
    }
    
    # Check if performance meets minimum requirements
    if metrics['accuracy'] < 0.6:
        raise ValueError(f"Model accuracy too low: {metrics['accuracy']:.2%}")
    
    logger.info(f"Model validation passed - Accuracy: {metrics['accuracy']:.2%}")
    return metrics
```

### Model Evaluation

```python
@task
def evaluate_model_comprehensive(
    model: ActionPredictor,
    test_features: pd.DataFrame,
    test_labels: pd.Series
) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    
    logger = get_run_logger()
    logger.info("Performing comprehensive model evaluation")
    
    # Make predictions
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    
    # Calculate detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions, average='weighted'),
        'recall': recall_score(test_labels, predictions, average='weighted'),
        'f1_score': f1_score(test_labels, predictions, average='weighted'),
        'classification_report': classification_report(test_labels, predictions, output_dict=True),
        'confusion_matrix': confusion_matrix(test_labels, predictions).tolist()
    }
    
    # Calculate business metrics
    business_metrics = calculate_business_metrics(predictions, probabilities, test_labels)
    metrics.update(business_metrics)
    
    logger.info(f"Comprehensive evaluation completed - F1 Score: {metrics['f1_score']:.3f}")
    return metrics
```

## Deployment Pipeline

### Model Registration

```python
@task(retries=2, retry_delay_seconds=30)
def register_model(
    model: ActionPredictor,
    metrics: Dict[str, float],
    model_name: str = "financial_predictor"
) -> str:
    """Register model in MLflow registry."""
    
    logger = get_run_logger()
    logger.info(f"Registering model: {model_name}")
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Register model
    model_uri = registry.register_model(
        model=model.model,
        model_name=model_name,
        metrics=metrics,
        parameters=model.model_config,
        tags={
            'asset_class': model.asset_class,
            'training_date': datetime.now().isoformat(),
            'pipeline_version': '1.0.0'
        }
    )
    
    logger.info(f"Model registered successfully: {model_uri}")
    return model_uri
```

### Model Deployment

```python
@task(retries=2, retry_delay_seconds=30)
def deploy_model(
    model_version: str,
    model_name: str = "financial_predictor"
) -> bool:
    """Deploy model to production."""
    
    logger = get_run_logger()
    logger.info(f"Deploying model: {model_name}")
    
    try:
        # Initialize registry
        registry = ModelRegistry()
        
        # Transition model to production
        success = registry.transition_model_stage(
            model_name=model_name,
            version=model_version,
            stage="Production"
        )
        
        if success:
            logger.info(f"Model {model_name} v{model_version} deployed to production")
        else:
            logger.error(f"Failed to deploy model {model_name} v{model_version}")
        
        return success
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False
```

### Health Checks

```python
@task
def perform_deployment_health_checks(
    model_name: str,
    model_version: str
) -> Dict[str, Any]:
    """Perform health checks after deployment."""
    
    logger = get_run_logger()
    logger.info("Performing deployment health checks")
    
    health_checks = {
        'model_loading': False,
        'prediction_service': False,
        'performance_baseline': False
    }
    
    try:
        # Test model loading
        registry = ModelRegistry()
        model = registry.load_model(model_name, model_version)
        health_checks['model_loading'] = True
        
        # Test prediction service
        test_data = generate_test_data()
        predictions = model.predict(test_data)
        health_checks['prediction_service'] = len(predictions) > 0
        
        # Test performance baseline
        baseline_metrics = evaluate_model_performance(model, test_data)
        health_checks['performance_baseline'] = baseline_metrics['accuracy'] > 0.6
        
        logger.info("Health checks completed successfully")
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
    
    return health_checks
```

## Monitoring Pipeline

### Performance Monitoring

```python
@task(retries=2, retry_delay_seconds=30)
def monitor_model_performance(
    model_name: str = "financial_predictor",
    window_size: int = 100
) -> Dict[str, Any]:
    """Monitor model performance in production."""
    
    logger = get_run_logger()
    logger.info(f"Monitoring performance for model: {model_name}")
    
    # Get recent predictions and actual outcomes
    recent_data = get_recent_predictions(model_name, window_size)
    
    if len(recent_data) < window_size:
        logger.warning(f"Insufficient data for monitoring: {len(recent_data)} samples")
        return {'status': 'insufficient_data'}
    
    # Calculate performance metrics
    current_metrics = calculate_performance_metrics(recent_data)
    
    # Compare with baseline
    baseline_metrics = get_baseline_metrics(model_name)
    performance_degradation = compare_performance(current_metrics, baseline_metrics)
    
    # Generate alert if performance degraded
    if performance_degradation > 0.1:  # 10% degradation
        logger.warning(f"Performance degradation detected: {performance_degradation:.2%}")
        send_performance_alert(model_name, performance_degradation)
    
    return {
        'current_metrics': current_metrics,
        'baseline_metrics': baseline_metrics,
        'degradation': performance_degradation,
        'status': 'monitoring_completed'
    }
```

### Drift Detection

```python
@task(retries=2, retry_delay_seconds=30)
def check_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> Dict[str, Any]:
    """Check for data drift."""
    
    logger = get_run_logger()
    logger.info("Checking for data drift")
    
    # Initialize drift monitor
    drift_monitor = DataDriftMonitor(reference_data)
    
    # Check for drift
    drift_results = drift_monitor.check_data_drift(current_data)
    
    if drift_results['drift_detected']:
        logger.warning(f"Data drift detected in features: {drift_results['drift_features']}")
        send_drift_alert(drift_results)
    
    return drift_results
```

### Automated Retraining

```python
@task
def trigger_retraining(
    drift_detected: bool,
    performance_degraded: bool,
    model_name: str = "financial_predictor"
) -> bool:
    """Trigger model retraining if needed."""
    
    logger = get_run_logger()
    
    if drift_detected or performance_degraded:
        logger.info("Triggering model retraining")
        
        # Schedule retraining pipeline
        retraining_flow = market_master_training_pipeline.to_deployment(
            name=f"retraining-{model_name}",
            version="1.0.0",
            work_queue_name="retraining"
        )
        
        # Submit retraining job
        flow_run = retraining_flow.submit()
        
        logger.info(f"Retraining triggered - Flow run ID: {flow_run.id}")
        return True
    else:
        logger.info("No retraining needed")
        return False
```

## Model Registry Management

### Registry Operations

```python
class ModelRegistryManager:
    """Manage model registry operations."""
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return self.registry.list_models()
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model."""
        return self.registry.list_model_versions(model_name)
    
    def compare_models(self, model_name: str, version1: int, version2: int) -> Dict[str, Any]:
        """Compare two model versions."""
        return self.registry.compare_models(model_name, version1, version2)
    
    def rollback_model(self, model_name: str, target_version: int) -> bool:
        """Rollback to a previous model version."""
        return self.registry.transition_model_stage(
            model_name, target_version, "Production"
        )
    
    def archive_model(self, model_name: str, version: int) -> bool:
        """Archive a model version."""
        return self.registry.transition_model_stage(
            model_name, version, "Archived"
        )
```

### Version Management

```python
@task
def manage_model_versions(
    model_name: str,
    max_versions: int = 10
) -> Dict[str, Any]:
    """Manage model versions and cleanup old versions."""
    
    logger = get_run_logger()
    logger.info(f"Managing versions for model: {model_name}")
    
    registry = ModelRegistry()
    versions = registry.list_model_versions(model_name)
    
    # Sort versions by creation date
    versions.sort(key=lambda x: x['creation_timestamp'], reverse=True)
    
    # Keep only the latest versions
    if len(versions) > max_versions:
        versions_to_delete = versions[max_versions:]
        
        for version in versions_to_delete:
            registry.delete_model_version(model_name, version['version'])
            logger.info(f"Deleted version {version['version']}")
    
    return {
        'total_versions': len(versions),
        'versions_kept': min(len(versions), max_versions),
        'versions_deleted': max(0, len(versions) - max_versions)
    }
```

## Pipeline Deployment

### Prefect Deployment

```python
def deploy_training_pipeline():
    """Deploy the training pipeline to Prefect."""
    
    # Create deployment
    deployment = market_master_training_pipeline.to_deployment(
        name="market-master-training",
        version="1.0.0",
        work_queue_name="training",
        tags=["training", "mlops"],
        description="Market Master training pipeline"
    )
    
    # Apply deployment
    deployment.apply()
    
    print("Training pipeline deployed successfully")

def deploy_monitoring_pipeline():
    """Deploy the monitoring pipeline to Prefect."""
    
    # Create deployment
    deployment = market_master_monitoring_pipeline.to_deployment(
        name="market-master-monitoring",
        version="1.0.0",
        work_queue_name="monitoring",
        tags=["monitoring", "mlops"],
        description="Market Master monitoring pipeline"
    )
    
    # Apply deployment
    deployment.apply()
    
    print("Monitoring pipeline deployed successfully")
```

### Docker Deployment

```dockerfile
# Dockerfile for MLOps pipeline
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy pipeline code
COPY src/ ./src/
COPY workflows/ ./workflows/

# Set environment variables
ENV PREFECT_API_URL=http://localhost:4200/api
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Run pipeline
CMD ["python", "-m", "prefect", "deployment", "apply", "workflows/mlops_pipeline.py"]
```

### Kubernetes Deployment

```yaml
# k8s/pipeline-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market-master-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: market-master-pipeline
  template:
    metadata:
      labels:
        app: market-master-pipeline
    spec:
      containers:
      - name: pipeline
        image: market-master-pipeline:latest
        env:
        - name: PREFECT_API_URL
          value: "http://prefect-server:4200/api"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## Scheduling & Automation

### Cron Scheduling

```python
from prefect.server.schemas.schedules import CronSchedule

# Schedule training pipeline
training_deployment = market_master_training_pipeline.to_deployment(
    name="scheduled-training",
    schedule=CronSchedule(cron="0 2 * * 1"),  # Every Monday at 2 AM
    work_queue_name="training"
)

# Schedule monitoring pipeline
monitoring_deployment = market_master_monitoring_pipeline.to_deployment(
    name="scheduled-monitoring",
    schedule=CronSchedule(cron="*/15 * * * *"),  # Every 15 minutes
    work_queue_name="monitoring"
)
```

### Event-Driven Triggers

```python
@task
def trigger_pipeline_on_event(event_data: Dict[str, Any]) -> bool:
    """Trigger pipeline based on events."""
    
    logger = get_run_logger()
    
    # Check if retraining is needed
    if event_data.get('drift_detected') or event_data.get('performance_degraded'):
        logger.info("Event triggered retraining")
        
        # Submit training pipeline
        flow_run = market_master_training_pipeline.submit(
            asset_class=event_data.get('asset_class', 'crypto'),
            symbol=event_data.get('symbol', 'BTC/USD')
        )
        
        return True
    
    return False
```

### Conditional Execution

```python
@flow
def conditional_training_pipeline(
    asset_class: str,
    symbol: str,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """Conditional training pipeline."""
    
    # Check if retraining is needed
    if not force_retrain:
        current_performance = check_current_performance()
        if current_performance['status'] == 'healthy':
            return {'status': 'no_retraining_needed'}
    
    # Proceed with training
    return market_master_training_pipeline(asset_class, symbol)
```

## Best Practices

### 1. Pipeline Design

- **Modularity**: Break pipelines into small, focused tasks
- **Reusability**: Design tasks to be reusable across different pipelines
- **Error Handling**: Implement comprehensive error handling and retries
- **Monitoring**: Add logging and monitoring at each step

### 2. Data Management

- **Versioning**: Version your data and track data lineage
- **Validation**: Validate data at each pipeline stage
- **Caching**: Use caching for expensive operations
- **Cleanup**: Implement data cleanup and retention policies

### 3. Model Management

- **Versioning**: Use semantic versioning for models
- **Registry**: Maintain a centralized model registry
- **Rollback**: Implement easy rollback mechanisms
- **Testing**: Test models before deployment

### 4. Monitoring & Alerting

- **Metrics**: Track key metrics at each pipeline stage
- **Alerts**: Set up alerts for pipeline failures
- **Dashboards**: Create dashboards for pipeline monitoring
- **Logging**: Implement comprehensive logging

### 5. Security

- **Secrets**: Use Prefect secrets for sensitive data
- **Access Control**: Implement proper access controls
- **Audit**: Maintain audit trails for all operations
- **Compliance**: Ensure compliance with data regulations

## Troubleshooting

### Common Issues

#### 1. Pipeline Failures

**Symptoms**: Tasks failing with errors

**Solutions**:
- Check task logs for specific error messages
- Verify input data quality and format
- Ensure all dependencies are installed
- Check resource constraints (memory, CPU)

#### 2. Slow Pipeline Execution

**Symptoms**: Pipeline taking too long to complete

**Solutions**:
- Optimize data processing with caching
- Use parallel execution where possible
- Scale up compute resources
- Profile and optimize slow tasks

#### 3. Model Registry Issues

**Symptoms**: Models not registering or loading properly

**Solutions**:
- Check MLflow server connectivity
- Verify model format and serialization
- Check registry permissions
- Validate model metadata

#### 4. Monitoring Alerts

**Symptoms**: Too many false positive alerts

**Solutions**:
- Adjust alert thresholds based on historical data
- Implement alert cooldown periods
- Use different severity levels
- Review and tune alert rules

### Debugging Commands

```bash
# Check Prefect server status
prefect server status

# List deployments
prefect deployment ls

# Check flow runs
prefect flow-run ls

# View task logs
prefect flow-run logs <flow-run-id>

# Check MLflow server
curl http://localhost:5000/health

# Check model registry
mlflow models list

# Monitor pipeline metrics
prefect deployment logs <deployment-name>
```

### Performance Optimization

```python
# Optimize task execution
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def expensive_computation(data: pd.DataFrame) -> pd.DataFrame:
    """Cache expensive computations."""
    return perform_expensive_computation(data)

# Parallel execution
@task
def parallel_processing(data_chunks: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Process data chunks in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, data_chunks))
    return results

# Resource optimization
@task(memory_mb=1024, cpu=2)
def memory_intensive_task(data: pd.DataFrame) -> pd.DataFrame:
    """Task with specific resource requirements."""
    return process_large_dataset(data)
```

---

For more detailed information about specific pipeline implementations, see the source code in `src/mlops/pipeline.py` and `workflows/mlops_pipeline.py`. 