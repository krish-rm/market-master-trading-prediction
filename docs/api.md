# API Documentation

## Overview

Market Master provides a comprehensive set of APIs for financial market prediction, model management, and MLOps operations. This documentation covers all available endpoints, functions, and usage examples.

## Table of Contents

- [Core Application APIs](#core-application-apis)
- [Model APIs](#model-apis)
- [Data APIs](#data-apis)
- [MLOps APIs](#mlops-apis)
- [Utility APIs](#utility-apis)
- [Configuration APIs](#configuration-apis)
- [Web Application APIs](#web-application-apis)
- [Error Handling](#error-handling)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)

## Core Application APIs

### MarketMasterApp

The main application class that orchestrates all Market Master operations.

#### `__init__()`

Initialize the Market Master application.

```python
from src.main import MarketMasterApp

app = MarketMasterApp()
```

**Returns**: MarketMasterApp instance

#### `run_demo(asset_class: str, instrument: str, n_samples: int) -> Dict[str, Any]`

Run a complete Market Master demo with data generation, model training, and evaluation.

```python
results = app.run_demo(
    asset_class="crypto",
    instrument="BTC/USD", 
    n_samples=10000
)
```

**Parameters**:
- `asset_class` (str): Asset class ("equity", "crypto", "forex", "commodity", "indices")
- `instrument` (str): Specific instrument (e.g., "AAPL", "BTC/USD", "EUR/USD")
- `n_samples` (int): Number of training samples

**Returns**: Dictionary with demo results including:
- `demo_start_time`: ISO timestamp
- `asset_class`: Asset class used
- `instrument`: Instrument used
- `n_samples`: Number of samples
- `steps`: Dictionary with step-by-step results

**Example Response**:
```json
{
    "demo_start_time": "2024-01-15T10:30:00",
    "asset_class": "crypto",
    "instrument": "BTC/USD",
    "n_samples": 10000,
    "steps": {
        "data_generation": {
            "status": "success",
            "data_shape": [10000, 28],
            "features_count": 25,
            "label_distribution": {
                "buy": 2500,
                "sell": 2500,
                "hold": 3000,
                "strong_buy": 1000,
                "strong_sell": 1000
            }
        },
        "model_training": {
            "status": "success",
            "training_metrics": {
                "accuracy": 0.69,
                "f1_score": 0.72,
                "precision": 0.71,
                "recall": 0.69
            },
            "model_config": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            }
        }
    }
}
```

#### `run_inference(model_name: str, data: pd.DataFrame) -> Dict[str, Any]`

Run inference using a trained model.

```python
predictions = app.run_inference("action_predictor_v1", market_data)
```

**Parameters**:
- `model_name` (str): Name of the model to use
- `data` (pd.DataFrame): Market data with OHLCV columns

**Returns**: Dictionary with inference results

#### `list_models() -> Dict[str, Any]`

List all available models in the registry.

```python
models = app.list_models()
```

**Returns**: Dictionary with model information

#### `get_model_info(model_name: str) -> Dict[str, Any]`

Get detailed information about a specific model.

```python
model_info = app.get_model_info("action_predictor_v1")
```

**Parameters**:
- `model_name` (str): Name of the model

**Returns**: Dictionary with model metadata

## Model APIs

### ActionPredictor

The core prediction model for trading actions.

#### `__init__(model_path: Optional[str] = None, use_mlflow: bool = True)`

Initialize the Action Predictor model.

```python
from src.models.action_predictor import ActionPredictor

# Initialize with default settings
model = ActionPredictor()

# Initialize with pre-trained model
model = ActionPredictor(model_path="models/action_predictor_v1.pkl")

# Initialize without MLflow
model = ActionPredictor(use_mlflow=False)
```

**Parameters**:
- `model_path` (str, optional): Path to pre-trained model file
- `use_mlflow` (bool): Whether to use MLflow for tracking

#### `prepare_features(data: pd.DataFrame) -> pd.DataFrame`

Prepare features from raw market data.

```python
features = model.prepare_features(market_data)
```

**Parameters**:
- `data` (pd.DataFrame): Raw market data with OHLCV columns

**Returns**: DataFrame with 25 technical indicators

**Required Columns**:
- `open`: Opening prices
- `high`: High prices
- `low`: Low prices
- `close`: Closing prices
- `volume`: Trading volume

**Generated Features**:
```python
feature_names = [
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'obv', 'vwap', 'stoch_k', 'stoch_d',
    'williams_r', 'cci', 'adx', 'supertrend',
    'atr', 'atr_ratio', 'price_change', 'volume_change',
    'pivot_point'
]
```

#### `train(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]`

Train the model on provided data.

```python
metrics = model.train(X_train, y_train, X_val, y_val)
```

**Parameters**:
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training labels
- `X_val` (pd.DataFrame): Validation features
- `y_val` (pd.Series): Validation labels

**Returns**: Dictionary with training metrics:
- `accuracy`: Overall accuracy
- `f1_score`: F1 score
- `precision`: Precision score
- `recall`: Recall score
- `training_time`: Training duration in seconds

#### `predict(features: pd.DataFrame) -> np.ndarray`

Make predictions on input features.

```python
predictions = model.predict(features)
```

**Parameters**:
- `features` (pd.DataFrame): Input features

**Returns**: Array of predicted actions:
- `'buy'`: Moderate buy signal
- `'sell'`: Moderate sell signal
- `'hold'`: No action recommended
- `'strong_buy'`: Strong buy signal
- `'strong_sell'`: Strong sell signal

#### `predict_proba(features: pd.DataFrame) -> np.ndarray`

Get prediction probabilities for all classes.

```python
probabilities = model.predict_proba(features)
```

**Parameters**:
- `features` (pd.DataFrame): Input features

**Returns**: Array of probability arrays for each class

#### `predict_with_confidence(features: pd.DataFrame) -> List[Dict[str, Any]]`

Make predictions with confidence scores and probabilities.

```python
detailed_predictions = model.predict_with_confidence(features)
```

**Parameters**:
- `features` (pd.DataFrame): Input features

**Returns**: List of dictionaries with:
- `action`: Predicted action
- `confidence`: Confidence score (0-1)
- `probabilities`: Dictionary of class probabilities

**Example Response**:
```python
[
    {
        "action": "buy",
        "confidence": 0.85,
        "probabilities": {
            "buy": 0.45,
            "sell": 0.15,
            "hold": 0.25,
            "strong_buy": 0.10,
            "strong_sell": 0.05
        }
    }
]
```

#### `get_feature_importance() -> Dict[str, float]`

Get feature importance scores.

```python
importance = model.get_feature_importance()
```

**Returns**: Dictionary mapping feature names to importance scores

#### `save_model(filepath: str)`

Save the trained model to file.

```python
model.save_model("models/action_predictor_v1.pkl")
```

**Parameters**:
- `filepath` (str): Path to save the model

#### `load_model(filepath: str) -> bool`

Load a trained model from file.

```python
success = model.load_model("models/action_predictor_v1.pkl")
```

**Parameters**:
- `filepath` (str): Path to the model file

**Returns**: Boolean indicating success

#### `evaluate(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]`

Evaluate model performance on test data.

```python
results = model.evaluate(X_test, y_test)
```

**Parameters**:
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test labels

**Returns**: Dictionary with evaluation metrics and detailed results

### ModelFactory

Factory for creating and managing ML models.

#### `__init__()`

Initialize the model factory.

```python
from src.models.model_factory import ModelFactory

factory = ModelFactory()
```

#### `create_model(model_type: str, config_name: str = 'default', **kwargs) -> Any`

Create a model instance with specified configuration.

```python
# Create with default configuration
model = factory.create_model('action_predictor')

# Create with specific configuration
model = factory.create_model('action_predictor', config_name='accurate')

# Create with custom parameters
model = factory.create_model('action_predictor', n_estimators=200, max_depth=15)
```

**Parameters**:
- `model_type` (str): Type of model ("action_predictor")
- `config_name` (str): Configuration preset ("default", "fast", "accurate")
- `**kwargs`: Additional model parameters

**Available Configurations**:
- `default`: Balanced performance and speed
- `fast`: Optimized for speed
- `accurate`: Maximum accuracy

#### `get_available_models() -> List[str]`

Get list of available model types.

```python
models = factory.get_available_models()
```

**Returns**: List of available model types

#### `get_available_configs(model_type: str) -> List[str]`

Get available configurations for a model type.

```python
configs = factory.get_available_configs('action_predictor')
```

**Parameters**:
- `model_type` (str): Model type

**Returns**: List of available configurations

## Data APIs

### MarketDataGenerator

Generate realistic market data for training and testing.

#### `__init__(asset_class: AssetClass, instrument: str, seed: Optional[int] = None)`

Initialize the market data generator.

```python
from src.data.data_generator import MarketDataGenerator
from src.data.asset_classes import AssetClass

generator = MarketDataGenerator(
    asset_class=AssetClass.CRYPTO,
    instrument="BTC/USD",
    seed=42
)
```

**Parameters**:
- `asset_class` (AssetClass): Asset class enum
- `instrument` (str): Specific instrument
- `seed` (int, optional): Random seed for reproducibility

#### `generate_tick_data(n_ticks: int = 10000, start_time: Optional[datetime] = None) -> pd.DataFrame`

Generate realistic tick data.

```python
data = generator.generate_tick_data(n_ticks=1000)
```

**Parameters**:
- `n_ticks` (int): Number of ticks to generate
- `start_time` (datetime, optional): Start time for data generation

**Returns**: DataFrame with OHLCV data and metadata

**Columns**:
- `timestamp`: Timestamp
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `asset_class`: Asset class
- `instrument`: Instrument name
- `session`: Market session info
- `volatility_regime`: Volatility regime

#### `generate_training_data(n_samples: int = 10000, seed: Optional[int] = None) -> pd.DataFrame`

Generate training data with labels.

```python
training_data = generator.generate_training_data(n_samples=10000)
```

**Parameters**:
- `n_samples` (int): Number of training samples
- `seed` (int, optional): Random seed

**Returns**: DataFrame with features and action labels

#### `validate_generated_data(data: pd.DataFrame) -> Dict[str, Any]`

Validate generated data quality.

```python
validation = generator.validate_generated_data(data)
```

**Parameters**:
- `data` (pd.DataFrame): Generated data

**Returns**: Dictionary with validation results

### Asset Classes

#### `AssetClass`

Enumeration of supported asset classes.

```python
from src.data.asset_classes import AssetClass

# Available asset classes
AssetClass.EQUITY      # Stocks, ETFs
AssetClass.CRYPTO      # Cryptocurrencies
AssetClass.FOREX       # Foreign exchange
AssetClass.COMMODITY   # Commodities
AssetClass.INDICES     # Market indices
```

#### `get_asset_config(asset_class: AssetClass) -> Dict[str, Any]`

Get configuration for an asset class.

```python
config = get_asset_config(AssetClass.CRYPTO)
```

**Parameters**:
- `asset_class` (AssetClass): Asset class enum

**Returns**: Dictionary with asset class configuration

#### `get_instrument_config(asset_class: AssetClass, instrument: str) -> Dict[str, Any]`

Get configuration for a specific instrument.

```python
config = get_instrument_config(AssetClass.CRYPTO, "BTC/USD")
```

**Parameters**:
- `asset_class` (AssetClass): Asset class enum
- `instrument` (str): Instrument name

**Returns**: Dictionary with instrument configuration

## MLOps APIs

### ModelRegistry

MLflow-based model registry for versioning and deployment.

#### `__init__(tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None)`

Initialize the model registry.

```python
from src.mlops.model_registry import ModelRegistry

# Initialize with default settings
registry = ModelRegistry()

# Initialize with custom URIs
registry = ModelRegistry(
    tracking_uri="http://localhost:5000",
    registry_uri="http://localhost:5000"
)
```

**Parameters**:
- `tracking_uri` (str, optional): MLflow tracking URI
- `registry_uri` (str, optional): MLflow registry URI

#### `register_model(model, model_name: str, metrics: Dict[str, float], parameters: Dict[str, Any] = None, tags: Dict[str, str] = None) -> str`

Register a model in the registry.

```python
model_uri = registry.register_model(
    model=action_predictor,
    model_name="action_predictor_v1",
    metrics={"accuracy": 0.69, "f1_score": 0.72},
    parameters={"n_estimators": 100, "max_depth": 10},
    tags={"environment": "production", "version": "1.0"}
)
```

**Parameters**:
- `model`: Trained model object
- `model_name` (str): Name for the model
- `metrics` (Dict[str, float]): Performance metrics
- `parameters` (Dict[str, Any], optional): Model parameters
- `tags` (Dict[str, str], optional): Additional tags

**Returns**: Model URI string

#### `load_model(model_name: str, version: Optional[int] = None, stage: str = "Production") -> Any`

Load a model from the registry.

```python
# Load latest production model
model = registry.load_model("action_predictor")

# Load specific version
model = registry.load_model("action_predictor", version=1)

# Load staging model
model = registry.load_model("action_predictor", stage="Staging")
```

**Parameters**:
- `model_name` (str): Name of the model
- `version` (int, optional): Specific version number
- `stage` (str): Model stage ("Production", "Staging", "Archived")

**Returns**: Loaded model object

#### `list_models() -> List[Dict[str, Any]]`

List all models in the registry.

```python
models = registry.list_models()
```

**Returns**: List of model metadata dictionaries

#### `list_model_versions(model_name: str) -> List[Dict[str, Any]]`

List all versions of a specific model.

```python
versions = registry.list_model_versions("action_predictor")
```

**Parameters**:
- `model_name` (str): Name of the model

**Returns**: List of version metadata dictionaries

#### `transition_model_stage(model_name: str, version: int, stage: str) -> bool`

Transition a model to a different stage.

```python
success = registry.transition_model_stage(
    model_name="action_predictor",
    version=1,
    stage="Production"
)
```

**Parameters**:
- `model_name` (str): Name of the model
- `version` (int): Version number
- `stage` (str): Target stage

**Returns**: Boolean indicating success

#### `delete_model_version(model_name: str, version: int) -> bool`

Delete a specific model version.

```python
success = registry.delete_model_version("action_predictor", version=1)
```

**Parameters**:
- `model_name` (str): Name of the model
- `version` (int): Version number

**Returns**: Boolean indicating success

#### `get_model_metadata(model_name: str, version: int) -> Dict[str, Any]`

Get detailed metadata for a model version.

```python
metadata = registry.get_model_metadata("action_predictor", version=1)
```

**Parameters**:
- `model_name` (str): Name of the model
- `version` (int): Version number

**Returns**: Dictionary with model metadata

#### `compare_models(model_name: str, version1: int, version2: int) -> Dict[str, Any]`

Compare two model versions.

```python
comparison = registry.compare_models("action_predictor", version1=1, version2=2)
```

**Parameters**:
- `model_name` (str): Name of the model
- `version1` (int): First version number
- `version2` (int): Second version number

**Returns**: Dictionary with comparison results

### ComprehensiveMonitor

Model monitoring and drift detection.

#### `__init__(reference_data: pd.DataFrame, model_name: str)`

Initialize the monitor with reference data.

```python
from src.mlops.monitoring import ComprehensiveMonitor

monitor = ComprehensiveMonitor(
    reference_data=training_data,
    model_name="action_predictor"
)
```

**Parameters**:
- `reference_data` (pd.DataFrame): Reference dataset
- `model_name` (str): Name of the model to monitor

#### `monitor_data_quality(current_data: pd.DataFrame) -> Dict[str, Any]`

Monitor data quality metrics.

```python
quality_report = monitor.monitor_data_quality(current_data)
```

**Parameters**:
- `current_data` (pd.DataFrame): Current data to monitor

**Returns**: Dictionary with data quality metrics

#### `monitor_data_drift(current_data: pd.DataFrame) -> Dict[str, Any]`

Monitor data drift.

```python
drift_report = monitor.monitor_data_drift(current_data)
```

**Parameters**:
- `current_data` (pd.DataFrame): Current data to monitor

**Returns**: Dictionary with drift detection results

#### `monitor_model_performance(predictions: np.ndarray, actual: np.ndarray) -> Dict[str, Any]`

Monitor model performance.

```python
performance_report = monitor.monitor_model_performance(predictions, actual)
```

**Parameters**:
- `predictions` (np.ndarray): Model predictions
- `actual` (np.ndarray): Actual values

**Returns**: Dictionary with performance metrics

## Utility APIs

### Technical Indicators

#### `calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame`

Calculate all technical indicators for market data.

```python
from src.utils.helpers import calculate_technical_indicators

indicators = calculate_technical_indicators(market_data)
```

**Parameters**:
- `data` (pd.DataFrame): OHLCV market data

**Returns**: DataFrame with 25 technical indicators

#### Individual Indicator Functions

```python
from src.utils.helpers import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_obv, calculate_vwap, calculate_stochastic,
    calculate_williams_r, calculate_cci, calculate_adx,
    calculate_supertrend, calculate_atr, calculate_pivot_point
)

# RSI (Relative Strength Index)
rsi = calculate_rsi(prices, period=14)

# MACD (Moving Average Convergence Divergence)
macd, signal, histogram = calculate_macd(prices, fast=12, slow=26, signal=9)

# Bollinger Bands
upper, middle, lower = calculate_bollinger_bands(prices, period=20, std_dev=2)

# OBV (On-Balance Volume)
obv = calculate_obv(prices, volumes)

# VWAP (Volume Weighted Average Price)
vwap = calculate_vwap(prices, volumes)

# Stochastic
stoch_k, stoch_d = calculate_stochastic(high, low, close, period=14)

# Williams %R
williams_r = calculate_williams_r(high, low, close, period=14)

# CCI (Commodity Channel Index)
cci = calculate_cci(high, low, close, period=20)

# ADX (Average Directional Index)
adx = calculate_adx(high, low, close, period=14)

# SuperTrend
supertrend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)

# ATR (Average True Range)
atr = calculate_atr(high, low, close, period=14)

# Pivot Point
pivot = calculate_pivot_point(high, low, close)
```

### Data Validation

#### `validate_market_data(data: pd.DataFrame) -> Dict[str, Any]`

Validate market data quality.

```python
from src.utils.helpers import validate_market_data

validation = validate_market_data(market_data)
```

**Parameters**:
- `data` (pd.DataFrame): Market data to validate

**Returns**: Dictionary with validation results:
- `is_valid`: Boolean indicating if data is valid
- `issues`: List of validation issues
- `data_quality_score`: Quality score (0-1)
- `missing_values`: Missing value counts
- `outliers`: Outlier information
- `data_types`: Data type information

### Data Storage

#### `save_data(data: pd.DataFrame, filepath: str, format: str = 'parquet')`

Save data to file.

```python
from src.utils.data_storage import save_data

save_data(market_data, "data/market_data.parquet", format="parquet")
```

**Parameters**:
- `data` (pd.DataFrame): Data to save
- `filepath` (str): File path
- `format` (str): File format ("parquet", "csv", "json")

#### `load_data(filepath: str, format: str = None) -> pd.DataFrame`

Load data from file.

```python
from src.utils.data_storage import load_data

data = load_data("data/market_data.parquet")
```

**Parameters**:
- `filepath` (str): File path
- `format` (str, optional): File format (auto-detected if None)

**Returns**: Loaded DataFrame

## Configuration APIs

### Settings

#### `get_settings() -> Settings`

Get application settings.

```python
from src.config.settings import get_settings

settings = get_settings()
```

**Returns**: Settings object with all configuration

#### Settings Properties

```python
# Environment detection
settings.is_production      # Boolean
settings.is_development     # Boolean

# AWS credentials
settings.has_aws_credentials  # Boolean

# Email configuration
settings.has_email_config     # Boolean

# Slack configuration
settings.has_slack_config     # Boolean
```

#### Environment Variables

All settings can be configured via environment variables:

```bash
# Application
export APP_NAME="market-master"
export APP_VERSION="1.0.0"
export DEBUG="false"
export LOG_LEVEL="INFO"

# Database
export DATABASE_URL="postgresql://user:pass@localhost:5432/market_master"
export REDIS_URL="redis://localhost:6379"

# MLflow
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="market_master_ai"

# AWS
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
export AWS_S3_BUCKET="market-master-mlflow-artifacts"

# Monitoring
export EVIDENTLY_SERVICE_URL="http://localhost:8080"
export GRAFANA_URL="http://localhost:3000"
export PROMETHEUS_URL="http://localhost:9090"

# Model Configuration
export MODEL_ACCURACY_THRESHOLD="0.6"
export MODEL_F1_THRESHOLD="0.5"
export RETRAINING_INTERVAL_DAYS="7"

# Trading Configuration
export GAME_DURATION_SECONDS="300"
export TICK_INTERVAL_SECONDS="1"
export MAX_POSITION_SIZE="2.0"
export RISK_FREE_RATE="0.02"
```

## Web Application APIs

### Streamlit Application

The main web application provides a user interface for Market Master.

#### Access Points

- **Web App**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

#### Main Functions

```python
# Generate sample data
def generate_sample_data(n_samples=1000, instrument="AAPL") -> pd.DataFrame

# Generate technical indicators
def generate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame

# Generate predictions
def generate_predictions(data: pd.DataFrame) -> Tuple[List[str], List[float]]

# Create price chart
def create_price_chart(data: pd.DataFrame, title: str) -> go.Figure

# Create technical indicators chart
def create_technical_indicators_chart(data: pd.DataFrame) -> go.Figure

# Display prediction results
def display_prediction_results(predictions: List[str], confidence_scores: List[float], key_suffix: str = "")
```

## Error Handling

### Common Exceptions

#### `ValueError`

Raised when input data is invalid.

```python
try:
    features = model.prepare_features(data)
except ValueError as e:
    print(f"Invalid data: {e}")
```

#### `FileNotFoundError`

Raised when model file is not found.

```python
try:
    model.load_model("nonexistent.pkl")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
```

#### `MLflowException`

Raised when MLflow operations fail.

```python
try:
    model_uri = registry.register_model(model, "test_model", metrics)
except MLflowException as e:
    print(f"MLflow error: {e}")
```

### Error Response Format

```json
{
    "error": {
        "type": "ValueError",
        "message": "Missing required columns: ['open', 'high', 'low', 'close', 'volume']",
        "details": {
            "missing_columns": ["open", "high", "low", "close", "volume"],
            "available_columns": ["timestamp", "price", "volume"]
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Authentication

### API Keys

For production deployments, API keys may be required:

```python
import requests

headers = {
    "Authorization": "Bearer your_api_key_here",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://api.market-master.com/predict",
    headers=headers,
    json={"data": market_data}
)
```

### JWT Tokens

For user authentication:

```python
import jwt

# Create token
token = jwt.encode(
    {"user_id": 123, "exp": datetime.utcnow() + timedelta(hours=24)},
    settings.jwt_secret_key,
    algorithm="HS256"
)

# Verify token
try:
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
    user_id = payload["user_id"]
except jwt.ExpiredSignatureError:
    print("Token has expired")
except jwt.InvalidTokenError:
    print("Invalid token")
```

## Rate Limiting

### Request Limits

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1000 requests/hour
- **Enterprise**: Custom limits

### Rate Limit Headers

```python
# Check rate limit headers
response = requests.get("https://api.market-master.com/models")
print(f"Remaining requests: {response.headers.get('X-RateLimit-Remaining')}")
print(f"Reset time: {response.headers.get('X-RateLimit-Reset')}")
```

### Rate Limit Response

```json
{
    "error": {
        "type": "RateLimitExceeded",
        "message": "Rate limit exceeded. Please try again later.",
        "details": {
            "limit": 100,
            "remaining": 0,
            "reset_time": "2024-01-15T11:30:00Z"
        }
    }
}
```

## Usage Examples

### Complete Workflow

```python
from src.main import MarketMasterApp
from src.data.asset_classes import AssetClass
from src.models.action_predictor import ActionPredictor
from src.mlops.model_registry import ModelRegistry

# Initialize application
app = MarketMasterApp()

# Run complete demo
results = app.run_demo(
    asset_class="crypto",
    instrument="BTC/USD",
    n_samples=10000
)

# Create model
model = ActionPredictor()

# Generate data
from src.data.data_generator import MarketDataGenerator
generator = MarketDataGenerator(AssetClass.CRYPTO, "BTC/USD")
data = generator.generate_tick_data(n_ticks=1000)

# Prepare features
features = model.prepare_features(data)

# Make predictions
predictions = model.predict_with_confidence(features.iloc[-1:])

# Register model
registry = ModelRegistry()
model_uri = registry.register_model(
    model=model,
    model_name="btc_predictor_v1",
    metrics={"accuracy": 0.69, "f1_score": 0.72}
)

print(f"Model registered: {model_uri}")
print(f"Prediction: {predictions[0]['action']} (confidence: {predictions[0]['confidence']:.2%})")
```

### Batch Processing

```python
import pandas as pd
from src.models.action_predictor import ActionPredictor
from src.utils.helpers import calculate_technical_indicators

# Load multiple datasets
datasets = {
    "BTC/USD": pd.read_parquet("data/btc_usd.parquet"),
    "ETH/USD": pd.read_parquet("data/eth_usd.parquet"),
    "AAPL": pd.read_parquet("data/aapl.parquet")
}

# Initialize model
model = ActionPredictor(model_path="models/action_predictor_v1.pkl")

# Process all datasets
results = {}
for instrument, data in datasets.items():
    # Calculate indicators
    indicators = calculate_technical_indicators(data)
    
    # Prepare features
    features = model.prepare_features(indicators)
    
    # Make predictions
    predictions = model.predict_with_confidence(features.iloc[-1:])
    
    results[instrument] = {
        "action": predictions[0]["action"],
        "confidence": predictions[0]["confidence"],
        "probabilities": predictions[0]["probabilities"]
    }

# Display results
for instrument, result in results.items():
    print(f"{instrument}: {result['action']} (confidence: {result['confidence']:.2%})")
```

---

For more detailed information about specific implementations, see the source code in the `src/` directory. 