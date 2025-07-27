# Model Documentation

## Overview

Market Master implements a sophisticated machine learning system for financial market prediction, featuring multiple model types, automated training pipelines, and production-ready deployment capabilities.

## Model Architecture

### Core Models

#### 1. Action Predictor (`ActionPredictor`)

**Purpose**: Predicts optimal trading actions based on technical indicators

**Model Type**: Random Forest Classifier with ensemble methods

**Target Classes**:
- `buy`: Moderate buy signal
- `sell`: Moderate sell signal  
- `hold`: No action recommended
- `strong_buy`: Strong buy signal
- `strong_sell`: Strong sell signal

**Features (25 Technical Indicators)**:
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

**Model Configuration**:
```python
model_config = {
    'n_estimators': 100,      # Number of trees
    'max_depth': 10,          # Maximum tree depth
    'min_samples_split': 5,   # Minimum samples to split
    'min_samples_leaf': 2,    # Minimum samples per leaf
    'random_state': 42        # Reproducibility
}
```

### Model Factory (`ModelFactory`)

**Purpose**: Centralized model creation and configuration management

**Available Configurations**:
- `default`: Balanced performance and speed
- `fast`: Optimized for speed with reduced complexity
- `accurate`: Maximum accuracy with increased complexity

**Usage Example**:
```python
from src.models.model_factory import ModelFactory

factory = ModelFactory()
model = factory.create_model('action_predictor', config_name='accurate')
```

## Data Pipeline

### Market Data Generator (`MarketDataGenerator`)

**Purpose**: Generates realistic market data for training and testing

**Supported Asset Classes**:
- **Cryptocurrency**: Bitcoin, Ethereum, altcoins (24/7 trading)
- **Forex**: Major currency pairs (24/5 trading)
- **Equity**: Stocks, ETFs (market hours)
- **Commodity**: Gold, oil, agricultural products
- **Index**: S&P 500, NASDAQ, DAX

**Data Generation Features**:
- Realistic price movements with volatility regimes
- Volume patterns matching market behavior
- Session-aware data (market hours, after-hours)
- Technical indicator calculation
- Label generation for supervised learning

### Technical Indicators

**Price Action Indicators**:
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **MACD**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility and trend indicator
- **Moving Averages**: SMA (20, 50), EMA (12, 26)

**Volume Indicators**:
- **OBV (On-Balance Volume)**: Volume-price relationship
- **VWAP (Volume Weighted Average Price)**: Intraday price reference

**Momentum Indicators**:
- **Stochastic**: Momentum oscillator
- **Williams %R**: Overbought/oversold indicator
- **CCI (Commodity Channel Index)**: Cyclical indicator

**Trend Indicators**:
- **ADX (Average Directional Index)**: Trend strength
- **SuperTrend**: Trend-following indicator
- **ATR (Average True Range)**: Volatility measure

## Model Training

### Training Process

1. **Data Preparation**:
   ```python
   # Generate training data
   generator = MarketDataGenerator(AssetClass.CRYPTO, 'BTC/USD')
   data = generator.generate_training_data(n_samples=10000)
   
   # Prepare features
   features = model.prepare_features(data)
   ```

2. **Model Training**:
   ```python
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       features, data['action'], test_size=0.2, random_state=42
   )
   
   # Train model
   model.train(X_train, y_train, X_val, y_val)
   ```

3. **Evaluation**:
   ```python
   # Evaluate performance
   metrics = model.evaluate(X_test, y_test)
   print(f"Accuracy: {metrics['accuracy']:.2%}")
   print(f"F1-Score: {metrics['f1_score']:.3f}")
   ```

### Performance Metrics

**Standard Metrics**:
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

**Financial Metrics**:
- **Confidence Score**: Probability-based prediction confidence
- **Risk Assessment**: Volatility-adjusted predictions
- **Sharpe Ratio**: Risk-adjusted returns (if backtesting)

### Model Validation

**Cross-Validation**:
```python
# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

**Feature Importance**:
```python
importance = model.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.3f}")
```

## Model Deployment

### MLflow Integration

**Model Registration**:
```python
# Log model with MLflow
with mlflow.start_run():
    mlflow.log_params(model.model_config)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "action_predictor")
```

**Model Loading**:
```python
# Load model from registry
model_uri = "models:/action_predictor/Production"
loaded_model = mlflow.sklearn.load_model(model_uri)
```

### Production Deployment

**Docker Container**:
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
COPY models/ /app/models/
CMD ["python", "-m", "src.main"]
```

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: market-master-models
spec:
  replicas: 3
  selector:
    matchLabels:
      app: market-master
  template:
    metadata:
      labels:
        app: market-master
    spec:
      containers:
      - name: model-server
        image: market-master:latest
        ports:
        - containerPort: 8080
```

## Model Monitoring

### Performance Monitoring

**Real-time Metrics**:
- Prediction accuracy over time
- Feature drift detection
- Model confidence distribution
- Inference latency

**Alerting**:
- Accuracy degradation (>5% drop)
- High latency (>1 second)
- Data quality issues
- Model drift detection

### Drift Detection

**Data Drift**:
```python
# Monitor feature distributions
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
```

**Model Drift**:
```python
# Monitor prediction distributions
from evidently.metric_preset import TargetDriftPreset

report = Report(metrics=[TargetDriftPreset()])
report.run(reference_data=reference, current_data=current)
```

## Usage Examples

### Basic Prediction

```python
from src.models.action_predictor import ActionPredictor

# Initialize model
model = ActionPredictor()

# Generate sample data
from src.data.data_generator import MarketDataGenerator
from src.data.asset_classes import AssetClass

generator = MarketDataGenerator(AssetClass.CRYPTO, 'BTC/USD')
data = generator.generate_tick_data(n_ticks=1000)

# Prepare features
features = model.prepare_features(data)

# Make prediction
prediction = model.predict(features.iloc[-1:])
confidence = model.predict_with_confidence(features.iloc[-1:])

print(f"Action: {prediction[0]}")
print(f"Confidence: {confidence[0]['confidence']:.2%}")
```

### Batch Prediction

```python
# Predict for multiple time periods
predictions = model.predict(features)
probabilities = model.predict_proba(features)

# Get detailed predictions with confidence
detailed_predictions = model.predict_with_confidence(features)

for i, pred in enumerate(detailed_predictions):
    print(f"Time {i}: {pred['action']} (confidence: {pred['confidence']:.2%})")
```

### Model Training Pipeline

```python
from src.models import train_action_predictor
from src.data.data_generator import generate_training_data
from src.data.asset_classes import AssetClass

# Generate training data
data = generate_training_data(AssetClass.CRYPTO, 'BTC/USD', n_samples=10000)

# Prepare features and labels
features = model.prepare_features(data)
labels = data['action']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train model
trained_model, metrics = train_action_predictor(
    X_train, y_train, X_test, y_test, use_mlflow=True
)

print(f"Training completed. Accuracy: {metrics['accuracy']:.2%}")
```

## Model Configuration

### Hyperparameter Tuning

**Grid Search**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Model Persistence

**Save Model**:
```python
# Save to local file
model.save_model('models/action_predictor_v1.pkl')

# Save with MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "action_predictor")
```

**Load Model**:
```python
# Load from local file
model = ActionPredictor()
model.load_model('models/action_predictor_v1.pkl')

# Load from MLflow
model_uri = "models:/action_predictor/Production"
model = mlflow.sklearn.load_model(model_uri)
```

## Troubleshooting

### Common Issues

**1. Missing Features**:
```
Error: Missing required columns: ['open', 'high', 'low', 'close', 'volume']
Solution: Ensure input data contains OHLCV columns
```

**2. Model Loading Errors**:
```
Error: Model file not found
Solution: Check model path and ensure file exists
```

**3. Prediction Errors**:
```
Error: Input features don't match training features
Solution: Use model.prepare_features() to ensure consistent feature engineering
```

### Performance Optimization

**1. Reduce Training Time**:
- Use `fast` configuration in ModelFactory
- Reduce `n_estimators` and `max_depth`
- Use smaller training datasets

**2. Improve Accuracy**:
- Use `accurate` configuration
- Increase training data size
- Add more technical indicators
- Implement ensemble methods

**3. Reduce Inference Latency**:
- Use model quantization
- Implement caching
- Optimize feature calculation

## Best Practices

### Data Quality
- Validate input data before processing
- Handle missing values appropriately
- Normalize features consistently
- Monitor data drift regularly

### Model Management
- Version control all model artifacts
- Document model configurations
- Implement A/B testing for new models
- Maintain model lineage

### Production Deployment
- Use containerization for consistency
- Implement health checks
- Set up monitoring and alerting
- Plan for model rollback scenarios

### Security
- Validate all inputs
- Implement rate limiting
- Secure model artifacts
- Monitor for adversarial attacks

## API Reference

### ActionPredictor Methods

#### `__init__(model_path=None, use_mlflow=True)`
Initialize the Action Predictor model.

**Parameters**:
- `model_path` (str, optional): Path to pre-trained model
- `use_mlflow` (bool): Whether to use MLflow tracking

#### `prepare_features(data: pd.DataFrame) -> pd.DataFrame`
Prepare features from raw market data.

**Parameters**:
- `data` (pd.DataFrame): Raw market data with OHLCV columns

**Returns**:
- `pd.DataFrame`: Engineered features

#### `train(X_train, y_train, X_val, y_val) -> Dict[str, float]`
Train the model on provided data.

**Parameters**:
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training labels
- `X_val` (pd.DataFrame): Validation features
- `y_val` (pd.Series): Validation labels

**Returns**:
- `Dict[str, float]`: Training metrics

#### `predict(features: pd.DataFrame) -> np.ndarray`
Make predictions on input features.

**Parameters**:
- `features` (pd.DataFrame): Input features

**Returns**:
- `np.ndarray`: Predicted actions

#### `predict_with_confidence(features: pd.DataFrame) -> List[Dict]`
Make predictions with confidence scores.

**Parameters**:
- `features` (pd.DataFrame): Input features

**Returns**:
- `List[Dict]`: Predictions with confidence and probabilities

### ModelFactory Methods

#### `create_model(model_type: str, config_name: str = 'default', **kwargs) -> Any`
Create a model instance with specified configuration.

**Parameters**:
- `model_type` (str): Type of model to create
- `config_name` (str): Configuration preset name
- `**kwargs`: Additional model parameters

**Returns**:
- Model instance

#### `get_available_models() -> List[str]`
Get list of available model types.

**Returns**:
- `List[str]`: Available model types

#### `get_available_configs(model_type: str) -> List[str]`
Get available configurations for a model type.

**Parameters**:
- `model_type` (str): Model type

**Returns**:
- `List[str]`: Available configurations

---

For more information about specific model implementations, see the source code in `src/models/` directory. 