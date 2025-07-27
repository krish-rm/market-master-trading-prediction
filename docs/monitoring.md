# Monitoring Guide

## Overview

Market Master implements a comprehensive monitoring system that tracks model performance, data quality, and system health in real-time. This guide covers all monitoring components, setup, configuration, and best practices.

## Table of Contents

- [Monitoring Architecture](#monitoring-architecture)
- [Data Quality Monitoring](#data-quality-monitoring)
- [Data Drift Detection](#data-drift-detection)
- [Model Performance Monitoring](#model-performance-monitoring)
- [System Health Monitoring](#system-health-monitoring)
- [Alerting & Notifications](#alerting--notifications)
- [Dashboard Setup](#dashboard-setup)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Monitoring Architecture

### Overview

Market Master uses a multi-layered monitoring approach:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Monitoring     │    │   Alerting      │
│                 │    │  Components     │    │   & Dashboards  │
│ • Model Inputs  │───▶│ • Data Quality  │───▶│ • Grafana       │
│ • Predictions   │    │ • Data Drift    │    │ • Prometheus    │
│ • Performance   │    │ • Performance   │    │ • AlertManager  │
│ • System Metrics│    │ • System Health │    │ • Email/Slack   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Monitoring Stack

- **Data Quality**: Custom quality metrics and validation
- **Data Drift**: Statistical drift detection
- **Model Performance**: Accuracy, latency, and business metrics
- **System Health**: Infrastructure and application metrics
- **Visualization**: Grafana dashboards
- **Metrics Storage**: Prometheus
- **Alerting**: AlertManager with email/Slack integration

## Data Quality Monitoring

### Overview

Data quality monitoring ensures that input data meets quality standards before model inference.

### Quality Metrics

#### 1. Missing Values
```python
# Calculate missing value ratio
missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
```

#### 2. Duplicate Detection
```python
# Count duplicate rows
duplicates = data.duplicated().sum()
duplicate_ratio = duplicates / len(data)
```

#### 3. Data Type Validation
```python
# Validate expected data types
expected_types = {
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64'
}

for col, expected_type in expected_types.items():
    if data[col].dtype != expected_type:
        raise ValueError(f"Column {col} has wrong type: {data[col].dtype}")
```

#### 4. Outlier Detection
```python
# IQR-based outlier detection
def detect_outliers(data: pd.DataFrame, column: str) -> float:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = ((data[column] < (Q1 - 1.5 * IQR)) | 
                (data[column] > (Q3 + 1.5 * IQR))).sum()
    
    return outliers / len(data)
```

### Quality Score Calculation

```python
def calculate_quality_score(data: pd.DataFrame) -> float:
    """Calculate overall data quality score (0-1)."""
    
    # Missing value penalty
    missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    
    # Duplicate penalty
    duplicate_ratio = data.duplicated().sum() / len(data)
    
    # Outlier penalty
    outlier_ratio = 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        outlier_ratio += detect_outliers(data, col)
    outlier_ratio /= len(numeric_cols)
    
    # Calculate quality score
    quality_score = 1.0 - (missing_ratio + duplicate_ratio + outlier_ratio)
    return max(0.0, min(1.0, quality_score))
```

### Usage Example

```python
from src.mlops.monitoring import DataQualityMonitor

# Initialize monitor with reference data
monitor = DataQualityMonitor(reference_data)

# Check data quality
quality_report = monitor.check_data_quality(current_data)

print(f"Quality Score: {quality_report['data_quality_score']:.2%}")
print(f"Missing Values: {quality_report['missing_ratio']:.2%}")
print(f"Duplicates: {quality_report['duplicates']}")
```

## Data Drift Detection

### Overview

Data drift detection identifies when the distribution of input data changes significantly from the training data distribution.

### Drift Detection Methods

#### 1. Statistical Drift Detection

```python
def calculate_statistical_drift(reference_data: pd.DataFrame, 
                              current_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate statistical drift for numeric columns."""
    
    drift_scores = {}
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in reference_data.columns:
            # Calculate mean and std drift
            ref_mean = reference_data[col].mean()
            ref_std = reference_data[col].std()
            curr_mean = current_data[col].mean()
            curr_std = current_data[col].std()
            
            # Relative difference
            mean_drift = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-8)
            std_drift = abs(curr_std - ref_std) / (abs(ref_std) + 1e-8)
            
            drift_scores[col] = (mean_drift + std_drift) / 2
    
    return drift_scores
```

#### 2. Distribution Drift Detection

```python
from scipy import stats

def calculate_distribution_drift(reference_data: pd.DataFrame,
                               current_data: pd.DataFrame,
                               column: str) -> float:
    """Calculate distribution drift using Kolmogorov-Smirnov test."""
    
    ref_values = reference_data[column].dropna()
    curr_values = current_data[column].dropna()
    
    # Perform KS test
    ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
    
    # Convert to drift score (0-1)
    drift_score = 1 - p_value
    return drift_score
```

#### 3. Feature Drift Detection

```python
def detect_feature_drift(reference_data: pd.DataFrame,
                        current_data: pd.DataFrame,
                        threshold: float = 0.05) -> Dict[str, Any]:
    """Detect drift in individual features."""
    
    drift_results = {
        'drift_detected': False,
        'drift_features': [],
        'drift_scores': {}
    }
    
    for col in current_data.columns:
        if col in reference_data.columns:
            drift_score = calculate_distribution_drift(
                reference_data, current_data, col
            )
            
            drift_results['drift_scores'][col] = drift_score
            
            if drift_score > threshold:
                drift_results['drift_detected'] = True
                drift_results['drift_features'].append(col)
    
    return drift_results
```

### Usage Example

```python
from src.mlops.monitoring import DataDriftMonitor

# Initialize drift monitor
drift_monitor = DataDriftMonitor(
    reference_data=training_data,
    drift_threshold=0.05
)

# Check for drift
drift_report = drift_monitor.check_data_drift(current_data)

if drift_report['drift_detected']:
    print("⚠️ Data drift detected!")
    print(f"Drift features: {drift_report['drift_features']}")
else:
    print("✅ No significant drift detected")
```

## Model Performance Monitoring

### Overview

Model performance monitoring tracks prediction accuracy, latency, and business metrics over time.

### Performance Metrics

#### 1. Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_classification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate classification performance metrics."""
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
```

#### 2. Business Metrics

```python
def calculate_business_metrics(predictions: np.ndarray,
                             actual_returns: np.ndarray,
                             confidence_scores: np.ndarray) -> Dict[str, float]:
    """Calculate business-relevant metrics."""
    
    # Filter high-confidence predictions
    high_conf_mask = confidence_scores > 0.8
    filtered_predictions = predictions[high_conf_mask]
    filtered_returns = actual_returns[high_conf_mask]
    
    # Calculate metrics
    metrics = {
        'total_predictions': len(predictions),
        'high_confidence_predictions': len(filtered_predictions),
        'average_confidence': np.mean(confidence_scores),
        'predicted_return': np.mean(filtered_returns),
        'prediction_volume': len(filtered_predictions) / len(predictions)
    }
    
    return metrics
```

#### 3. Latency Monitoring

```python
import time
from functools import wraps

def monitor_latency(func):
    """Decorator to monitor function latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Log latency metric
        logger.info(f"Function {func.__name__} latency: {latency:.3f}s")
        
        return result
    return wrapper

# Usage
@monitor_latency
def predict_action(features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)
```

### Performance Tracking

```python
class PerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, reference_metrics: Dict[str, float]):
        self.reference_metrics = reference_metrics
        self.performance_history = []
    
    def check_performance(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if performance has degraded."""
        
        degradation = {}
        for metric, ref_value in self.reference_metrics.items():
            if metric in current_metrics:
                curr_value = current_metrics[metric]
                degradation[metric] = (ref_value - curr_value) / ref_value
        
        # Determine overall status
        avg_degradation = np.mean(list(degradation.values()))
        
        if avg_degradation > 0.1:  # 10% degradation
            status = "degraded"
        elif avg_degradation > 0.05:  # 5% degradation
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'status': status,
            'degradation': degradation,
            'average_degradation': avg_degradation,
            'current_metrics': current_metrics,
            'timestamp': datetime.now().isoformat()
        }
```

### Usage Example

```python
from src.mlops.monitoring import PerformanceMonitor

# Initialize performance monitor
performance_monitor = PerformanceMonitor(
    reference_data=training_data,
    target_column='action'
)

# Check performance
performance_report = performance_monitor.check_performance(
    current_data=test_data,
    predictions=model_predictions
)

print(f"Performance Status: {performance_report['status']}")
print(f"Average Degradation: {performance_report['average_degradation']:.2%}")
```

## System Health Monitoring

### Overview

System health monitoring tracks infrastructure metrics, application health, and resource utilization.

### Infrastructure Metrics

#### 1. Resource Utilization

```python
import psutil

def get_system_metrics() -> Dict[str, float]:
    """Get system resource utilization metrics."""
    
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'network_io': psutil.net_io_counters(),
        'load_average': psutil.getloadavg()
    }
```

#### 2. Application Health

```python
def check_application_health() -> Dict[str, Any]:
    """Check application health status."""
    
    health_checks = {
        'database_connection': check_database_connection(),
        'mlflow_connection': check_mlflow_connection(),
        'model_loading': check_model_loading(),
        'prediction_service': check_prediction_service()
    }
    
    overall_status = 'healthy' if all(health_checks.values()) else 'unhealthy'
    
    return {
        'status': overall_status,
        'checks': health_checks,
        'timestamp': datetime.now().isoformat()
    }
```

#### 3. Service Dependencies

```python
import requests

def check_service_health(service_url: str, timeout: int = 5) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(f"{service_url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

# Check all services
services = {
    'mlflow': 'http://localhost:5000',
    'grafana': 'http://localhost:3000',
    'prometheus': 'http://localhost:9090'
}

service_health = {
    service: check_service_health(url) 
    for service, url in services.items()
}
```

## Alerting & Notifications

### Overview

Market Master supports multiple alerting channels including email, Slack, and webhooks.

### Alert Configuration

#### 1. Alert Rules

```python
class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, settings):
        self.settings = settings
        self.alert_history = []
    
    def check_alerts(self, monitoring_results: Dict[str, Any]) -> List[str]:
        """Check if any alerts should be triggered."""
        
        alerts = []
        
        # Data quality alerts
        if monitoring_results.get('data_quality_score', 1.0) < 0.8:
            alerts.append("Data quality below threshold (80%)")
        
        # Drift alerts
        if monitoring_results.get('drift_detected', False):
            alerts.append("Data drift detected")
        
        # Performance alerts
        if monitoring_results.get('performance_status') == 'degraded':
            alerts.append("Model performance degraded")
        
        # System health alerts
        if monitoring_results.get('system_status') != 'healthy':
            alerts.append("System health issues detected")
        
        return alerts
    
    def send_alert(self, alert_message: str, alert_type: str = 'warning'):
        """Send alert through configured channels."""
        
        # Log alert
        logger.warning(f"Alert [{alert_type}]: {alert_message}")
        
        # Send email alert
        if self.settings.has_email_config:
            self._send_email_alert(alert_message, alert_type)
        
        # Send Slack alert
        if self.settings.has_slack_config:
            self._send_slack_alert(alert_message, alert_type)
        
        # Store in history
        self.alert_history.append({
            'message': alert_message,
            'type': alert_type,
            'timestamp': datetime.now().isoformat()
        })
```

#### 2. Email Alerts

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(subject: str, message: str, settings):
    """Send email alert."""
    
    msg = MIMEMultipart()
    msg['From'] = settings.smtp_username
    msg['To'] = settings.alert_email
    msg['Subject'] = f"[Market Master Alert] {subject}"
    
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        server = smtplib.SMTP(settings.smtp_host, settings.smtp_port)
        server.starttls()
        server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(msg)
        server.quit()
        logger.info("Email alert sent successfully")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
```

#### 3. Slack Alerts

```python
import requests
import json

def send_slack_alert(message: str, webhook_url: str, channel: str = "#alerts"):
    """Send Slack alert."""
    
    payload = {
        "channel": channel,
        "text": f"🚨 Market Master Alert: {message}",
        "icon_emoji": ":warning:"
    }
    
    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        logger.info("Slack alert sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")
```

### Alert Thresholds

```python
# Default alert thresholds
ALERT_THRESHOLDS = {
    'data_quality_score': 0.8,
    'data_drift_threshold': 0.05,
    'performance_degradation': 0.1,
    'system_cpu_percent': 80.0,
    'system_memory_percent': 85.0,
    'prediction_latency_ms': 1000
}
```

## Dashboard Setup

### Grafana Configuration

#### 1. Data Sources

```yaml
# grafana/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

#### 2. Dashboard Panels

```json
{
  "dashboard": {
    "title": "Market Master Monitoring",
    "panels": [
      {
        "title": "Data Quality Score",
        "type": "stat",
        "targets": [
          {
            "expr": "market_master_data_quality_score",
            "legendFormat": "Quality Score"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "market_master_model_accuracy",
            "legendFormat": "Accuracy"
          },
          {
            "expr": "market_master_model_f1_score",
            "legendFormat": "F1 Score"
          }
        ]
      },
      {
        "title": "Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(market_master_prediction_duration_seconds[5m])",
            "legendFormat": "Latency"
          }
        ]
      }
    ]
  }
}
```

#### 3. Alert Rules

```yaml
# prometheus/alert_rules.yml
groups:
  - name: market_master_alerts
    rules:
      - alert: DataQualityDegraded
        expr: market_master_data_quality_score < 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data quality score is below threshold"
          
      - alert: ModelPerformanceDegraded
        expr: market_master_model_accuracy < 0.6
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model performance has degraded"
          
      - alert: HighLatency
        expr: rate(market_master_prediction_duration_seconds[5m]) > 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Prediction latency is high"
```

### Prometheus Metrics

#### 1. Custom Metrics

```python
from prometheus_client import Counter, Gauge, Histogram

# Define metrics
PREDICTION_COUNTER = Counter(
    'market_master_predictions_total',
    'Total number of predictions',
    ['model_name', 'asset_class']
)

PREDICTION_DURATION = Histogram(
    'market_master_prediction_duration_seconds',
    'Prediction duration in seconds',
    ['model_name']
)

DATA_QUALITY_SCORE = Gauge(
    'market_master_data_quality_score',
    'Data quality score (0-1)',
    ['data_source']
)

MODEL_ACCURACY = Gauge(
    'market_master_model_accuracy',
    'Model accuracy score',
    ['model_name']
)
```

#### 2. Metrics Collection

```python
def collect_metrics(predictions: np.ndarray, 
                   duration: float,
                   model_name: str,
                   asset_class: str):
    """Collect and record metrics."""
    
    # Record prediction count
    PREDICTION_COUNTER.labels(
        model_name=model_name,
        asset_class=asset_class
    ).inc(len(predictions))
    
    # Record prediction duration
    PREDICTION_DURATION.labels(model_name=model_name).observe(duration)
    
    # Record data quality score
    quality_score = calculate_quality_score(input_data)
    DATA_QUALITY_SCORE.labels(data_source='market_data').set(quality_score)
    
    # Record model accuracy
    accuracy = calculate_accuracy(predictions, actual_values)
    MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
```

## Configuration

### Environment Variables

```bash
# Monitoring Configuration
export EVIDENTLY_SERVICE_URL="http://localhost:8080"
export GRAFANA_URL="http://localhost:3000"
export PROMETHEUS_URL="http://localhost:9090"

# Alert Configuration
export ALERT_EMAIL="alerts@company.com"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/xxx/yyy/zzz"

# Threshold Configuration
export DATA_QUALITY_THRESHOLD="0.8"
export DRIFT_THRESHOLD="0.05"
export PERFORMANCE_THRESHOLD="0.6"
export LATENCY_THRESHOLD="1000"

# Monitoring Intervals
export MONITORING_INTERVAL_SECONDS="300"
export ALERT_CHECK_INTERVAL_SECONDS="60"
```

### Settings Configuration

```python
from src.config.settings import get_settings

settings = get_settings()

# Monitoring URLs
print(f"Evidently: {settings.evidently_service_url}")
print(f"Grafana: {settings.grafana_url}")
print(f"Prometheus: {settings.prometheus_url}")

# Alert configuration
if settings.has_email_config:
    print("Email alerts configured")
if settings.has_slack_config:
    print("Slack alerts configured")
```

### Monitoring Setup

```python
from src.mlops.monitoring import ComprehensiveMonitor

# Initialize comprehensive monitor
monitor = ComprehensiveMonitor(
    reference_data=training_data,
    target_column='action'
)

# Run monitoring
monitoring_results = monitor.run_monitoring(
    current_data=live_data,
    predictions=model_predictions
)

# Check for alerts
alerts = monitor.generate_alert(threshold=0.8)

if alerts:
    print("⚠️ Alerts generated:")
    for alert in alerts:
        print(f"  - {alert}")
```

## Best Practices

### 1. Monitoring Strategy

- **Start Simple**: Begin with basic quality and performance metrics
- **Gradual Expansion**: Add drift detection and business metrics over time
- **Focus on Business Impact**: Monitor metrics that directly affect trading decisions
- **Automate Everything**: Set up automated monitoring and alerting

### 2. Alert Management

- **Avoid Alert Fatigue**: Set appropriate thresholds and cooldown periods
- **Escalation Procedures**: Define clear escalation paths for critical alerts
- **Alert Documentation**: Document each alert type and resolution procedures
- **Regular Review**: Periodically review and adjust alert thresholds

### 3. Performance Optimization

- **Efficient Metrics Collection**: Use sampling for high-volume metrics
- **Storage Management**: Implement retention policies for historical data
- **Resource Monitoring**: Monitor monitoring system resource usage
- **Caching**: Cache frequently accessed metrics and dashboards

### 4. Data Quality

- **Proactive Validation**: Validate data before it reaches the model
- **Quality Gates**: Implement quality gates in data pipelines
- **Root Cause Analysis**: Investigate and fix quality issues promptly
- **Quality Metrics**: Track quality metrics over time

### 5. Drift Detection

- **Multiple Methods**: Use statistical and ML-based drift detection
- **Feature Importance**: Focus drift detection on important features
- **Contextual Analysis**: Consider market conditions when interpreting drift
- **Retraining Triggers**: Automate retraining based on drift detection

## Troubleshooting

### Common Issues

#### 1. High False Positive Alerts

**Symptoms**: Too many alerts for minor issues

**Solutions**:
- Adjust alert thresholds based on historical data
- Implement alert cooldown periods
- Use different severity levels for different issues
- Review and tune alert rules regularly

#### 2. Missing Metrics

**Symptoms**: Metrics not appearing in dashboards

**Solutions**:
- Check metric collection code is running
- Verify Prometheus configuration
- Check network connectivity between services
- Review metric naming conventions

#### 3. High Latency in Monitoring

**Symptoms**: Slow dashboard updates or delayed alerts

**Solutions**:
- Optimize metric collection frequency
- Use metric sampling for high-volume data
- Implement metric caching
- Scale monitoring infrastructure

#### 4. Data Quality Issues

**Symptoms**: Poor quality scores or validation failures

**Solutions**:
- Investigate data source issues
- Check data pipeline transformations
- Validate data format and types
- Implement data quality checks earlier in pipeline

### Debugging Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3000/api/health

# Check metric values
curl "http://localhost:9090/api/v1/query?query=market_master_data_quality_score"

# Check alert manager
curl http://localhost:9093/api/v1/alerts

# Check service logs
docker logs market-master-prometheus
docker logs market-master-grafana
docker logs market-master-alertmanager
```

### Performance Tuning

```python
# Optimize metric collection
import time
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_metrics():
    """Cache expensive metric calculations."""
    return calculate_expensive_metrics()

# Batch metric updates
def batch_update_metrics(metrics_batch: List[Dict]):
    """Update multiple metrics in batch."""
    for metric in metrics_batch:
        update_single_metric(metric)
```

---

For more detailed information about specific monitoring implementations, see the source code in `src/mlops/monitoring.py` and `src/mlops/monitoring_simple.py`. 