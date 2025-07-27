"""
Simplified monitoring module for Market Master.
Doesn't depend on Evidently to avoid import issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SimpleModelMonitor:
    """Simplified model monitoring without Evidently dependencies."""
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize the monitor.
        
        Args:
            reference_data: Reference dataset for comparison
        """
        self.reference_data = reference_data
        self.monitoring_history = []
        
    def calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        
        # Check for duplicates
        duplicate_ratio = data.duplicated().sum() / len(data)
        
        # Check for outliers (simple approach)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_ratio = 0
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_ratio += outliers / len(data)
            outlier_ratio /= len(numeric_cols)
        
        # Calculate quality score
        quality_score = 1.0 - (missing_ratio + duplicate_ratio + outlier_ratio)
        return max(0.0, min(1.0, quality_score))
    
    def calculate_data_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data drift scores."""
        drift_scores = {}
        
        # Compare basic statistics
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.reference_data.columns:
                ref_mean = self.reference_data[col].mean()
                ref_std = self.reference_data[col].std()
                curr_mean = current_data[col].mean()
                curr_std = current_data[col].std()
                
                # Calculate drift as relative difference
                mean_drift = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-8)
                std_drift = abs(curr_std - ref_std) / (abs(ref_std) + 1e-8)
                
                drift_scores[col] = (mean_drift + std_drift) / 2
        
        return drift_scores
    
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics


class SimpleDataQualityMonitor(SimpleModelMonitor):
    """Simplified data quality monitor."""
    
    def check_data_quality(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality metrics."""
        quality_score = self.calculate_data_quality_score(current_data)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'data_quality_score': quality_score,
            'missing_values': current_data.isnull().sum().to_dict(),
            'duplicates': current_data.duplicated().sum(),
            'data_shape': current_data.shape
        }
        
        self.monitoring_history.append(metrics)
        return metrics


class SimpleDataDriftMonitor(SimpleModelMonitor):
    """Simplified data drift monitor."""
    
    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift."""
        drift_scores = self.calculate_data_drift(current_data)
        
        # Calculate overall drift
        overall_drift = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        
        drift_metrics = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift': overall_drift,
            'column_drift': drift_scores,
            'drift_detected': overall_drift > 0.1  # Threshold of 10%
        }
        
        self.monitoring_history.append(drift_metrics)
        return drift_metrics


class SimplePerformanceMonitor(SimpleModelMonitor):
    """Simplified performance monitor."""
    
    def check_performance(self, current_data: pd.DataFrame, predictions: np.ndarray = None, 
                         y_true: np.ndarray = None) -> Dict[str, Any]:
        """Check model performance."""
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        # If we have true labels and predictions, calculate metrics
        if y_true is not None and predictions is not None:
            metrics = self.calculate_performance_metrics(y_true, predictions)
            performance_metrics.update(metrics)
        
        self.monitoring_history.append(performance_metrics)
        return performance_metrics


class SimpleComprehensiveMonitor:
    """Simplified comprehensive monitoring system."""
    
    def __init__(self, reference_data: pd.DataFrame, target_column: str = 'action'):
        """Initialize comprehensive monitor."""
        self.data_quality_monitor = SimpleDataQualityMonitor(reference_data)
        self.data_drift_monitor = SimpleDataDriftMonitor(reference_data)
        self.performance_monitor = SimplePerformanceMonitor(reference_data)
        
        self.monitoring_results = {
            'data_quality': [],
            'data_drift': [],
            'performance': []
        }
        
        logger.info("Simple comprehensive monitor initialized")
    
    def run_monitoring(self, current_data: pd.DataFrame, predictions: np.ndarray = None,
                      y_true: np.ndarray = None) -> Dict[str, Any]:
        """Run comprehensive monitoring."""
        logger.info("Running simple comprehensive monitoring")
        
        # Run all monitors
        quality_results = self.data_quality_monitor.check_data_quality(current_data)
        drift_results = self.data_drift_monitor.check_data_drift(current_data)
        performance_results = self.performance_monitor.check_performance(current_data, predictions, y_true)
        
        # Store results
        self.monitoring_results['data_quality'].append(quality_results)
        self.monitoring_results['data_drift'].append(drift_results)
        self.monitoring_results['performance'].append(performance_results)
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_quality_score': quality_results['data_quality_score'],
            'drift_detected': drift_results['drift_detected'],
            'performance_metrics': {
                'accuracy': performance_results['accuracy'],
                'f1_score': performance_results['f1_score']
            },
            'overall_status': self._calculate_overall_status(quality_results, drift_results, performance_results)
        }
        
        logger.info("Simple monitoring completed", summary=summary)
        return summary
    
    def _calculate_overall_status(self, quality_results: Dict, drift_results: Dict, 
                                performance_results: Dict) -> str:
        """Calculate overall system status."""
        # Check data quality
        if quality_results['data_quality_score'] < 0.8:
            return "WARNING"
        
        # Check for drift
        if drift_results['drift_detected']:
            return "DRIFT_DETECTED"
        
        # Check performance
        if performance_results['accuracy'] < 0.6:
            return "PERFORMANCE_DEGRADED"
        
        return "HEALTHY"
    
    def get_monitoring_history(self) -> Dict[str, List]:
        """Get monitoring history."""
        return self.monitoring_results
    
    def generate_alert(self, threshold: float = 0.8) -> List[str]:
        """Generate alerts based on monitoring results."""
        alerts = []
        
        # Check latest results
        if self.monitoring_results['data_quality']:
            latest_quality = self.monitoring_results['data_quality'][-1]
            if latest_quality['data_quality_score'] < threshold:
                alerts.append(f"Data quality degraded: {latest_quality['data_quality_score']:.3f}")
        
        if self.monitoring_results['data_drift']:
            latest_drift = self.monitoring_results['data_drift'][-1]
            if latest_drift['drift_detected']:
                alerts.append(f"Data drift detected: {latest_drift['dataset_drift']:.3f}")
        
        if self.monitoring_results['performance']:
            latest_performance = self.monitoring_results['performance'][-1]
            if latest_performance['accuracy'] < threshold:
                alerts.append(f"Performance degraded: {latest_performance['accuracy']:.3f}")
        
        return alerts 