"""
Simplified monitoring module for Market Master.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import logging

# Use basic logging instead of the logger that might have import issues
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Base class for model monitoring."""
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize the monitor.
        
        Args:
            reference_data: Reference dataset for comparison
        """
        self.reference_data = reference_data
        self.monitoring_history = []
        
    def generate_report(self, current_data: pd.DataFrame, report_type: str) -> Dict[str, Any]:
        """
        Generate monitoring report.
        
        Args:
            current_data: Current dataset
            report_type: Type of report to generate
            
        Returns:
            Monitoring report as dictionary
        """
        raise NotImplementedError


class DataQualityMonitor(ModelMonitor):
    """Monitor data quality metrics."""
    
    def generate_report(self, current_data: pd.DataFrame, report_type: str = "data_quality") -> Dict[str, Any]:
        """Generate data quality report."""
        if report_type == "data_quality":
            return self._generate_data_quality_report(current_data)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def _generate_data_quality_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report."""
        # Calculate basic quality metrics
        missing_values = current_data.isnull().sum().to_dict()
        missing_ratio = current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns))
        duplicates = current_data.duplicated().sum()
        data_types = current_data.dtypes.astype(str).to_dict()
        
        # Calculate quality score
        quality_score = 1.0 - missing_ratio - (duplicates / len(current_data)) * 0.1
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'data_quality_score': quality_score,
            'missing_values': missing_values,
            'missing_ratio': missing_ratio,
            'duplicates': duplicates,
            'data_types': data_types,
            'total_rows': len(current_data),
            'total_columns': len(current_data.columns)
        }
    
    def check_data_quality(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.
        
        Args:
            current_data: Current dataset
            
        Returns:
            Data quality metrics
        """
        report = self.generate_report(current_data, "data_quality")
        self.monitoring_history.append(report)
        return report


class DataDriftMonitor(ModelMonitor):
    """Monitor data drift."""
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.05):
        super().__init__(reference_data)
        self.drift_threshold = drift_threshold
    
    def generate_report(self, current_data: pd.DataFrame, report_type: str = "data_drift") -> Dict[str, Any]:
        """Generate data drift report."""
        if report_type == "data_drift":
            return self._generate_data_drift_report(current_data)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def _generate_data_drift_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data drift report."""
        drift_metrics = {}
        column_drift = {}
        
        # Calculate drift for numerical columns
        numerical_cols = current_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in self.reference_data.columns:
                ref_mean = self.reference_data[col].mean()
                ref_std = self.reference_data[col].std()
                curr_mean = current_data[col].mean()
                curr_std = current_data[col].std()
                
                # Calculate drift using statistical distance
                if ref_std > 0:
                    drift_score = abs(curr_mean - ref_mean) / ref_std
                else:
                    drift_score = 0.0
                
                column_drift[col] = drift_score
        
        # Calculate overall dataset drift
        if column_drift:
            dataset_drift = np.mean(list(column_drift.values()))
        else:
            dataset_drift = 0.0
        
        drift_detected = dataset_drift > self.drift_threshold
        
        return {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift': dataset_drift,
            'column_drift': column_drift,
            'drift_detected': drift_detected,
            'drift_threshold': self.drift_threshold
        }
    
    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for data drift.
        
        Args:
            current_data: Current dataset
            
        Returns:
            Drift detection results
        """
        report = self.generate_report(current_data, "data_drift")
        self.monitoring_history.append(report)
        return report


class PerformanceMonitor(ModelMonitor):
    """Monitor model performance."""
    
    def __init__(self, reference_data: pd.DataFrame, target_column: str = 'action'):
        super().__init__(reference_data)
        self.target_column = target_column
    
    def generate_report(self, current_data: pd.DataFrame, report_type: str = "classification_performance") -> Dict[str, Any]:
        """Generate performance report."""
        if report_type == "classification_performance":
            return self._generate_performance_report(current_data)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def _generate_performance_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance report."""
        # Basic performance metrics calculation
        if self.target_column in current_data.columns:
            target_values = current_data[self.target_column]
            if 'prediction' in current_data.columns:
                predictions = current_data['prediction']
                
                # Calculate accuracy
                accuracy = (target_values == predictions).mean()
                
                # Calculate precision, recall, F1 for binary classification
                if len(target_values.unique()) == 2:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    try:
                        precision = precision_score(target_values, predictions, average='weighted')
                        recall = recall_score(target_values, predictions, average='weighted')
                        f1 = f1_score(target_values, predictions, average='weighted')
                    except:
                        precision = recall = f1 = 0.0
                else:
                    precision = recall = f1 = 0.0
            else:
                accuracy = precision = recall = f1 = 0.0
        else:
            accuracy = precision = recall = f1 = 0.0
        
        # Calculate target drift
        if self.target_column in current_data.columns and self.target_column in self.reference_data.columns:
            ref_dist = self.reference_data[self.target_column].value_counts(normalize=True)
            curr_dist = current_data[self.target_column].value_counts(normalize=True)
            
            # Calculate distribution difference
            common_values = set(ref_dist.index) & set(curr_dist.index)
            if common_values:
                drift_score = sum(abs(ref_dist.get(val, 0) - curr_dist.get(val, 0)) for val in common_values)
            else:
                drift_score = 1.0
        else:
            drift_score = 0.0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'target_drift': drift_score
        }
    
    def check_performance(self, current_data: pd.DataFrame, predictions: np.ndarray = None) -> Dict[str, Any]:
        """
        Check model performance.
        
        Args:
            current_data: Current dataset
            predictions: Model predictions (optional)
            
        Returns:
            Performance metrics
        """
        # Add predictions if provided
        if predictions is not None:
            current_data = current_data.copy()
            current_data['prediction'] = predictions
        
        report = self.generate_report(current_data, "classification_performance")
        self.monitoring_history.append(report)
        return report


class ComprehensiveMonitor:
    """Comprehensive monitoring system."""
    
    def __init__(self, reference_data: pd.DataFrame, target_column: str = 'action'):
        """
        Initialize comprehensive monitor.
        
        Args:
            reference_data: Reference dataset
            target_column: Target column name
        """
        self.data_quality_monitor = DataQualityMonitor(reference_data)
        self.data_drift_monitor = DataDriftMonitor(reference_data)
        self.performance_monitor = PerformanceMonitor(reference_data, target_column)
        
        self.monitoring_results = {
            'data_quality': [],
            'data_drift': [],
            'performance': []
        }
        
        logger.info("Comprehensive monitor initialized")
    
    def run_monitoring(self, current_data: pd.DataFrame, predictions: np.ndarray = None) -> Dict[str, Any]:
        """
        Run comprehensive monitoring.
        
        Args:
            current_data: Current dataset
            predictions: Model predictions (optional)
            
        Returns:
            Comprehensive monitoring results
        """
        logger.info("Running comprehensive monitoring")
        
        # Run all monitors
        quality_results = self.data_quality_monitor.check_data_quality(current_data)
        drift_results = self.data_drift_monitor.check_data_drift(current_data)
        performance_results = self.performance_monitor.check_performance(current_data, predictions)
        
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
        
        logger.info("Monitoring completed", summary=summary)
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
    
    def export_monitoring_report(self, filepath: str):
        """Export monitoring results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.monitoring_results, f, indent=2, default=str)
        
        logger.info(f"Monitoring report exported to {filepath}")
    
    def generate_alert(self, threshold: float = 0.8) -> List[str]:
        """
        Generate alerts based on monitoring results.
        
        Args:
            threshold: Alert threshold
            
        Returns:
            List of alerts
        """
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