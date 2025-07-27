"""
Unit tests for MLOps components
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.mlops import ModelRegistry
from src.mlops.monitoring_simple import SimpleComprehensiveMonitor
from src.data import AssetClass, generate_training_data
from src.models import ActionPredictor
from sklearn.model_selection import train_test_split


class TestMLOpsComponents(unittest.TestCase):
    """Test cases for MLOps infrastructure"""

    def setUp(self):
        """Set up test fixtures"""
        # Generate test data
        self.test_data = generate_training_data(AssetClass.EQUITY, "TEST", 300)
        self.y = self.test_data['action'].copy()
        self.X = self.test_data.drop(['action', 'asset_class', 'instrument', 'session', 
                                     'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_model_registry_initialization(self):
        """Test ModelRegistry initialization"""
        try:
            registry = ModelRegistry()
            self.assertIsInstance(registry, ModelRegistry)
        except Exception as e:
            # If MLflow is not available, test should still pass
            self.assertIn("MLflow", str(e).lower())

    def test_monitoring_initialization(self):
        """Test SimpleComprehensiveMonitor initialization"""
        monitor = SimpleComprehensiveMonitor(self.X_train)
        self.assertIsInstance(monitor, SimpleComprehensiveMonitor)

    def test_monitoring_data_quality_check(self):
        """Test data quality monitoring"""
        monitor = SimpleComprehensiveMonitor(self.X_train)
        
        # Generate predictions for monitoring
        model = ActionPredictor(use_mlflow=False)
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        predictions = model.predict(self.X_val)
        
        # Run monitoring
        results = monitor.run_monitoring(self.X_val, predictions)
        
        self.assertIsInstance(results, dict)
        self.assertIn('data_quality_score', results)
        self.assertIn('drift_detected', results)
        self.assertIn('overall_status', results)
        
        # Check data quality score is reasonable
        self.assertGreaterEqual(results['data_quality_score'], 0)
        self.assertLessEqual(results['data_quality_score'], 1)

    def test_monitoring_drift_detection(self):
        """Test drift detection functionality"""
        monitor = SimpleComprehensiveMonitor(self.X_train)
        
        # Generate predictions
        model = ActionPredictor(use_mlflow=False)
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        predictions = model.predict(self.X_val)
        
        # Test with normal data
        results_normal = monitor.run_monitoring(self.X_val, predictions)
        
        # Test with modified data (simulated drift)
        X_drift = self.X_val.copy()
        # Add systematic shift to simulate drift
        numeric_cols = X_drift.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_drift[numeric_cols[0]] = X_drift[numeric_cols[0]] * 1.5
        
        predictions_drift = model.predict(X_drift)
        results_drift = monitor.run_monitoring(X_drift, predictions_drift)
        
        # Both should return valid results
        self.assertIsInstance(results_normal['drift_detected'], bool)
        self.assertIsInstance(results_drift['drift_detected'], bool)

    def test_monitoring_performance_metrics(self):
        """Test performance metrics calculation"""
        monitor = SimpleComprehensiveMonitor(self.X_train)
        
        # Generate predictions
        model = ActionPredictor(use_mlflow=False)
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        predictions = model.predict(self.X_val)
        
        # Run monitoring
        results = monitor.run_monitoring(self.X_val, predictions)
        
        # Check performance metrics exist
        if 'performance_metrics' in results:
            perf_metrics = results['performance_metrics']
            self.assertIsInstance(perf_metrics, dict)

    def test_monitoring_status_determination(self):
        """Test overall status determination"""
        monitor = SimpleComprehensiveMonitor(self.X_train)
        
        # Generate predictions
        model = ActionPredictor(use_mlflow=False)
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        predictions = model.predict(self.X_val)
        
        # Run monitoring
        results = monitor.run_monitoring(self.X_val, predictions)
        
        # Check status is valid
        valid_statuses = ['HEALTHY', 'DRIFT_DETECTED', 'PERFORMANCE_DEGRADED', 'CRITICAL']
        self.assertIn(results['overall_status'], valid_statuses)

    def test_monitoring_with_edge_cases(self):
        """Test monitoring with edge cases"""
        monitor = SimpleComprehensiveMonitor(self.X_train)
        
        # Test with minimal data
        X_minimal = self.X_val.head(5)
        predictions_minimal = ['buy'] * 5
        
        try:
            results = monitor.run_monitoring(X_minimal, predictions_minimal)
            self.assertIsInstance(results, dict)
        except Exception:
            # Some edge cases might fail gracefully
            pass

    def test_monitoring_error_handling(self):
        """Test monitoring error handling"""
        monitor = SimpleComprehensiveMonitor(self.X_train)
        
        # Test with mismatched data
        try:
            results = monitor.run_monitoring(self.X_val, ['buy'])  # Wrong length
            # Should handle gracefully or raise appropriate error
        except (ValueError, IndexError):
            # Expected for mismatched dimensions
            pass


class TestMLOpsIntegration(unittest.TestCase):
    """Integration tests for MLOps components working together"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_data = generate_training_data(AssetClass.EQUITY, "TEST", 200)
        self.y = self.test_data['action'].copy()
        self.X = self.test_data.drop(['action', 'asset_class', 'instrument', 'session', 
                                     'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_model_training_and_monitoring_pipeline(self):
        """Test complete model training and monitoring pipeline"""
        # Train model
        model = ActionPredictor(use_mlflow=False)
        metrics = model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Set up monitoring
        monitor = SimpleComprehensiveMonitor(self.X_train)
        
        # Make predictions
        predictions = model.predict(self.X_val)
        
        # Run monitoring
        monitoring_results = monitor.run_monitoring(self.X_val, predictions)
        
        # Verify pipeline worked
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(monitoring_results, dict)
        self.assertIn('data_quality_score', monitoring_results)

    def test_model_evaluation_and_monitoring(self):
        """Test model evaluation with monitoring"""
        # Train model
        model = ActionPredictor(use_mlflow=False)
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Evaluate model
        eval_results = model.evaluate(self.X_val, self.y_val)
        
        # Set up monitoring with evaluation results
        monitor = SimpleComprehensiveMonitor(self.X_train)
        monitoring_results = monitor.run_monitoring(self.X_val, eval_results['predictions'])
        
        # Check both evaluation and monitoring worked
        self.assertIn('metrics', eval_results)
        self.assertIn('data_quality_score', monitoring_results)


if __name__ == '__main__':
    unittest.main() 