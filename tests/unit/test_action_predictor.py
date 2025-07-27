"""
Unit tests for ActionPredictor model
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.models import ActionPredictor
from src.data import AssetClass, generate_training_data
from sklearn.model_selection import train_test_split


class TestActionPredictor(unittest.TestCase):
    """Test cases for ActionPredictor model"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = ActionPredictor(use_mlflow=False)
        
        # Generate small test dataset
        self.test_data = generate_training_data(AssetClass.EQUITY, "TEST", 200)
        self.y = self.test_data['action'].copy()
        self.X = self.test_data.drop(['action', 'asset_class', 'instrument', 'session', 
                                     'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        # Split for training
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_model_initialization(self):
        """Test ActionPredictor initialization"""
        model = ActionPredictor(use_mlflow=False)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.scaler)
        self.assertEqual(len(model.feature_names), 25)  # Expected number of features
        self.assertEqual(len(model.classes), 5)  # buy, sell, hold, strong_buy, strong_sell

    def test_feature_preparation(self):
        """Test feature preparation functionality"""
        # Remove metadata columns to simulate raw OHLCV data
        raw_features = self.X_train.copy()
        
        prepared_features = self.model.prepare_features(raw_features)
        
        self.assertIsInstance(prepared_features, pd.DataFrame)
        self.assertGreater(len(prepared_features.columns), 0)
        self.assertFalse(prepared_features.isnull().all().any())  # No completely empty columns

    def test_model_training(self):
        """Test model training functionality"""
        # Train the model
        metrics = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        
        # Check required metrics exist
        required_metrics = ['train_accuracy', 'val_accuracy', 'train_f1', 'val_f1']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)

    def test_model_prediction(self):
        """Test model prediction functionality"""
        # Train the model first
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Test prediction
        predictions = self.model.predict(self.X_val)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_val))
        
        # Check predictions are valid actions
        valid_actions = set(self.model.classes)
        for prediction in predictions:
            self.assertIn(prediction, valid_actions)

    def test_model_prediction_probabilities(self):
        """Test model prediction probabilities"""
        # Train the model first
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Test prediction probabilities
        probabilities = self.model.predict_proba(self.X_val)
        
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape[0], len(self.X_val))
        self.assertEqual(probabilities.shape[1], len(self.model.classes))
        
        # Check probabilities sum to 1
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

    def test_prediction_with_confidence(self):
        """Test prediction with confidence scores"""
        # Train the model first
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Test prediction with confidence
        results = self.model.predict_with_confidence(self.X_val.head(5))
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)
        
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertIn('probabilities', result)
            
            # Check confidence is between 0 and 1
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)

    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Train the model first
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # Check all importance values are non-negative
        for feature, score in importance.items():
            self.assertGreaterEqual(score, 0)

    def test_model_persistence(self):
        """Test model save and load functionality"""
        # Train the model first
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.joblib')
            
            # Save model
            self.model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            new_model = ActionPredictor(use_mlflow=False)
            load_success = new_model.load_model(model_path)
            
            self.assertTrue(load_success)
            
            # Test that loaded model gives same predictions
            original_predictions = self.model.predict(self.X_val.head(10))
            loaded_predictions = new_model.predict(self.X_val.head(10))
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        # Train the model first
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Evaluate model
        eval_results = self.model.evaluate(self.X_val, self.y_val)
        
        self.assertIsInstance(eval_results, dict)
        self.assertIn('metrics', eval_results)
        self.assertIn('per_class_metrics', eval_results)
        self.assertIn('predictions', eval_results)
        
        # Check metrics are reasonable
        metrics = eval_results['metrics']
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)

    def test_model_config_customization(self):
        """Test custom model configuration"""
        custom_config = {
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = ActionPredictor(use_mlflow=False)
        model.model_config = custom_config
        model._initialize_model()
        
        # Check that config was applied
        self.assertEqual(model.model.n_estimators, 10)
        self.assertEqual(model.model.max_depth, 3)
        self.assertEqual(model.model.random_state, 42)


if __name__ == '__main__':
    unittest.main() 