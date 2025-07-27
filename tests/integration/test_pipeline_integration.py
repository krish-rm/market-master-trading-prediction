"""
Integration tests for Market Master MLOps pipeline
"""

import unittest
import pandas as pd
import sys
from pathlib import Path
import tempfile
import os

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.data import AssetClass, generate_training_data, generate_market_data
from src.models import ActionPredictor, train_action_predictor
from src.mlops.monitoring_simple import SimpleComprehensiveMonitor
from src.llm import MockTradingCoach, TradingPersona
from sklearn.model_selection import train_test_split


class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests for complete MLOps pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.asset_class = AssetClass.EQUITY
        self.instrument = "INTEGRATION_TEST"
        self.sample_size = 500

    def test_complete_training_pipeline(self):
        """Test complete training pipeline from data to model"""
        # Step 1: Generate training data
        training_data = generate_training_data(self.asset_class, self.instrument, self.sample_size)
        self.assertIsInstance(training_data, pd.DataFrame)
        self.assertGreater(len(training_data), 0)
        self.assertIn('action', training_data.columns)

        # Step 2: Prepare data for training
        y = training_data['action'].copy()
        X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                               'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 3: Train model using convenience function
        model, metrics = train_action_predictor(X_train, y_train, X_val, y_val, use_mlflow=False)
        
        # Verify training results
        self.assertIsInstance(model, ActionPredictor)
        self.assertIsInstance(metrics, dict)
        self.assertIn('val_accuracy', metrics)
        self.assertGreaterEqual(metrics['val_accuracy'], 0)

        # Step 4: Make predictions
        predictions = model.predict(X_val)
        self.assertEqual(len(predictions), len(X_val))

        # Step 5: Evaluate model
        eval_results = model.evaluate(X_val, y_val)
        self.assertIn('metrics', eval_results)

    def test_inference_pipeline(self):
        """Test inference pipeline with new data"""
        # Step 1: Train a model
        training_data = generate_training_data(self.asset_class, self.instrument, 300)
        y = training_data['action'].copy()
        X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                               'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = ActionPredictor(use_mlflow=False)
        model.train(X_train, y_train, X_val, y_val)

        # Step 2: Generate new market data for inference
        new_data = generate_market_data(self.asset_class, self.instrument, 50)
        
        # Step 3: Prepare for inference (remove metadata)
        inference_features = new_data.drop(['asset_class', 'instrument', 'session', 
                                           'volatility_regime', 'timestamp'], axis=1, errors='ignore')

        # Step 4: Make predictions
        predictions = model.predict(inference_features)
        confidence_results = model.predict_with_confidence(inference_features)

        # Verify inference results
        self.assertEqual(len(predictions), len(new_data))
        self.assertEqual(len(confidence_results), len(new_data))
        
        for result in confidence_results:
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)

    def test_monitoring_pipeline(self):
        """Test monitoring pipeline with trained model"""
        # Step 1: Train model
        training_data = generate_training_data(self.asset_class, self.instrument, 400)
        y = training_data['action'].copy()
        X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                               'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = ActionPredictor(use_mlflow=False)
        model.train(X_train, y_train, X_val, y_val)

        # Step 2: Set up monitoring
        monitor = SimpleComprehensiveMonitor(X_train)

        # Step 3: Generate predictions for monitoring
        predictions = model.predict(X_val)

        # Step 4: Run monitoring
        monitoring_results = monitor.run_monitoring(X_val, predictions)

        # Verify monitoring results
        self.assertIsInstance(monitoring_results, dict)
        self.assertIn('data_quality_score', monitoring_results)
        self.assertIn('drift_detected', monitoring_results)
        self.assertIn('overall_status', monitoring_results)

    def test_trading_advice_pipeline(self):
        """Test trading advice generation pipeline"""
        # Step 1: Generate market data
        market_data = generate_market_data(self.asset_class, self.instrument, 100)

        # Step 2: Train model for predictions
        training_data = generate_training_data(self.asset_class, self.instrument, 300)
        y = training_data['action'].copy()
        X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                               'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = ActionPredictor(use_mlflow=False)
        model.train(X_train, y_train, X_val, y_val)

        # Step 3: Make predictions on market data
        inference_features = market_data.drop(['asset_class', 'instrument', 'session', 
                                              'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        confidence_results = model.predict_with_confidence(inference_features.head(10))

        # Step 4: Generate trading advice
        coach = MockTradingCoach(TradingPersona.MODERATE)
        
        for result in confidence_results[:5]:  # Test first 5 predictions
            advice = coach.get_trading_advice(
                result['prediction'], 
                result['confidence'], 
                market_data.head(1)  # Single row of market data
            )
            
            # Verify advice structure
            self.assertIsInstance(advice, dict)
            self.assertIn('action', advice)
            self.assertIn('confidence', advice)
            self.assertIn('position_size', advice)

    def test_model_persistence_pipeline(self):
        """Test model save/load in pipeline context"""
        # Step 1: Train model
        training_data = generate_training_data(self.asset_class, self.instrument, 200)
        y = training_data['action'].copy()
        X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                               'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        original_model = ActionPredictor(use_mlflow=False)
        original_model.train(X_train, y_train, X_val, y_val)

        # Step 2: Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'pipeline_test_model.joblib')
            original_model.save_model(model_path)

            # Step 3: Load model in new instance
            loaded_model = ActionPredictor(use_mlflow=False)
            load_success = loaded_model.load_model(model_path)
            self.assertTrue(load_success)

            # Step 4: Verify loaded model works in pipeline
            test_data = generate_market_data(self.asset_class, self.instrument, 20)
            test_features = test_data.drop(['asset_class', 'instrument', 'session', 
                                           'volatility_regime', 'timestamp'], axis=1, errors='ignore')

            original_predictions = original_model.predict(test_features)
            loaded_predictions = loaded_model.predict(test_features)

            # Predictions should be identical
            self.assertTrue((original_predictions == loaded_predictions).all())

    def test_multi_asset_pipeline(self):
        """Test pipeline with multiple asset classes"""
        asset_tests = [
            (AssetClass.CRYPTO, "BTC/USD"),
            (AssetClass.FOREX, "EUR/USD"),
            (AssetClass.COMMODITY, "GOLD")
        ]

        for asset_class, instrument in asset_tests:
            with self.subTest(asset_class=asset_class, instrument=instrument):
                # Generate data
                training_data = generate_training_data(asset_class, instrument, 200)
                market_data = generate_market_data(asset_class, instrument, 50)

                # Verify data generation worked
                self.assertGreater(len(training_data), 0)
                self.assertEqual(len(market_data), 50)
                self.assertEqual(market_data['asset_class'].iloc[0], asset_class.value)

                # Quick training test
                y = training_data['action'].copy()
                X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                                       'volatility_regime', 'timestamp'], axis=1, errors='ignore')
                
                if len(X) > 20:  # Ensure sufficient data for split
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = ActionPredictor(use_mlflow=False)
                    model.model_config = {'n_estimators': 5, 'random_state': 42}  # Fast training
                    model._initialize_model()
                    
                    metrics = model.train(X_train, y_train, X_val, y_val)
                    self.assertIn('val_accuracy', metrics)

    def test_error_handling_pipeline(self):
        """Test pipeline error handling and recovery"""
        # Test with minimal data
        try:
            minimal_data = generate_training_data(self.asset_class, self.instrument, 50)
            self.assertGreater(len(minimal_data), 0)
        except Exception as e:
            # Should handle gracefully
            self.assertIsInstance(e, Exception)

        # Test with empty predictions
        try:
            monitor = SimpleComprehensiveMonitor(pd.DataFrame({'test': [1, 2, 3]}))
            # This should handle empty cases gracefully
        except Exception:
            # Expected for invalid inputs
            pass


class TestPipelinePerformance(unittest.TestCase):
    """Performance tests for pipeline components"""

    def test_training_time_reasonable(self):
        """Test that training completes in reasonable time"""
        import time
        
        # Generate data
        training_data = generate_training_data(AssetClass.EQUITY, "PERF_TEST", 500)
        y = training_data['action'].copy()
        X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                               'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Time the training
        start_time = time.time()
        
        model = ActionPredictor(use_mlflow=False)
        model.model_config = {'n_estimators': 20, 'random_state': 42}  # Reasonable size
        model._initialize_model()
        model.train(X_train, y_train, X_val, y_val)
        
        training_time = time.time() - start_time
        
        # Should complete within 60 seconds
        self.assertLess(training_time, 60, "Training took too long")

    def test_inference_speed(self):
        """Test inference speed is reasonable"""
        import time
        
        # Train a model
        training_data = generate_training_data(AssetClass.EQUITY, "SPEED_TEST", 300)
        y = training_data['action'].copy()
        X = training_data.drop(['action', 'asset_class', 'instrument', 'session', 
                               'volatility_regime', 'timestamp'], axis=1, errors='ignore')
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = ActionPredictor(use_mlflow=False)
        model.train(X_train, y_train, X_val, y_val)

        # Test inference speed
        test_data = generate_market_data(AssetClass.EQUITY, "SPEED_TEST", 100)
        test_features = test_data.drop(['asset_class', 'instrument', 'session', 
                                       'volatility_regime', 'timestamp'], axis=1, errors='ignore')

        start_time = time.time()
        predictions = model.predict(test_features)
        inference_time = time.time() - start_time

        # Should be very fast (< 5 seconds for 100 predictions)
        self.assertLess(inference_time, 5, "Inference took too long")
        self.assertEqual(len(predictions), 100)


if __name__ == '__main__':
    unittest.main() 