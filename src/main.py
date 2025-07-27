"""
Main application entry point for Market Master.
"""

import sys
import os
import argparse
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from utils.logger import get_logger, setup_logging
from data import generate_training_data, AssetClass, get_asset_config
from models import ActionPredictor, train_action_predictor, evaluate_model
from mlops import ModelRegistry, ComprehensiveMonitor
from llm import MockTradingCoach, TradingPersona, get_trading_advice

logger = get_logger(__name__)


class MarketMasterApp:
    """Main Market Master application."""
    
    def __init__(self):
        """Initialize the Market Master application."""
        self.settings = get_settings()
        setup_logging(self.settings.log_level)
        
        self.registry = ModelRegistry()
        self.coach = MockTradingCoach(TradingPersona.MODERATE)
        self.monitor = None
        
        logger.info("Market Master application initialized")
    
    def run_demo(self, asset_class: str = "equity", instrument: str = "AAPL", 
                n_samples: int = 10000) -> Dict[str, Any]:
        """
        Run complete Market Master demo.
        
        Args:
            asset_class: Asset class for demo
            instrument: Instrument for demo
            n_samples: Number of samples to generate
            
        Returns:
            Demo results
        """
        logger.info("Starting Market Master demo", 
                   asset_class=asset_class, 
                   instrument=instrument, 
                   n_samples=n_samples)
        
        results = {
            'demo_start_time': datetime.now().isoformat(),
            'asset_class': asset_class,
            'instrument': instrument,
            'n_samples': n_samples,
            'steps': {}
        }
        
        try:
            # Step 1: Generate training data
            logger.info("Step 1: Generating training data")
            asset_enum = AssetClass(asset_class)
            training_data = generate_training_data(asset_enum, instrument, n_samples)
            
            results['steps']['data_generation'] = {
                'status': 'success',
                'data_shape': training_data.shape,
                'features_count': len(training_data.columns) - 3,  # Exclude action, asset_class, instrument
                'label_distribution': training_data['action'].value_counts().to_dict()
            }
            
            # Step 2: Train model
            logger.info("Step 2: Training Action Predictor model")
            train_data, test_data = self._split_data(training_data, 0.2)
            
            X_train = train_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
            y_train = train_data['action']
            X_test = test_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
            y_test = test_data['action']
            
            model, train_metrics = train_action_predictor(X_train, y_train, X_test, y_test)
            
            results['steps']['model_training'] = {
                'status': 'success',
                'training_metrics': train_metrics,
                'model_config': model.model_config
            }
            
            # Step 3: Evaluate model
            logger.info("Step 3: Evaluating model performance")
            eval_results = evaluate_model(model, X_test, y_test)
            
            results['steps']['model_evaluation'] = {
                'status': 'success',
                'evaluation_metrics': eval_results['metrics'],
                'per_class_metrics': eval_results['per_class_metrics']
            }
            
            # Step 4: Register model
            logger.info("Step 4: Registering model in MLflow")
            model_name = f"action_predictor_{asset_class}_{instrument}"
            model_uri = self.registry.register_model(
                model.model,
                model_name,
                train_metrics,
                parameters=model.model_config,
                tags={
                    'asset_class': asset_class,
                    'instrument': instrument,
                    'demo_run': 'true'
                }
            )
            
            results['steps']['model_registration'] = {
                'status': 'success',
                'model_uri': model_uri,
                'model_name': model_name
            }
            
            # Step 5: Setup monitoring
            logger.info("Step 5: Setting up model monitoring")
            self.monitor = ComprehensiveMonitor(train_data)
            monitoring_results = self.monitor.run_monitoring(test_data, eval_results['predictions'])
            
            results['steps']['monitoring_setup'] = {
                'status': 'success',
                'monitoring_results': monitoring_results
            }
            
            # Step 6: Generate trading advice
            logger.info("Step 6: Generating trading advice")
            # Use last few rows for real-time simulation
            recent_data = test_data.tail(10)
            predictions = model.predict(recent_data)
            confidences = np.max(model.predict_proba(recent_data), axis=1)
            
            trading_advice = []
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                advice = self.coach.get_trading_advice(
                    pred, conf, recent_data.iloc[i:i+1]
                )
                trading_advice.append(advice)
            
            results['steps']['trading_advice'] = {
                'status': 'success',
                'advice_count': len(trading_advice),
                'sample_advice': trading_advice[0] if trading_advice else None
            }
            
            # Step 7: Save model locally
            logger.info("Step 7: Saving model locally")
            model_path = f"models/action_predictor_{asset_class}_{instrument}.joblib"
            os.makedirs("models", exist_ok=True)
            model.save_model(model_path)
            
            results['steps']['model_saving'] = {
                'status': 'success',
                'model_path': model_path
            }
            
            # Overall results
            results['demo_end_time'] = datetime.now().isoformat()
            results['overall_status'] = 'success'
            results['summary'] = {
                'model_accuracy': eval_results['metrics']['accuracy'],
                'model_f1_score': eval_results['metrics']['f1_score'],
                'data_quality_score': monitoring_results['data_quality_score'],
                'drift_detected': monitoring_results['drift_detected'],
                'overall_status': monitoring_results['overall_status']
            }
            
            logger.info("Market Master demo completed successfully", 
                       accuracy=results['summary']['model_accuracy'],
                       f1_score=results['summary']['model_f1_score'])
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            results['overall_status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def run_inference(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run inference with a trained model.
        
        Args:
            model_name: Name of the model to use
            data: Input data for inference
            
        Returns:
            Inference results
        """
        logger.info(f"Running inference with model: {model_name}")
        
        try:
            # Load model
            model = self.registry.load_model(model_name)
            
            # Prepare features
            features = data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
            
            # Make predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            confidences = np.max(probabilities, axis=1)
            
            # Generate trading advice
            trading_advice = []
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                advice = self.coach.get_trading_advice(
                    pred, conf, data.iloc[i:i+1]
                )
                trading_advice.append(advice)
            
            results = {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidences': confidences,
                'trading_advice': trading_advice,
                'prediction_distribution': pd.Series(predictions).value_counts().to_dict(),
                'avg_confidence': np.mean(confidences),
                'inference_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Inference completed successfully", 
                       avg_confidence=results['avg_confidence'],
                       prediction_count=len(predictions))
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        try:
            models = self.registry.list_models()
            return {
                'models': models,
                'count': len(models)
            }
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {'models': [], 'count': 0}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            versions = self.registry.list_model_versions(model_name)
            return {
                'model_name': model_name,
                'versions': versions,
                'version_count': len(versions)
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'model_name': model_name, 'versions': [], 'version_count': 0}
    
    def _split_data(self, data: pd.DataFrame, test_size: float) -> tuple:
        """Split data into train and test sets."""
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        return train_data, test_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Market Master - AI-Powered Trading Assistant")
    parser.add_argument("--demo", action="store_true", help="Run complete demo")
    parser.add_argument("--asset-class", default="equity", choices=["equity", "commodity", "forex", "crypto", "indices"],
                       help="Asset class for demo")
    parser.add_argument("--instrument", default="AAPL", help="Instrument for demo")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model-info", help="Get info about specific model")
    parser.add_argument("--inference", help="Run inference with model name")
    
    args = parser.parse_args()
    
    app = MarketMasterApp()
    
    if args.demo:
        print("🚀 Starting Market Master Demo...")
        results = app.run_demo(args.asset_class, args.instrument, args.samples)
        
        if results['overall_status'] == 'success':
            print("✅ Demo completed successfully!")
            print(f"📊 Model Accuracy: {results['summary']['model_accuracy']:.3f}")
            print(f"🎯 F1 Score: {results['summary']['model_f1_score']:.3f}")
            print(f"📈 Data Quality: {results['summary']['data_quality_score']:.3f}")
            print(f"🔍 Overall Status: {results['summary']['overall_status']}")
        else:
            print("❌ Demo failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
    
    elif args.list_models:
        print("📋 Available Models:")
        models_info = app.list_models()
        for model in models_info['models']:
            print(f"  - {model['name']}")
    
    elif args.model_info:
        print(f"📊 Model Info for: {args.model_info}")
        model_info = app.get_model_info(args.model_info)
        print(f"Versions: {model_info['version_count']}")
        for version in model_info['versions']:
            print(f"  - Version {version['version']}: {version['stage']}")
    
    elif args.inference:
        print(f"🔮 Running inference with model: {args.inference}")
        # Generate some test data for inference
        asset_enum = AssetClass("equity")
        test_data = generate_training_data(asset_enum, "AAPL", 100)
        results = app.run_inference(args.inference, test_data)
        print(f"Predictions: {results['prediction_distribution']}")
        print(f"Average Confidence: {results['avg_confidence']:.3f}")
    
    else:
        print("🎯 Market Master - AI-Powered Trading Assistant")
        print("Use --help for available options")
        print("Try --demo to run the complete demonstration")


if __name__ == "__main__":
    main() 