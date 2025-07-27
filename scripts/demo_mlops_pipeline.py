#!/usr/bin/env python3
"""
Market Master MLOps Pipeline Demo Script
Demonstrates the complete MLOps pipeline for Market Master.
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.main import MarketMasterApp
from src.data import AssetClass, generate_training_data
from src.models import ActionPredictor, train_action_predictor, evaluate_model
from src.mlops import ModelRegistry, ComprehensiveMonitor
from src.llm import MockTradingCoach, TradingPersona
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)


def print_step(step_num: int, title: str):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {title}")
    print("-" * 60)


def print_success(message: str, data: dict = None):
    """Print success message."""
    print(f"✅ {message}")
    if data:
        for key, value in data.items():
            print(f"   {key}: {value}")


def print_metrics(metrics: dict, title: str = "Metrics"):
    """Print formatted metrics."""
    print(f"\n📊 {title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


def run_complete_demo():
    """Run the complete Market Master MLOps demo."""
    print_header("Market Master - AI-Powered Trading Assistant")
    print("🚀 Complete MLOps Pipeline Demonstration")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize application
    app = MarketMasterApp()
    
    # Demo configuration
    asset_class = AssetClass.EQUITY
    instrument = "AAPL"
    n_samples = 10000
    
    print(f"\n🎯 Demo Configuration:")
    print(f"   Asset Class: {asset_class.value}")
    print(f"   Instrument: {instrument}")
    print(f"   Training Samples: {n_samples:,}")
    
    demo_results = {
        'demo_start_time': datetime.now().isoformat(),
        'configuration': {
            'asset_class': asset_class.value,
            'instrument': instrument,
            'n_samples': n_samples
        },
        'steps': {}
    }
    
    try:
        # Step 1: Data Generation
        print_step(1, "Data Generation & Technical Analysis")
        start_time = time.time()
        
        print("📈 Generating realistic market data...")
        training_data = generate_training_data(asset_class, instrument, n_samples)
        
        generation_time = time.time() - start_time
        
        print_success("Data generation completed", {
            'data_shape': training_data.shape,
            'features_count': len(training_data.columns) - 3,
            'generation_time': f"{generation_time:.2f}s"
        })
        
        print_metrics({
            'total_samples': len(training_data),
            'label_distribution': training_data['action'].value_counts().to_dict(),
            'feature_columns': len(training_data.columns) - 3
        }, "Data Statistics")
        
        demo_results['steps']['data_generation'] = {
            'status': 'success',
            'data_shape': training_data.shape,
            'generation_time': generation_time,
            'label_distribution': training_data['action'].value_counts().to_dict()
        }
        
        # Step 2: Model Training
        print_step(2, "Model Training & Experiment Tracking")
        start_time = time.time()
        
        print("🤖 Training Action Predictor model...")
        
        # Split data
        train_size = int(len(training_data) * 0.8)
        train_data = training_data.iloc[:train_size]
        val_data = training_data.iloc[train_size:int(len(training_data) * 0.9)]
        test_data = training_data.iloc[int(len(training_data) * 0.9):]
        
        # Prepare features and labels
        X_train = train_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        y_train = train_data['action']
        X_val = val_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        y_val = val_data['action']
        X_test = test_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        y_test = test_data['action']
        
        # Train model
        model, train_metrics = train_action_predictor(X_train, y_train, X_val, y_val)
        
        training_time = time.time() - start_time
        
        print_success("Model training completed", {
            'training_time': f"{training_time:.2f}s",
            'model_type': 'Random Forest Classifier',
            'features_used': len(model.feature_names)
        })
        
        print_metrics(train_metrics, "Training Metrics")
        
        demo_results['steps']['model_training'] = {
            'status': 'success',
            'training_metrics': train_metrics,
            'training_time': training_time,
            'model_config': model.model_config
        }
        
        # Step 3: Model Evaluation
        print_step(3, "Model Evaluation & Performance Analysis")
        start_time = time.time()
        
        print("📊 Evaluating model performance...")
        eval_results = evaluate_model(model, X_test, y_test)
        
        evaluation_time = time.time() - start_time
        
        print_success("Model evaluation completed", {
            'evaluation_time': f"{evaluation_time:.2f}s",
            'test_samples': len(X_test)
        })
        
        print_metrics(eval_results['metrics'], "Performance Metrics")
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n🔝 Top 10 Feature Importance:")
        for feature, importance in top_features:
            print(f"   {feature}: {importance:.4f}")
        
        demo_results['steps']['model_evaluation'] = {
            'status': 'success',
            'evaluation_metrics': eval_results['metrics'],
            'evaluation_time': evaluation_time,
            'top_features': top_features
        }
        
        # Step 4: Model Registry
        print_step(4, "Model Registry & Versioning")
        start_time = time.time()
        
        print("📦 Registering model in MLflow...")
        
        model_name = f"action_predictor_{asset_class.value}_{instrument}"
        model_uri = app.registry.register_model(
            model.model,
            model_name,
            train_metrics,
            parameters=model.model_config,
            tags={
                'asset_class': asset_class.value,
                'instrument': instrument,
                'demo_run': 'true',
                'accuracy': f"{eval_results['metrics']['accuracy']:.4f}"
            }
        )
        
        registry_time = time.time() - start_time
        
        print_success("Model registered successfully", {
            'model_uri': model_uri,
            'model_name': model_name,
            'registry_time': f"{registry_time:.2f}s"
        })
        
        # List models
        models_info = app.list_models()
        print(f"\n📋 Registered Models: {models_info['count']}")
        
        demo_results['steps']['model_registry'] = {
            'status': 'success',
            'model_uri': model_uri,
            'model_name': model_name,
            'registry_time': registry_time
        }
        
        # Step 5: Model Monitoring
        print_step(5, "Model Monitoring & Drift Detection")
        start_time = time.time()
        
        print("🔍 Setting up comprehensive monitoring...")
        
        monitor = ComprehensiveMonitor(train_data)
        monitoring_results = monitor.run_monitoring(test_data, eval_results['predictions'])
        
        monitoring_time = time.time() - start_time
        
        print_success("Monitoring setup completed", {
            'monitoring_time': f"{monitoring_time:.2f}s",
            'overall_status': monitoring_results['overall_status']
        })
        
        print_metrics({
            'data_quality_score': monitoring_results['data_quality_score'],
            'drift_detected': monitoring_results['drift_detected'],
            'performance_metrics': monitoring_results['performance_metrics']
        }, "Monitoring Results")
        
        demo_results['steps']['model_monitoring'] = {
            'status': 'success',
            'monitoring_results': monitoring_results,
            'monitoring_time': monitoring_time
        }
        
        # Step 6: Trading Coach
        print_step(6, "Trading Coach & Decision Support")
        start_time = time.time()
        
        print("🎯 Generating trading advice...")
        
        # Use recent data for real-time simulation
        recent_data = test_data.tail(10)
        predictions = model.predict(recent_data)
        confidences = model.predict_proba(recent_data).max(axis=1)
        
        coach = MockTradingCoach(TradingPersona.MODERATE)
        trading_advice = []
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            advice = coach.get_trading_advice(pred, conf, recent_data.iloc[i:i+1])
            trading_advice.append(advice)
        
        coaching_time = time.time() - start_time
        
        print_success("Trading advice generated", {
            'advice_count': len(trading_advice),
            'coaching_time': f"{coaching_time:.2f}s"
        })
        
        # Show sample advice
        if trading_advice:
            sample_advice = trading_advice[0]
            print(f"\n💡 Sample Trading Advice:")
            print(f"   Action: {sample_advice['action']}")
            print(f"   Confidence: {sample_advice['confidence']:.3f}")
            print(f"   Position Size: {sample_advice['position_size']:.1%}")
            print(f"   Stop Loss: {sample_advice['stop_loss']:.1%}")
            print(f"   Take Profit: {sample_advice['take_profit']:.1%}")
            print(f"   Market Condition: {sample_advice['market_condition']}")
            print(f"   Explanation: {sample_advice['explanation']}")
        
        demo_results['steps']['trading_coach'] = {
            'status': 'success',
            'advice_count': len(trading_advice),
            'coaching_time': coaching_time,
            'sample_advice': trading_advice[0] if trading_advice else None
        }
        
        # Step 7: Model Persistence
        print_step(7, "Model Persistence & Deployment Ready")
        start_time = time.time()
        
        print("💾 Saving model locally...")
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"action_predictor_{asset_class.value}_{instrument}.joblib"
        model.save_model(str(model_path))
        
        persistence_time = time.time() - start_time
        
        print_success("Model saved successfully", {
            'model_path': str(model_path),
            'file_size': f"{model_path.stat().st_size / 1024:.1f} KB",
            'persistence_time': f"{persistence_time:.2f}s"
        })
        
        demo_results['steps']['model_persistence'] = {
            'status': 'success',
            'model_path': str(model_path),
            'persistence_time': persistence_time
        }
        
        # Final Summary
        print_header("Demo Summary & Results")
        
        total_time = time.time() - demo_results['demo_start_time']
        demo_results['demo_end_time'] = datetime.now().isoformat()
        demo_results['total_time'] = total_time
        
        print("🎉 Market Master MLOps Pipeline Demo Completed Successfully!")
        print(f"⏱️  Total Demo Time: {total_time:.2f} seconds")
        
        print("\n📊 Key Results:")
        print(f"   🎯 Model Accuracy: {eval_results['metrics']['accuracy']:.1%}")
        print(f"   🎯 F1 Score: {eval_results['metrics']['f1_score']:.3f}")
        print(f"   📈 Data Quality Score: {monitoring_results['data_quality_score']:.3f}")
        print(f"   🔍 Drift Detected: {'Yes' if monitoring_results['drift_detected'] else 'No'}")
        print(f"   🏥 Overall Status: {monitoring_results['overall_status']}")
        print(f"   💡 Trading Advice Generated: {len(trading_advice)}")
        
        print("\n🚀 MLOps Components Demonstrated:")
        print("   ✅ Data Generation & Technical Analysis")
        print("   ✅ Model Training & Experiment Tracking (MLflow)")
        print("   ✅ Model Evaluation & Performance Analysis")
        print("   ✅ Model Registry & Versioning")
        print("   ✅ Model Monitoring & Drift Detection (Evidently)")
        print("   ✅ Trading Coach & Decision Support")
        print("   ✅ Model Persistence & Deployment Ready")
        
        # Save demo results
        results_file = Path("demo_results.json")
        with open(results_file, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        print(f"\n💾 Demo results saved to: {results_file}")
        
        return demo_results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        demo_results['error'] = str(e)
        demo_results['demo_end_time'] = datetime.now().isoformat()
        return demo_results


def main():
    """Main entry point for the demo script."""
    print("🎯 Market Master MLOps Pipeline Demo")
    print("=" * 50)
    
    # Run the complete demo
    results = run_complete_demo()
    
    if 'error' not in results:
        print("\n🎊 Demo completed successfully!")
        print("Ready for MLOps Zoomcamp submission! 🚀")
    else:
        print("\n💥 Demo encountered an error.")
        print("Please check the logs for details.")


if __name__ == "__main__":
    main() 