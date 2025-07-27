#!/usr/bin/env python3
"""
Simple Market Master Demo Script
Bypasses import issues and directly demonstrates the core functionality.
"""

import sys
import os
import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import AssetClass, generate_training_data, generate_market_data
from src.models import ActionPredictor
from src.utils.logger import get_logger

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


def ensure_directories():
    """Ensure required directories exist."""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}/")


def save_data_files(all_data, asset_classes):
    """Save data files to data/ directory."""
    print("\n💾 Saving data files...")
    
    # Save individual asset data
    for (asset_class, instrument), data in zip(asset_classes, all_data):
        filename = f"data/{asset_class.value}_{instrument.replace('/', '_')}_training.csv"
        data.to_csv(filename, index=False)
        print(f"   📄 {filename} ({len(data)} samples)")
    
    # Save combined data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_filename = "data/combined_training_data.csv"
    combined_data.to_csv(combined_filename, index=False)
    print(f"   📄 {combined_filename} ({len(combined_data)} samples)")
    
    return combined_data


def save_model(model, metrics):
    """Save model to models/ directory."""
    print("\n🤖 Saving model...")
    
    # Save model file
    model_filename = "models/market_master_model.joblib"
    model.save_model(model_filename)
    print(f"   📄 {model_filename}")
    
    # Save model metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'features_count': len(model.feature_names),
        'classes': list(model.classes),
        'training_metrics': metrics,
        'created_at': datetime.now().isoformat()
    }
    
    metadata_filename = "models/model_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   📄 {metadata_filename}")
    
    return model_filename


def save_logs(demo_results):
    """Save logs to logs/ directory."""
    print("\n📝 Saving logs...")
    
    # Save demo results
    log_filename = f"logs/demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    print(f"   📄 {log_filename}")
    
    # Save summary log
    summary_filename = "logs/demo_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write("Market Master Demo Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {demo_results['results']['data_generation']['total_samples']:,}\n")
        f.write(f"Asset Classes: {len(demo_results['configuration']['asset_classes'])}\n")
        f.write(f"Model Accuracy: {demo_results['results']['model_training']['metrics']['train_accuracy']:.4f}\n")
        f.write(f"Training Time: {demo_results['results']['model_training']['training_time']:.2f}s\n")
        f.write(f"Generation Time: {demo_results['results']['data_generation']['generation_time']:.2f}s\n")
    
    print(f"   📄 {summary_filename}")


def run_simple_demo():
    """Run a simple Market Master demo."""
    print_header("Market Master - Complete Demo")
    print("🚀 Generating Synthetic Data, Training Models & Saving Files")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Demo configuration
    asset_classes = [
        (AssetClass.CRYPTO, "BTC/USD"),
        (AssetClass.FOREX, "EUR/USD"),
        (AssetClass.EQUITY, "AAPL"),
        (AssetClass.COMMODITY, "GOLD"),
        (AssetClass.INDICES, "SPY")
    ]
    
    n_samples = 1000  # 1000 samples per asset class
    
    print(f"\n🎯 Demo Configuration:")
    print(f"   Asset Classes: {len(asset_classes)}")
    print(f"   Samples per Asset: {n_samples:,}")
    print(f"   Total Samples: {len(asset_classes) * n_samples:,}")
    
    demo_results = {
        'demo_start_time': datetime.now().isoformat(),
        'configuration': {
            'asset_classes': [ac.value for ac, _ in asset_classes],
            'instruments': [inst for _, inst in asset_classes],
            'n_samples': n_samples
        },
        'results': {}
    }
    
    try:
        # Step 1: Data Generation for All Asset Classes
        print_step(1, "Data Generation for All Asset Classes")
        start_time = time.time()
        
        all_data = []
        
        for asset_class, instrument in asset_classes:
            print(f"\n📈 Generating data for {asset_class.value}: {instrument}")
            
            # Generate market data
            market_data = generate_market_data(asset_class, instrument, n_samples)
            print_success(f"Market data generated", {
                'shape': market_data.shape,
                'columns': list(market_data.columns)
            })
            
            # Generate training data with labels
            training_data = generate_training_data(asset_class, instrument, n_samples)
            print_success(f"Training data generated", {
                'shape': training_data.shape,
                'features': len(training_data.columns) - 3,  # Exclude action, asset_class, instrument
                'actions': training_data['action'].value_counts().to_dict()
            })
            
            all_data.append(training_data)
        
        generation_time = time.time() - start_time
        
        print_success("All data generation completed", {
            'total_samples': sum(len(data) for data in all_data),
            'generation_time': f"{generation_time:.2f}s"
        })
        
        demo_results['results']['data_generation'] = {
            'status': 'success',
            'total_samples': sum(len(data) for data in all_data),
            'generation_time': generation_time,
            'asset_data': [
                {
                    'asset_class': asset_class.value,
                    'instrument': instrument,
                    'samples': len(data),
                    'features': len(data.columns) - 3
                }
                for (asset_class, instrument), data in zip(asset_classes, all_data)
            ]
        }
        
        # Step 2: Save Data Files
        print_step(2, "Saving Data Files")
        combined_data = save_data_files(all_data, asset_classes)
        
        # Step 3: Model Training
        print_step(3, "Model Training")
        start_time = time.time()
        
        # Prepare features and labels
        X = combined_data.drop(['action', 'asset_class', 'instrument'], axis=1, errors='ignore')
        y = combined_data['action']
        
        print_success("Data prepared for training", {
            'features': X.shape[1],
            'samples': X.shape[0],
            'classes': len(y.unique())
        })
        
        # Train model
        model = ActionPredictor(use_mlflow=False)
        metrics = model.train(X, y, X, y)  # Using same data for train/val for demo
        
        training_time = time.time() - start_time
        
        print_success("Model training completed", {
            'training_time': f"{training_time:.2f}s",
            'accuracy': f"{metrics.get('train_accuracy', 0):.4f}",
            'f1_score': f"{metrics.get('train_f1', 0):.4f}"
        })
        
        demo_results['results']['model_training'] = {
            'status': 'success',
            'training_time': training_time,
            'metrics': metrics
        }
        
        # Step 4: Save Model
        print_step(4, "Saving Model")
        model_filename = save_model(model, metrics)
        
        # Step 5: Model Testing
        print_step(5, "Model Testing & Predictions")
        
        # Test predictions on sample data
        sample_data = X.head(10)
        predictions = model.predict(sample_data)
        probabilities = model.predict_proba(sample_data)
        
        print_success("Model predictions generated", {
            'predictions': list(predictions),
            'confidence_scores': [f"{max(prob):.3f}" for prob in probabilities]
        })
        
        demo_results['results']['model_testing'] = {
            'status': 'success',
            'predictions': list(predictions),
            'confidence_scores': [float(max(prob)) for prob in probabilities]
        }
        
        # Step 6: Save Logs
        print_step(6, "Saving Logs & Results")
        save_logs(demo_results)
        
        # Print summary
        print_header("Demo Summary")
        print(f"✅ Data Generation: {sum(len(data) for data in all_data):,} samples across {len(asset_classes)} asset classes")
        print(f"✅ Data Files: Saved to data/ directory")
        print(f"✅ Model Training: {training_time:.2f}s training time")
        print(f"✅ Model Files: Saved to models/ directory")
        print(f"✅ Model Accuracy: {metrics.get('train_accuracy', 0):.4f}")
        print(f"✅ Predictions: Working with confidence scores")
        print(f"✅ Log Files: Saved to logs/ directory")
        
        return demo_results
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    run_simple_demo() 