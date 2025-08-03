#!/usr/bin/env python
"""Prefect Training Deployment Script"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from src.mlops.pipeline import market_master_training_flow

if __name__ == "__main__":
    print("Running Market Master Training Flow via Prefect...")
    
    # Setup MLflow with proper experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Create experiment if it doesn't exist
    experiment_name = "market_master_training"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f"✅ Created experiment: {experiment_name}")
        else:
            print(f"✅ Using existing experiment: {experiment_name}")
    except Exception as e:
        print(f"⚠️  Error with experiment: {e}")
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    try:
        result = market_master_training_flow(
            asset_class="equity",
            instrument="AAPL", 
            n_samples=10000
        )
        print(f"✅ Training completed! Model URI: {result.get('model_uri', 'N/A')}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
