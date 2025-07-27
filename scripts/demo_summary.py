#!/usr/bin/env python3
"""
Comprehensive Summary of Market Master MLOps Demo
Provides all relevant details about the completed production-level MLOps deployment
"""

import json
import sqlite3
import mlflow
import mlflow.tracking
from pathlib import Path
from datetime import datetime
import pandas as pd

def get_mlflow_summary():
    """Get comprehensive MLflow summary."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get experiments
        experiments = client.search_experiments()
        
        # Get registered models
        models = client.search_registered_models()
        
        # Get recent runs
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id for exp in experiments],
            max_results=10
        )
        
        return {
            'experiments': len(experiments),
            'registered_models': len(models),
            'total_runs': len(list(runs)),
            'experiment_names': [exp.name for exp in experiments],
            'model_names': [model.name for model in models]
        }
    except Exception as e:
        return {'error': str(e)}

def get_database_summary():
    """Get SQLite database summary."""
    try:
        conn = sqlite3.connect('mlflow.db')
        cursor = conn.cursor()
        
        # Count records in key tables
        cursor.execute("SELECT COUNT(*) FROM experiments")
        experiments_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM runs")
        runs_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM registered_models")
        models_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_versions")
        versions_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'experiments_count': experiments_count,
            'runs_count': runs_count,
            'models_count': models_count,
            'versions_count': versions_count,
            'database_size_mb': Path('mlflow.db').stat().st_size / (1024 * 1024)
        }
    except Exception as e:
        return {'error': str(e)}

def get_file_structure_summary():
    """Get project file structure summary."""
    directories = {
        'data': 'data/',
        'models': 'models/',
        'logs': 'logs/',
        'mlartifacts': 'mlartifacts/',
        'mlruns': 'mlruns/',
        'src': 'src/',
        'scripts': 'scripts/',
        'tests': 'tests/',
        'docs': 'docs/'
    }
    
    structure = {}
    for name, path in directories.items():
        if Path(path).exists():
            if Path(path).is_dir():
                try:
                    files = list(Path(path).rglob('*'))
                    structure[name] = {
                        'type': 'directory',
                        'files_count': len([f for f in files if f.is_file()]),
                        'subdirs_count': len([f for f in files if f.is_dir()]),
                        'total_size_mb': sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                    }
                except Exception:
                    structure[name] = {'type': 'directory', 'error': 'Access denied'}
            else:
                structure[name] = {'type': 'file', 'size_mb': Path(path).stat().st_size / (1024 * 1024)}
        else:
            structure[name] = {'type': 'missing'}
    
    return structure

def get_demo_results():
    """Get demo results from JSON file."""
    results_file = "logs/production_mlops_demo_results.json"
    if Path(results_file).exists():
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {'error': f'Failed to load results: {e}'}
    else:
        return {'error': 'Results file not found'}

def print_comprehensive_summary():
    """Print comprehensive summary of the MLOps demo."""
    print("=" * 100)
    print("🎯 MARKET MASTER MLOPS DEMO - COMPREHENSIVE SUMMARY")
    print("=" * 100)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # MLflow Summary
    print("🔬 MLFLOW TRACKING & MODEL REGISTRY")
    print("-" * 50)
    mlflow_summary = get_mlflow_summary()
    if 'error' not in mlflow_summary:
        print(f"✅ Experiments: {mlflow_summary['experiments']}")
        print(f"✅ Registered Models: {mlflow_summary['registered_models']}")
        print(f"✅ Total Runs: {mlflow_summary['total_runs']}")
        print(f"📊 Experiment Names: {', '.join(mlflow_summary['experiment_names'])}")
        print(f"🤖 Model Names: {', '.join(mlflow_summary['model_names'])}")
    else:
        print(f"❌ MLflow Error: {mlflow_summary['error']}")
    print()
    
    # Database Summary
    print("🗄️ DATABASE STORAGE")
    print("-" * 50)
    db_summary = get_database_summary()
    if 'error' not in db_summary:
        print(f"📊 Database Size: {db_summary['database_size_mb']:.2f} MB")
        print(f"📈 Experiments: {db_summary['experiments_count']}")
        print(f"🏃 Runs: {db_summary['runs_count']}")
        print(f"📦 Models: {db_summary['models_count']}")
        print(f"🔢 Versions: {db_summary['versions_count']}")
    else:
        print(f"❌ Database Error: {db_summary['error']}")
    print()
    
    # File Structure
    print("📁 PROJECT STRUCTURE")
    print("-" * 50)
    structure = get_file_structure_summary()
    for name, info in structure.items():
        if info['type'] == 'directory' and 'error' not in info:
            print(f"📂 {name}/: {info['files_count']} files, {info['subdirs_count']} dirs, {info['total_size_mb']:.2f} MB")
        elif info['type'] == 'file':
            print(f"📄 {name}: {info['size_mb']:.2f} MB")
        elif info['type'] == 'missing':
            print(f"❌ {name}/: Missing")
    print()
    
    # Demo Results
    print("📋 DEMO EXECUTION RESULTS")
    print("-" * 50)
    demo_results = get_demo_results()
    if 'error' not in demo_results:
        steps = demo_results.get('steps', {})
        for step_name, step_info in steps.items():
            status = step_info.get('status', 'unknown')
            status_icon = '✅' if status == 'success' else '⚠️' if status == 'warning' else '❌'
            print(f"{status_icon} {step_name.replace('_', ' ').title()}: {status}")
        
        # Overall status
        overall_ready = demo_results.get('overall_ready', False)
        print(f"\n🎯 Overall Production Ready: {'✅ YES' if overall_ready else '❌ NO'}")
    else:
        print(f"❌ Demo Results Error: {demo_results['error']}")
    print()
    
    # Key Metrics
    print("📊 KEY PERFORMANCE METRICS")
    print("-" * 50)
    if 'error' not in demo_results:
        metrics = demo_results.get('metrics', {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"📈 {metric_name.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"📈 {metric_name.replace('_', ' ').title()}: {value}")
    print()
    
    # Access Information
    print("🔗 ACCESS INFORMATION")
    print("-" * 50)
    print("🌐 MLflow UI: http://localhost:5000")
    print("📁 Project Root: " + str(Path.cwd()))
    print("🗄️ Database: mlflow.db")
    print("📊 Artifacts: mlartifacts/")
    print("📄 Results: logs/production_mlops_demo_results.json")
    print()
    
    # Next Steps
    print("🚀 NEXT STEPS & RECOMMENDATIONS")
    print("-" * 50)
    print("1. 🌐 Open MLflow UI to explore experiments and models")
    print("2. 📊 Review monitoring results in logs/")
    print("3. 🤖 Test model predictions with different data")
    print("4. 🔄 Set up automated retraining pipelines")
    print("5. ☁️ Deploy to cloud infrastructure (AWS/GCP/Azure)")
    print("6. 📈 Set up production monitoring and alerting")
    print("7. 🔒 Implement security and access controls")
    print("8. 📚 Document API endpoints and usage")
    print()
    
    print("=" * 100)
    print("🎉 MARKET MASTER MLOPS DEMO SUMMARY COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    print_comprehensive_summary() 