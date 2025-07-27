# Market Master: Financial Market Prediction System
# Makefile for automation

.PHONY: help install install-dev setup test test-unit test-integration test-e2e lint format clean run demo deploy-infrastructure deploy-models start-monitoring stop-monitoring logs

# Default target
help:
	@echo "🎯 Market Master: Financial Market Prediction System"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install          - Install production dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  setup            - Setup environment and configuration"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-e2e         - Run end-to-end tests"
	@echo "  lint             - Run linting"
	@echo "  format           - Format code"
	@echo "  clean            - Clean build artifacts"
	@echo "  run              - Run the application"
	@echo "  demo             - Run complete MLOps demo"
	@echo "  deploy-infrastructure - Deploy infrastructure with Terraform"
	@echo "  deploy-models    - Deploy models to production"
	@echo "  start-monitoring - Start monitoring services"
	@echo "  stop-monitoring  - Stop monitoring services"
	@echo "  logs             - View application logs"

# Installation
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements-dev.txt
	pre-commit install

setup:
	@echo "⚙️  Setting up Market Master Financial Prediction System..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from template..."; \
		cp env.example .env; \
		echo "✅ Please edit .env with your configurations"; \
	fi
	@echo "📁 Creating necessary directories..."
	mkdir -p models logs data
	@echo "✅ Setup completed!"

# Testing
test: test-unit test-integration test-e2e

test-unit:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

test-integration:
	@echo "🔗 Running integration tests..."
	python -m pytest tests/integration/ -v

test-e2e:
	@echo "🌐 Running end-to-end tests..."
	python -m pytest tests/e2e/ -v

# Code Quality
lint:
	@echo "🔍 Running linting..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	pylint src/ --disable=C0114,C0116

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ --line-length=88
	isort src/ tests/

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf logs/*.log
	rm -rf models/*.joblib
	rm -rf demo_results.json
	@echo "✅ Cleanup completed!"

# Application
run:
	@echo "🚀 Starting Market Master Financial Prediction System..."
	streamlit run src/app.py --server.port 8501 --server.address localhost --server.headless false

demo:
	@echo "🎯 Running complete Market Master pipeline demo..."
	python -c "import sys; sys.path.append('.'); from scripts.simple_demo import run_simple_demo; run_simple_demo()"

demo-production:
	@echo "🚀 Running production-level MLOps demo..."
	python scripts/production_mlops_demo_simple.py

demo-summary:
	@echo "📊 Generating comprehensive demo summary..."
	python scripts/demo_summary.py

demo-crypto:
	@echo "🎯 Running crypto trading demo..."
	python scripts/crypto_trading_demo.py

# Infrastructure
deploy-infrastructure:
	@echo "🏗️  Deploying infrastructure with Terraform..."
	cd terraform && terraform init
	cd terraform && terraform plan
	cd terraform && terraform apply -auto-approve
	@echo "✅ Infrastructure deployed!"

deploy-models:
	@echo "🤖 Deploying models to production..."
	python -c "from src.mlops.pipeline import deploy_training_pipeline, deploy_inference_pipeline; deploy_training_pipeline(); deploy_inference_pipeline()"
	@echo "✅ Models deployed!"

# Monitoring
start-monitoring:
	@echo "📊 Starting monitoring services..."
	docker-compose up -d grafana prometheus mlflow
	@echo "✅ Monitoring services started!"
	@echo "📈 Grafana: http://localhost:3000"
	@echo "📊 Prometheus: http://localhost:9090"
	@echo "🔬 MLflow: http://localhost:5000"

stop-monitoring:
	@echo "🛑 Stopping monitoring services..."
	docker-compose down
	@echo "✅ Monitoring services stopped!"

logs:
	@echo "📋 Viewing application logs..."
	tail -f logs/market_master.log

# Development
dev-setup: install-dev setup
	@echo "🔧 Development environment setup completed!"

quick-test:
	@echo "⚡ Running quick tests..."
	python -m pytest tests/unit/ -v --tb=short

# Docker
build:
	@echo "🐳 Building Docker image..."
	docker build -t market-master:latest .

run-docker:
	@echo "🐳 Running Market Master Financial Prediction System in Docker..."
	docker run -p 8501:8501 market-master:latest

# MLflow
start-mlflow:
	@echo "🔬 Starting MLflow tracking server..."
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

mlflow-ui:
	@echo "🔗 Opening MLflow UI..."
	start http://localhost:5000

mlflow-stop:
	@echo "🛑 Stopping MLflow server..."
	pkill -f "mlflow server" || true

# Prefect
start-prefect:
	@echo "🔄 Starting Prefect server..."
	prefect server start

run-prefect-pipeline:
	@echo "🔄 Running Prefect Market Master pipeline..."
	python workflows/mlops_pipeline.py

# Data
generate-data:
	@echo "📊 Generating sample data..."
	python -c "from src.data import generate_training_data, AssetClass; data = generate_training_data(AssetClass.EQUITY, 'AAPL', 1000); print(f'Generated {len(data)} samples')"

# Model
train-model:
	@echo "🤖 Training model..."
	python -c "from src.main import MarketMasterApp; app = MarketMasterApp(); results = app.run_demo('equity', 'AAPL', 5000); print(f'Training completed with accuracy: {results[\"summary\"][\"model_accuracy\"]:.3f}')"

# Quick demo
quick-demo:
	@echo "⚡ Running quick demo..."
	python src/main.py --demo --samples 1000

# Production
prod-deploy:
	@echo "🚀 Production deployment..."
	make install
	make setup
	make test
	make deploy-infrastructure
	make deploy-models
	make start-monitoring
	@echo "✅ Production deployment completed!"

# Health check
health-check:
	@echo "🏥 Running health checks..."
	@echo "Checking Python environment..."
	python --version
	@echo "Checking dependencies..."
	pip list | grep -E "(mlflow|prefect|evidently|scikit-learn)"
	@echo "Checking configuration..."
	@if [ -f .env ]; then echo "✅ .env file exists"; else echo "❌ .env file missing"; fi
	@echo "✅ Health check completed!"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	pydoc-markdown --render-toc
	@echo "✅ Documentation generated!"

# Backup
backup:
	@echo "💾 Creating backup..."
	tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz models/ logs/ data/ --exclude=*.log
	@echo "✅ Backup created!"

# Restore
restore:
	@echo "📥 Restoring from backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "❌ Please specify BACKUP_FILE=filename.tar.gz"; \
		exit 1; \
	fi
	tar -xzf $(BACKUP_FILE)
	@echo "✅ Restore completed!"

# Performance
benchmark:
	@echo "⚡ Running performance benchmarks..."
	python -c "import time; from src.data import generate_training_data, AssetClass; start=time.time(); data=generate_training_data(AssetClass.EQUITY, 'AAPL', 10000); print(f'Data generation: {time.time()-start:.2f}s'); from src.models import train_action_predictor; X=data.drop(['action', 'asset_class', 'instrument'], axis=1); y=data['action']; start=time.time(); model, _ = train_action_predictor(X.iloc[:8000], y.iloc[:8000], X.iloc[8000:9000], y.iloc[8000:9000]); print(f'Training: {time.time()-start:.2f}s')"

# Security
security-check:
	@echo "🔒 Running security checks..."
	safety check
	bandit -r src/
	@echo "✅ Security check completed!"

# Market Master Pipeline
market-master-pipeline: install-dev setup test format lint demo run-prefect-pipeline
	@echo "🎉 Complete Market Master pipeline executed successfully!"

# All-in-one
all: install-dev setup test format lint demo
	@echo "🎉 All tasks completed successfully!" 