# 🚀 Deployment Guide

## 🏠 Local Development Setup

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository-url>
cd Project_MLOps
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start Demo
```bash
# Run complete end-to-end demo
make demo
# Generates data, trains model, saves files to data/, models/, logs/
```

### 3. Launch Web Application
```bash
# Start Streamlit web interface
make run
# Opens interactive dashboard at http://localhost:8501
```

### 4. Production-Level MLOps Demo
```bash
# Start MLflow server (in separate terminal)
make start-mlflow
# Opens MLflow UI at http://localhost:5000

# Run comprehensive production MLOps demo
make demo-production
# Demonstrates MLflow tracking, model registry, monitoring

# Generate comprehensive summary
make demo-summary
# Shows all demo results, metrics, and system status
```

### 5. Testing & Validation
```bash
# Run all tests
make test
# Unit tests, integration tests, validation

# Health check
make health-check
# Validates all components and dependencies
```

### 6. MLflow Management
```bash
# Start MLflow tracking server
make start-mlflow

# Open MLflow UI in browser
make mlflow-ui

# Stop MLflow server
make mlflow-stop
```

## ☁️ Cloud Deployment

### Staging Environment
```bash
make deploy-staging
```

### Production Environment
```bash
make deploy-production
```

## 📋 Deployment Prerequisites

### Local Development
- Python 3.8+
- Virtual environment (venv)
- Git
- Make (for automation)

### Cloud Deployment
- AWS CLI configured
- Terraform installed
- Docker installed
- Kubernetes cluster (for production)

## 🔧 Configuration

### Environment Variables
Copy the example environment file and configure:
```bash
cp env.example .env
# Edit .env with your configurations
```

### MLflow Configuration
- **Tracking URI**: http://localhost:5000 (local) or cloud endpoint
- **Backend Store**: SQLite (local) or PostgreSQL (cloud)
- **Artifact Store**: Local filesystem or S3 (cloud)

## 📊 Monitoring & Health Checks

### Local Monitoring
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000
- **Health Check**: `make health-check`

### Cloud Monitoring
- **Grafana Dashboards**: Model performance and system health
- **Prometheus Metrics**: Real-time monitoring
- **AlertManager**: Automated alerting

## 🚨 Troubleshooting

### Common Issues

#### MLflow Server Not Starting
```bash
# Check if port 5000 is available
netstat -an | grep 5000

# Kill existing processes
pkill -f "mlflow server"

# Start fresh
make start-mlflow
```

#### Streamlit App Not Loading
```bash
# Check if port 8501 is available
netstat -an | grep 8501

# Restart Streamlit
make run
```

#### Model Training Failures
```bash
# Check dependencies
pip list | grep -E "(scikit-learn|pandas|numpy)"

# Reinstall if needed
pip install -r requirements.txt
```

### Log Files
- **Application Logs**: `logs/`
- **MLflow Logs**: Console output
- **Demo Results**: `logs/production_mlops_demo_results.json`

## 🔄 CI/CD Pipeline

### GitHub Actions
The project includes automated CI/CD pipelines:
- **Testing**: Automated unit and integration tests
- **Building**: Docker image creation
- **Deployment**: Automated deployment to staging/production

### Manual Deployment
```bash
# Build Docker image
docker build -t market-master .

# Run container
docker run -p 8501:8501 market-master

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 📈 Scaling

### Horizontal Scaling
- **Load Balancer**: Distribute traffic across multiple instances
- **Auto-scaling**: Kubernetes HPA for automatic scaling
- **Database**: PostgreSQL with read replicas

### Vertical Scaling
- **Resource Limits**: Configure CPU/memory limits
- **Performance Tuning**: Optimize model inference
- **Caching**: Redis for session management

## 🔒 Security

### Local Development
- **Environment Variables**: Secure configuration management
- **Virtual Environment**: Isolated dependencies
- **Access Control**: Local-only access

### Production Security
- **Authentication**: OAuth2/JWT token authentication
- **Authorization**: Role-based access control
- **Encryption**: TLS/SSL for data in transit
- **Secrets Management**: Kubernetes secrets or AWS Secrets Manager

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform Documentation](https://www.terraform.io/docs)

---

**Need help? Check the troubleshooting section or create an issue in the repository! 🆘** 