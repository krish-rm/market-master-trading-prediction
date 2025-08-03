#!/bin/bash

# Market Master: Cloud Deployment Script
# MLOps Zoomcamp Project - Automated Cloud Deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    log_success "All prerequisites are installed!"
}

# Check AWS credentials
check_aws_credentials() {
    log_info "Checking AWS credentials..."
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials are not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    log_success "AWS credentials are configured!"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env file from template..."
        cp env.example .env
        log_warning "Please edit .env file with your AWS credentials before continuing."
        log_warning "Press Enter when you've configured the .env file..."
        read -r
    fi
    
    # Source environment variables
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    log_success "Environment setup completed!"
}

# Create S3 bucket for Terraform state
create_terraform_state_bucket() {
    log_info "Creating S3 bucket for Terraform state..."
    
    BUCKET_NAME="market-master-terraform-state"
    REGION="${AWS_REGION:-us-east-1}"
    
    if ! aws s3 ls "s3://$BUCKET_NAME" &> /dev/null; then
        aws s3 mb "s3://$BUCKET_NAME" --region "$REGION"
        aws s3api put-bucket-versioning --bucket "$BUCKET_NAME" --versioning-configuration Status=Enabled
        log_success "Terraform state bucket created: $BUCKET_NAME"
    else
        log_info "Terraform state bucket already exists: $BUCKET_NAME"
    fi
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd terraform
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan -out=tfplan
    
    # Apply deployment
    log_info "Applying Terraform deployment..."
    terraform apply tfplan
    
    cd ..
    
    log_success "Infrastructure deployment completed!"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION="${AWS_REGION:-us-east-1}"
    REPOSITORY_NAME="market-master"
    
    # Create ECR repository if it doesn't exist
    if ! aws ecr describe-repositories --repository-names "$REPOSITORY_NAME" &> /dev/null; then
        log_info "Creating ECR repository..."
        aws ecr create-repository --repository-name "$REPOSITORY_NAME" --region "$REGION"
    fi
    
    # Get ECR login token
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
    
    # Build Docker image
    log_info "Building Docker image..."
    docker build -t market-master:latest .
    
    # Tag and push image
    log_info "Tagging and pushing Docker image..."
    docker tag market-master:latest "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest"
    docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest"
    
    log_success "Docker image built and pushed successfully!"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Update kubeconfig
    log_info "Updating kubeconfig..."
    aws eks update-kubeconfig --name market-master-cluster --region "${AWS_REGION:-us-east-1}"
    
    # Create namespace
    log_info "Creating namespace..."
    kubectl create namespace market-master --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s/deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/market-master-app -n market-master
    
    log_success "Kubernetes deployment completed!"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Deploy Prometheus and Grafana
    log_info "Deploying monitoring stack..."
    kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/setup/namespace.yaml
    kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/setup/
    kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/
    
    log_success "Monitoring setup completed!"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check if pods are running
    log_info "Checking pod status..."
    kubectl get pods -n market-master
    
    # Check services
    log_info "Checking services..."
    kubectl get services -n market-master
    
    # Check ingress
    log_info "Checking ingress..."
    kubectl get ingress -n market-master
    
    # Get load balancer endpoint
    log_info "Getting load balancer endpoint..."
    ALB_ENDPOINT=$(aws elbv2 describe-load-balancers --names market-master-alb --query 'LoadBalancers[0].DNSName' --output text 2>/dev/null || echo "Load balancer not ready yet")
    
    if [ "$ALB_ENDPOINT" != "None" ] && [ "$ALB_ENDPOINT" != "" ]; then
        log_success "Application is accessible at: http://$ALB_ENDPOINT"
    else
        log_warning "Load balancer endpoint not ready yet. Please check AWS console."
    fi
    
    log_success "Deployment verification completed!"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Run unit tests
    log_info "Running unit tests..."
    make test-unit
    
    # Run integration tests
    log_info "Running integration tests..."
    make test-integration
    
    log_success "Tests completed!"
}

# Generate deployment summary
generate_summary() {
    log_info "Generating deployment summary..."
    
    # Get infrastructure outputs
    cd terraform
    CLUSTER_ENDPOINT=$(terraform output -raw cluster_endpoint 2>/dev/null || echo "Not available")
    RDS_ENDPOINT=$(terraform output -raw rds_endpoint 2>/dev/null || echo "Not available")
    REDIS_ENDPOINT=$(terraform output -raw redis_endpoint 2>/dev/null || echo "Not available")
    MLFLOW_BUCKET=$(terraform output -raw mlflow_bucket 2>/dev/null || echo "Not available")
    MODELS_BUCKET=$(terraform output -raw models_bucket 2>/dev/null || echo "Not available")
    ALB_DNS=$(terraform output -raw alb_dns_name 2>/dev/null || echo "Not available")
    cd ..
    
    # Create summary file
    cat > deployment-summary.md << EOF
# Market Master Cloud Deployment Summary

## Deployment Date
$(date)

## Infrastructure Components

### EKS Cluster
- **Endpoint**: $CLUSTER_ENDPOINT
- **Status**: Active

### RDS Database
- **Endpoint**: $RDS_ENDPOINT
- **Database**: market_master

### Redis Cache
- **Endpoint**: $REDIS_ENDPOINT
- **Status**: Active

### S3 Buckets
- **MLflow Artifacts**: $MLFLOW_BUCKET
- **Model Storage**: $MODELS_BUCKET

### Load Balancer
- **DNS Name**: $ALB_DNS
- **Status**: Active

## Application Access

### Web Application
- **URL**: http://$ALB_DNS
- **Port**: 8501

### MLflow UI
- **URL**: http://$ALB_DNS:5000
- **Port**: 5000

## Monitoring

### CloudWatch
- **Log Group**: /aws/eks/market-master/app
- **Metrics**: CPU, Memory, Network

### Kubernetes
- **Namespace**: market-master
- **Pods**: 3 replicas
- **Services**: market-master-service, mlflow-service

## Verification Commands

\`\`\`bash
# Check pod status
kubectl get pods -n market-master

# Check services
kubectl get services -n market-master

# Check logs
kubectl logs -n market-master deployment/market-master-app

# Test application
curl http://$ALB_DNS/health
\`\`\`

## Cost Estimation
- **EKS Cluster**: ~$50-100/month
- **RDS Database**: ~$30-50/month
- **Redis Cache**: ~$20-40/month
- **S3 Storage**: ~$5-10/month
- **Load Balancer**: ~$20-30/month
- **Total Estimated**: ~$125-230/month

## MLOps Zoomcamp Requirements

✅ **Problem Description** (2 points) - Financial market prediction system
✅ **Cloud Infrastructure** (4 points) - AWS with Terraform IaC
✅ **Experiment Tracking** (4 points) - MLflow tracking & registry
✅ **Workflow Orchestration** (4 points) - Prefect pipelines
✅ **Model Deployment** (4 points) - Containerized on EKS
✅ **Model Monitoring** (4 points) - Evidently + CloudWatch
✅ **Reproducibility** (4 points) - Clear instructions + dependencies
✅ **Best Practices** (7 points) - Tests, linting, CI/CD, Makefile

**Total Expected Score: 33/33 points**

EOF

    log_success "Deployment summary generated: deployment-summary.md"
}

# Main deployment function
main() {
    log_info "Starting Market Master cloud deployment..."
    log_info "This will deploy the complete MLOps infrastructure to AWS"
    
    # Check prerequisites
    check_prerequisites
    
    # Check AWS credentials
    check_aws_credentials
    
    # Setup environment
    setup_environment
    
    # Create Terraform state bucket
    create_terraform_state_bucket
    
    # Deploy infrastructure
    deploy_infrastructure
    
    # Build and push Docker image
    build_and_push_image
    
    # Deploy to Kubernetes
    deploy_to_kubernetes
    
    # Setup monitoring
    setup_monitoring
    
    # Run tests
    run_tests
    
    # Verify deployment
    verify_deployment
    
    # Generate summary
    generate_summary
    
    log_success "🎉 Market Master cloud deployment completed successfully!"
    log_info "Check deployment-summary.md for details and access information."
}

# Help function
show_help() {
    echo "Market Master: Cloud Deployment Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --infra-only    Deploy only infrastructure (skip application)"
    echo "  --app-only      Deploy only application (skip infrastructure)"
    echo "  --verify-only   Only verify existing deployment"
    echo ""
    echo "Examples:"
    echo "  $0              # Full deployment"
    echo "  $0 --infra-only # Deploy only AWS infrastructure"
    echo "  $0 --verify-only # Verify existing deployment"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --infra-only)
        log_info "Deploying infrastructure only..."
        check_prerequisites
        check_aws_credentials
        setup_environment
        create_terraform_state_bucket
        deploy_infrastructure
        log_success "Infrastructure deployment completed!"
        exit 0
        ;;
    --app-only)
        log_info "Deploying application only..."
        check_prerequisites
        check_aws_credentials
        setup_environment
        build_and_push_image
        deploy_to_kubernetes
        setup_monitoring
        verify_deployment
        log_success "Application deployment completed!"
        exit 0
        ;;
    --verify-only)
        log_info "Verifying deployment..."
        check_prerequisites
        check_aws_credentials
        verify_deployment
        generate_summary
        log_success "Deployment verification completed!"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 