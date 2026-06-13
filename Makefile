# ============================================
# CLV Predictor - Makefile
# ============================================

.PHONY: help install train test api dashboard docker-build docker-up docker-down clean

# Default target
help:
	@echo "=========================================="
	@echo "💰 CLV Predictor - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "  make install        Install dependencies"
	@echo "  make train          Train CLV models"
	@echo "  make test           Run API tests"
	@echo "  make api            Start API server"
	@echo "  make dashboard      Start Streamlit dashboard"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make clean          Clean generated files"
	@echo ""
	@echo "=========================================="

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Train models
train:
	@echo "📊 Training CLV models..."
	python -m src.train --data data/online_retail.csv --output models
	@echo "✅ Training complete"

# Run tests
test:
	@echo "🧪 Running API tests..."
	python test_api.py

# Start API
api:
	@echo "🚀 Starting API server..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start dashboard
dashboard:
	@echo "📈 Starting Streamlit dashboard..."
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Build Docker images
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

# Start Docker services
docker-up:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Dashboard: http://localhost:8501"
	@echo "  - API Docs: http://localhost:8000/docs"

# Stop Docker services
docker-down:
	@echo "🛑 Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

# View logs
docker-logs:
	docker-compose logs -f

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf models/*.pkl models/*.csv models/plots/*
	rm -rf __pycache__ src/__pycache__ api/__pycache__
	rm -rf .pytest_cache .coverage htmlcov
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "✅ Clean complete"

# Run notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook notebook.ipynb --port 8888

# Full setup (install + train + start)
setup: install train
	@echo ""
	@echo "✅ Full setup complete!"
	@echo ""
	@echo "To start services:"
	@echo "  make api       # Start API on port 8000"
	@echo "  make dashboard # Start dashboard on port 8501"
	@echo ""
	@echo "Or use Docker:"
	@echo "  make docker-up"