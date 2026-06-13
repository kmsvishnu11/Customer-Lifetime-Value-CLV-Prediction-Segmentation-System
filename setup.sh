#!/bin/bash
# ============================================
# CLV Predictor - Setup and Test Script
# ============================================

set -e

echo "=========================================="
echo "🚀 CLV Predictor Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python --version || { echo "Python not found!"; exit 1; }

# Create directories
echo "Creating directories..."
mkdir -p data
mkdir -p models/plots

# Install dependencies
echo "Installing Python dependencies..."
pip install --quiet -r requirements.txt

# Verify installations
echo "Verifying installations..."
python -c "
import pandas
import numpy
import lifetimes
import xgboost
import lightgbm
import fastapi
import streamlit
import plotly
print('✅ All packages installed successfully')
"

# Check if data exists
if [ -f "data/online_retail.csv" ]; then
    echo "✅ Dataset found at data/online_retail.csv"
    
    # Run training
    echo ""
    echo "=========================================="
    echo "📊 Training CLV Models"
    echo "=========================================="
    python -m src.train --data data/online_retail.csv --output models
    
    echo ""
    echo "=========================================="
    echo "✅ Training Complete!"
    echo "=========================================="
else
    echo "⚠️ Dataset not found at data/online_retail.csv"
    echo ""
    echo "Please download the Online Retail II dataset:"
    echo "1. Go to: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci"
    echo "2. Download the dataset"
    echo "3. Place it as data/online_retail.csv"
    echo "4. Run: python -m src.train"
fi

echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "To start the API:"
echo "  uvicorn api.main:app --reload"
echo ""
echo "To start the Dashboard:"
echo "  streamlit run app.py"
echo ""
echo "To start with Docker:"
echo "  docker-compose up -d"
echo ""