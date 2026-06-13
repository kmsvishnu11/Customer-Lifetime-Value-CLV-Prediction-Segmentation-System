"""
FastAPI Application for CLV Prediction Service
Provides REST API endpoints for CLV predictions and customer analytics
"""

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

# Import project modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_data, create_customer_features
from src.rfm import compute_rfm, score_rfm, segment_customers, get_segment_action_recommendations
from src.clv_models import (
    BGNBDModel, GammaGammaModel, MLCLVPredictor,
    get_recommended_action
)

# ==================== Pydantic Models ====================

class CustomerFeatures(BaseModel):
    """Customer features for CLV prediction."""
    customer_id: str = Field(..., description="Unique customer identifier")
    frequency: int = Field(..., ge=0, description="Number of repeat purchases")
    recency: float = Field(..., ge=0, description="Days since last purchase")
    T: float = Field(..., gt=0, description="Customer age in days (first to last purchase)")
    monetary_value: float = Field(..., ge=0, description="Average transaction value")
    n_unique_products: int = Field(default=0, ge=0, description="Number of unique products")
    country: str = Field(default="UK", description="Customer country")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "12345",
                "frequency": 5,
                "recency": 30.0,
                "T": 180.0,
                "monetary_value": 150.0,
                "n_unique_products": 20,
                "country": "UK"
            }
        }


class CustomerTransaction(BaseModel):
    """Single transaction."""
    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    amount: float = Field(..., ge=0, description="Transaction amount")


class CustomerHistory(BaseModel):
    """Customer transaction history."""
    customer_id: str
    transactions: List[CustomerTransaction]


class CLVPrediction(BaseModel):
    """CLV prediction response."""
    customer_id: str
    predicted_clv_90d: float = Field(description="Predicted 90-day revenue")
    predicted_clv_12m: float = Field(description="Predicted 12-month CLV")
    prob_alive: float = Field(description="Probability customer is still active")
    expected_purchases_90d: float = Field(description="Expected purchases in next 90 days")
    rfm_segment: str = Field(description="RFM segment classification")
    clv_segment: str = Field(description="CLV-based segment (VIP/High/Medium/Low)")
    ci_lower: float = Field(description="80% prediction interval lower bound")
    ci_upper: float = Field(description="80% prediction interval upper bound")
    recommended_action: str = Field(description="Recommended business action")
    bgnbd_clv: Optional[float] = Field(default=None, description="Probabilistic CLV estimate")


class SegmentSummary(BaseModel):
    """RFM segment summary."""
    segment: str
    customer_count: int
    revenue_share_pct: float
    avg_clv: float
    priority: str
    recommended_action: str


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str
    mae: float
    rmse: float
    mape: Optional[float]
    r2: Optional[float]


# ==================== FastAPI App ====================

app = FastAPI(
    title="CLV Prediction API",
    description="Customer Lifetime Value Prediction and Segmentation Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global State ====================

# Will be loaded on startup
models = {
    'bgnbd': None,
    'gamma_gamma': None,
    'ml': None
}
customers_df = None
rfm_df = None
segment_stats = None


def load_models():
    """Load all models from disk."""
    global models, customers_df, rfm_df, segment_stats
    
    models_dir = "models"
    
    # Check if models exist
    if not os.path.exists(f"{models_dir}/bgnbd_model.pkl"):
        print("⚠️ Models not found. Run train.py first or use demo mode.")
        return False
    
    try:
        # Load BG/NBD
        bgnbd = BGNBDModel()
        bgnbd.load(f"{models_dir}/bgnbd_model.pkl")
        models['bgnbd'] = bgnbd
        
        # Load Gamma-Gamma
        gg = GammaGammaModel()
        gg.load(f"{models_dir}/gamma_gamma_model.pkl")
        models['gamma_gamma'] = gg
        
        # Load ML model
        ml = MLCLVPredictor()
        ml.load(f"{models_dir}/ml_model")
        models['ml'] = ml
        
        # Load customer data
        customers_df = pd.read_csv(f"{models_dir}/customers_with_predictions.csv")
        
        # Load RFM data
        rfm_df = pd.read_csv(f"{models_dir}/rfm_segmentation.csv")
        
        # Load segment stats
        segment_stats = pd.read_csv(f"{models_dir}/segment_stats.csv")
        
        print("✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


def compute_customer_features_from_history(history: CustomerHistory) -> pd.DataFrame:
    """Compute customer features from transaction history."""
    transactions = pd.DataFrame([{
        'InvoiceDate': pd.to_datetime(t.date),
        'Revenue': t.amount,
        'Customer ID': history.customer_id
    } for t in history.transactions])
    
    if len(transactions) == 0:
        return None
    
    features = create_customer_features(transactions)
    features['Customer ID'] = history.customer_id
    
    return features


def predict_clv_for_customer(customer_id: str, features: Optional[pd.DataFrame] = None) -> CLVPrediction:
    """Generate CLV prediction for a customer."""
    
    if features is not None and len(features) > 0:
        # Use provided features (new customer)
        customer_row = features.iloc[0]
        frequency = float(customer_row.get('frequency', 0))
        recency = float(customer_row.get('recency', 0))
        T = float(customer_row.get('T', 30))
        monetary_value = float(customer_row.get('monetary_value', 0))
    elif customers_df is not None:
        # Look up from loaded data
        customer_row = customers_df[customers_df['Customer ID'] == float(customer_id)]
        if len(customer_row) == 0:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        customer_row = customer_row.iloc[0]
        frequency = float(customer_row.get('frequency', 0))
        recency = float(customer_row.get('recency', 0))
        T = float(customer_row.get('T', 30))
        monetary_value = float(customer_row.get('monetary_value', 0))
    else:
        raise HTTPException(status_code=400, detail="Customer data not loaded")
    
    # Ensure valid values for model
    frequency = max(0, frequency)
    recency = max(0, recency)
    T = max(1, T)
    monetary_value = max(0, monetary_value)
    
    # Handle edge cases for BG/NBD
    if frequency > 0 and recency > 0 and recency <= T:
        try:
            bgnbd_pred = models['bgnbd'].predict_purchases(
                pd.DataFrame([{
                    'Customer ID': customer_id,
                    'frequency': frequency,
                    'recency': recency,
                    'T': T
                }]),
                t=90
            ).values[0]
            prob_alive = models['bgnbd'].predict_alive_probability(
                pd.DataFrame([{
                    'Customer ID': customer_id,
                    'frequency': frequency,
                    'recency': recency,
                    'T': T
                }])
            ).values[0]
        except:
            # Fallback for model prediction errors
            bgnbd_pred = frequency * 90 / T if T > 0 else frequency
            prob_alive = 0.5
    else:
        # New/one-time buyers
        bgnbd_pred = monetary_value * 0.1 * 90 / 365 if monetary_value > 0 else 50
        prob_alive = 0.8 if frequency == 0 else 0.5
    
    # Gamma-Gamma monetary
    if monetary_value > 0:
        gamma_gamma_monetary = monetary_value * 0.10  # 10% margin
    else:
        gamma_gamma_monetary = 10.0
    
    # 12-month CLV estimate
    monthly_purchases = bgnbd_pred / 90 * 30
    clv_12m = monthly_purchases * gamma_gamma_monetary * 12
    
    # Clamp values to reasonable ranges
    bgnbd_pred = max(0, min(bgnbd_pred, 10000))
    prob_alive = max(0, min(prob_alive, 1))
    clv_12m = max(0, min(clv_12m, 100000))
    
    # Use probabilistic CLV as prediction
    ml_pred = clv_12m
    ci_lower = clv_12m * 0.3
    ci_upper = clv_12m * 2.0
    
    # RFM segment (for new customers, estimate based on features)
    if features is not None:
        # Estimate segment for new customer
        if recency <= 30 and frequency >= 5 and monetary_value >= 100:
            rfm_segment = 'Champions'
        elif frequency >= 5:
            rfm_segment = 'Loyal'
        elif recency <= 60 and frequency >= 2:
            rfm_segment = 'Potential Loyalists'
        elif recency <= 30 and frequency <= 1:
            rfm_segment = 'New Customers'
        elif recency >= 180 and frequency >= 3:
            rfm_segment = 'At Risk'
        elif recency >= 180 and frequency <= 2:
            rfm_segment = 'Hibernating'
        else:
            rfm_segment = 'Needs Attention'
    else:
        rfm_row = rfm_df[rfm_df['Customer ID'] == float(customer_id)] if rfm_df is not None else None
        rfm_segment = rfm_row.iloc[0]['segment'] if rfm_row is not None and len(rfm_row) > 0 else 'Unknown'
    
    # CLV segment based on predicted value
    if clv_12m >= 5000:
        clv_segment = 'VIP'
    elif clv_12m >= 1000:
        clv_segment = 'High Value'
    elif clv_12m >= 200:
        clv_segment = 'Medium Value'
    else:
        clv_segment = 'Low Value'
    
    # Recommended action
    recommended_action = get_recommended_action(rfm_segment, prob_alive, ml_pred)
    
    return CLVPrediction(
        customer_id=str(customer_id),
        predicted_clv_90d=round(ml_pred, 2),
        predicted_clv_12m=round(clv_12m, 2),
        prob_alive=round(prob_alive, 4),
        expected_purchases_90d=round(bgnbd_pred, 2),
        rfm_segment=rfm_segment,
        clv_segment=clv_segment,
        ci_lower=round(max(0, ci_lower), 2),
        ci_upper=round(ci_upper, 2),
        recommended_action=recommended_action,
        bgnbd_clv=round(clv_12m, 2)
    )


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.get("/", tags=["Info"])
async def root():
    """API info and model metrics."""
    return {
        "name": "CLV Prediction API",
        "version": "1.0.0",
        "description": "Customer Lifetime Value Prediction and Segmentation Service",
        "models_loaded": models['bgnbd'] is not None,
        "total_customers": len(customers_df) if customers_df is not None else 0,
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "POST /predict/clv": "Predict CLV for a customer",
            "POST /predict/batch": "Batch CLV predictions",
            "GET /segments": "RFM segments summary",
            "GET /customer/{id}": "Customer profile + CLV",
            "GET /top-customers": "Top 100 customers by predicted CLV",
            "GET /at-risk": "At-risk customers (high CLV, decreasing prob_alive)",
            "POST /rfm": "Compute RFM for transaction history"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "bgnbd": models['bgnbd'] is not None,
            "gamma_gamma": models['gamma_gamma'] is not None,
            "ml": models['ml'] is not None
        },
        "customers_loaded": customers_df is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/clv", response_model=CLVPrediction, tags=["Prediction"])
async def predict_clv(customer: CustomerFeatures):
    """Predict CLV for a customer with provided features."""
    
    if models['bgnbd'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")
    
    # Create customer dataframe
    customer_df = pd.DataFrame([{
        'Customer ID': customer.customer_id,
        'frequency': customer.frequency,
        'recency': customer.recency,
        'T': customer.T,
        'monetary_value': customer.monetary_value,
        'n_unique_products': customer.n_unique_products,
        'country': customer.country
    }])
    
    return predict_clv_for_customer(customer.customer_id, customer_df)


@app.post("/predict/batch", response_model=List[CLVPrediction], tags=["Prediction"])
async def predict_batch(customers: List[CustomerFeatures]):
    """Batch CLV predictions for multiple customers."""
    
    if models['bgnbd'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    
    predictions = []
    
    for customer in customers:
        try:
            pred = await predict_clv(customer)
            predictions.append(pred)
        except Exception as e:
            # Skip failed predictions
            continue
    
    return predictions


@app.get("/segments", response_model=List[SegmentSummary], tags=["Segments"])
async def get_segments():
    """Get RFM segment summary."""
    
    if segment_stats is None:
        raise HTTPException(status_code=503, detail="Segment statistics not loaded.")
    
    actions = get_segment_action_recommendations()
    
    segments = []
    for _, row in segment_stats.iterrows():
        segment_name = row['segment']
        action = actions.get(segment_name, {})
        
        segments.append(SegmentSummary(
            segment=segment_name,
            customer_count=int(row['customer_count']),
            revenue_share_pct=round(row['revenue_share_pct'], 2),
            avg_clv=round(row['avg_revenue_per_customer'], 2),
            priority=action.get('priority', 'Medium'),
            recommended_action=action.get('action', 'Standard engagement')
        ))
    
    return segments


@app.get("/customer/{customer_id}", response_model=CLVPrediction, tags=["Customer"])
async def get_customer(customer_id: str = Path(..., description="Customer ID")):
    """Get full customer profile with CLV prediction."""
    
    if models['bgnbd'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded.")
    
    return predict_clv_for_customer(customer_id)


@app.get("/top-customers", tags=["Business"])
async def get_top_customers(limit: int = 100):
    """Get top customers by predicted CLV."""
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded.")
    
    # Sort by predicted CLV
    if 'bgnbd_gg_clv' in customers_df.columns:
        top = customers_df.nlargest(limit, 'bgnbd_gg_clv')
    else:
        raise HTTPException(status_code=503, detail="CLV predictions not available.")
    
    return [
        {
            "customer_id": str(row['Customer ID']),
            "predicted_clv": round(row['bgnbd_gg_clv'], 2),
            "segment": row.get('segment', 'Unknown'),
            "prob_alive": round(row.get('prob_alive', 0), 4),
            "total_revenue_obs": round(row.get('total_revenue_obs', 0), 2)
        }
        for _, row in top.iterrows()
    ]


@app.get("/at-risk", tags=["Business"])
async def get_at_risk(limit: int = 50):
    """Get at-risk customers (high CLV but low probability of being alive)."""
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded.")
    
    # Filter: high CLV potential but low prob_alive
    at_risk = customers_df[
        (customers_df['prob_alive'] < 0.5) &
        (customers_df['bgnbd_gg_clv'] > customers_df['bgnbd_gg_clv'].quantile(0.5))
    ].nlargest(limit, 'bgnbd_gg_clv')
    
    return [
        {
            "customer_id": str(row['Customer ID']),
            "predicted_clv": round(row['bgnbd_gg_clv'], 2),
            "segment": row.get('segment', 'Unknown'),
            "prob_alive": round(row['prob_alive'], 4),
            "recency": round(row['recency'], 1),
            "recommended_action": get_recommended_action(
                row.get('segment', 'Unknown'),
                row['prob_alive'],
                row['bgnbd_gg_clv']
            )
        }
        for _, row in at_risk.iterrows()
    ]


@app.post("/rfm", tags=["RFM"])
async def compute_rfm_analysis(history: CustomerHistory):
    """Compute RFM analysis for customer transaction history."""
    
    features = compute_customer_features_from_history(history)
    
    if features is None or len(features) == 0:
        raise HTTPException(status_code=400, detail="No valid transactions provided")
    
    # Compute RFM
    rfm_data = pd.DataFrame([{
        'Customer ID': history.customer_id,
        'Recency': features.iloc[0]['recency'],
        'Frequency': features.iloc[0]['frequency'] + 1,  # Add 1 since frequency excludes first purchase
        'Monetary': features.iloc[0]['monetary_value']
    }])
    
    rfm_data = score_rfm(rfm_data)
    rfm_data = segment_customers(rfm_data)
    
    segment_actions = get_segment_action_recommendations()
    segment = rfm_data.iloc[0]['segment']
    action = segment_actions.get(segment, {})
    
    return {
        "customer_id": history.customer_id,
        "recency": rfm_data.iloc[0]['Recency'],
        "frequency": rfm_data.iloc[0]['Frequency'],
        "monetary": rfm_data.iloc[0]['Monetary'],
        "r_score": int(rfm_data.iloc[0]['R_score']),
        "f_score": int(rfm_data.iloc[0]['F_score']),
        "m_score": int(rfm_data.iloc[0]['M_score']),
        "rfm_score": int(rfm_data.iloc[0]['RFM_score']),
        "segment": segment,
        "priority": action.get('priority', 'Medium'),
        "recommended_action": action.get('action', 'Standard engagement')
    }


# ==================== Business Action Endpoints ====================

@app.get("/actions/vip", tags=["Business Actions"])
async def get_vip_customers(limit: int = 50):
    """Get VIP customers for loyalty rewards."""
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded.")
    
    # VIP: top CLV with high prob_alive
    vip = customers_df[
        (customers_df['bgnbd_gg_clv'] > customers_df['bgnbd_gg_clv'].quantile(0.9)) &
        (customers_df['prob_alive'] > 0.7)
    ].nlargest(limit, 'bgnbd_gg_clv')
    
    return [
        {
            "customer_id": str(row['Customer ID']),
            "predicted_clv": round(row['bgnbd_gg_clv'], 2),
            "prob_alive": round(row['prob_alive'], 4),
            "segment": row.get('segment', 'Unknown'),
            "action": "Loyalty rewards program - VIP treatment"
        }
        for _, row in vip.iterrows()
    ]


@app.get("/actions/churn-risk", tags=["Business Actions"])
async def get_churn_risk(limit: int = 50):
    """Get customers at high churn risk (high CLV but low prob_alive)."""
    
    return get_at_risk(limit)


@app.get("/actions/win-back", tags=["Business Actions"])
async def get_win_back(limit: int = 50):
    """Get hibernating high-value customers for win-back campaigns."""
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded.")
    
    # Hibernating with significant historical CLV
    win_back = customers_df[
        (customers_df['segment'] == 'Hibernating') &
        (customers_df['total_revenue_obs'] > customers_df['total_revenue_obs'].quantile(0.5))
    ].nlargest(limit, 'total_revenue_obs')
    
    return [
        {
            "customer_id": str(row['Customer ID']),
            "historical_revenue": round(row['total_revenue_obs'], 2),
            "last_purchase": row.get('last_purchase_date', 'Unknown'),
            "action": "Win-back campaign - exclusive discount"
        }
        for _, row in win_back.iterrows()
    ]


@app.get("/actions/nurture", tags=["Business Actions"])
async def get_nurture(limit: int = 50):
    """Get new high-potential customers for nurturing campaigns."""
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded.")
    
    # New customers with good monetary value
    nurture = customers_df[
        (customers_df['segment'] == 'New Customers') &
        (customers_df['monetary_value'] > customers_df['monetary_value'].quantile(0.5))
    ].nlargest(limit, 'monetary_value')
    
    return [
        {
            "customer_id": str(row['Customer ID']),
            "avg_order_value": round(row['monetary_value'], 2),
            "n_unique_products": int(row.get('n_unique_products', 0)),
            "action": "Onboarding campaign - welcome sequence"
        }
        for _, row in nurture.iterrows()
    ]


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)