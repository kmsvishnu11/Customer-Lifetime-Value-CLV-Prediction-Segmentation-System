# 💰 Customer Lifetime Value (CLV) Prediction System

<!-- Badges -->
<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-yellow.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-purple.svg)
![Docker](https://img.shields.io/badge/Docker-24+-cyan.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

</div>

---

## 🎯 Created by **K MOKSHITH SRI VISHNU**

---

## 📋 Problem Statement

### The Business Challenge

Customer acquisition is expensive. Businesses spend significant resources acquiring new customers, but **80% of their revenue often comes from just 20% of their customers**. The critical question is: **Which customers are worth investing in, and how much should we invest?**

Understanding Customer Lifetime Value (CLV) helps businesses:

- **Optimize Marketing Budget** - Allocate resources to high-value customers who will generate the most revenue
- **Reduce Churn** - Identify customers likely to stop purchasing and proactively engage them
- **Improve ROI** - Calculate how much to spend on acquiring each customer (CAC vs CLV analysis)
- **Personalize Experiences** - Tailor offers and communications based on customer value potential
- **Predict Revenue** - Forecast future revenue based on customer behavior patterns

### Why Online Retail II Dataset?

The Online Retail II dataset from UCI Machine Learning Repository contains **over 1 million transactions** from a UK-based online retailer spanning two years (2009-2011). This dataset is ideal for CLV analysis because:

- **Rich Transaction History** - Multiple purchases per customer over time
- **Date Information** - Enables recency, frequency, and tenure calculations
- **Monetary Values** - Supports revenue-based analysis
- **Geographic Diversity** - Customers from multiple countries
- **Product Diversity** - Multiple product categories

### The Challenge

Traditional approaches to customer value analysis often fail because they:

1. **Only look at historical revenue** - Don't predict future behavior
2. **Use simple heuristics** - Don't account for customer "death" (churn)
3. **Treat all customers equally** - Don't segment by value potential
4. **Lack uncertainty quantification** - No confidence intervals for predictions

---

## 💡 How the Problem is Solved (Under the Hood)

### Our Solution: A Multi-Model Ensemble Approach

We solve the CLV prediction problem by combining **probabilistic models** with **machine learning** to get the best of both worlds:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLV PREDICTION SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────────┐
                              │   RAW TRANSACTION    │
                              │        DATA          │
                              │  (1M+ records)       │
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
           ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
           │   PREPROCESSING │  │      RFM        │  │    COHORT       │
           │                 │  │   ANALYSIS      │  │    ANALYSIS     │
           │ • Remove cancelled│  │                 │  │                 │
           │ • Filter outliers│  │ • Recency       │  │ • Retention     │
           │ • Clean missing  │  │ • Frequency     │  │   Matrix        │
           │ • Feature eng.   │  │ • Monetary      │  │ • Monthly       │
           │                 │  │                 │  │   Cohorts       │
           └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │          CUSTOMER FEATURES              │
                    │  • frequency (repeat purchases)         │
                    │  • recency (days since last purchase)   │
                    │  • T (customer age/days since first)    │
                    │  • monetary_value (avg transaction)     │
                    │  • n_unique_products                    │
                    │  • country, segment, cohort_index       │
                    └─────────────────────┬───────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│   BG/NBD MODEL          │  │   GAMMA-GAMMA MODEL     │  │   FEATURE ENGINEERING   │
│   (Probabilistic)       │  │   (Probabilistic)       │  │   (ML Pipeline)         │
│                         │  │                         │  │                         │
│ • Predicts purchase     │  │ • Predicts avg          │  │ • Combines BG/NBD       │
│   frequency             │  │   transaction value     │  │   predictions as        │
│ • Calculates probability│  │ • Accounts for          │  │   features              │
│   customer is "alive"   │  │   heterogeneity         │  │ • Encodes categorical   │
│ • Based on Beta-        │  │ • Uses Gamma            │  │ • Scales numerical      │
│   Geometric assumption  │  │   distribution          │  │ • Prepares for ML       │
└────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
             │                            │                            │
             ▼                            ▼                            ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│ • Predicted purchases   │  │ • Predicted avg profit  │  │ • XGBoost               │
│   in next 90 days       │  │ • Per transaction       │  │ • LightGBM              │
│ • Probability alive     │  │                         │  │ • Quantile regression   │
│   (P(alive))            │  │                         │  │   for intervals         │
└────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
             │                            │                            │
             └────────────────────────────┼────────────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────┐
                         │      CLV COMBINATION        │
                         │                             │
                         │ CLV = Σ (Predicted Purchases │
                         │         × Avg Profit × Time)│
                         │                             │
                         │ • Probabilistic CLV         │
                         │ • ML-based CLV              │
                         │ • 80% Prediction Intervals  │
                         └─────────────┬───────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│      FASTAPI API        │  │    STREAMLIT APP        │  │      BUSINESS           │
│                         │  │                         │  │      INSIGHTS           │
│ • /predict/clv          │  │ • Customer Lookup       │  │                         │
│ • /segments             │  │ • CLV Predictor         │  │ • VIP Customer List     │
│ • /top-customers        │  │ • Segment Overview      │  │ • At-Risk Detection     │
│ • /at-risk              │  │ • Cohort Analysis       │  │ • Win-Back Campaigns    │
│ • /actions/*            │  │ • Model Performance     │  │ • Marketing Allocation  │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
```

### Model 1: BG/NBD (Beta-Geometric Negative Binomial Distribution)

**What it does:** Predicts how many purchases a customer will make and whether they're still "alive" (active).

**How it works under the hood:**

1. **Transaction Assumption**: While a customer is active, their purchases follow a Poisson process with rate λ (random, independent transactions)

2. **Dropout Assumption**: After any purchase, the customer has probability `p` of becoming inactive (dropping out)

3. **Geometry**: The time until dropout follows a Geometric distribution

4. **Parameter Estimation**: Uses Maximum Likelihood Estimation (MLE) with scipy optimization to fit:
   - `r`, `α` - parameters of the Gamma distribution for λ (heterogeneity in purchase rates)
   - `a`, `b` - parameters of the Beta distribution for `p` (heterogeneity in dropout probability)

5. **Prediction Formula**:
   ```
   E[Purchases in t days] = (r + x) / (r + α) × [1 - ((α + T) / (α + T + t))^(r + x)]
   
   Where:
   - x = frequency (repeat purchases)
   - T = customer age
   - t = prediction period
   - r, α, a, b = fitted parameters
   ```

6. **Alive Probability**:
   ```
   P(alive | x, t_x, T) = 1 / (1 + δ)
   
   where δ depends on parameters and customer history
   ```

### Model 2: Gamma-Gamma Model

**What it does:** Predicts the average transaction value for each customer.

**How it works under the hood:**

1. **Assumption**: Transaction values follow a Gamma distribution with shape `p` and scale `ν`

2. **Heterogeneity**: Different customers have different (p, ν) pairs, following another Gamma distribution

3. **Conditional Expectation**: Given a customer's transaction history, the expected average transaction value is:
   ```
   E[M | x, m] = (p × mean(m) + x × m) / (p + x)
   
   Where:
   - x = frequency
   - m = observed average transaction value
   - p, q, v = fitted parameters
   ```

4. **This model only applies to repeat customers** (frequency > 0)

### Model 3: XGBoost ML Ensemble

**What it does:** Learns complex non-linear relationships between customer features and future revenue.

**How it works under the hood:**

1. **Feature Engineering**:
   - Raw features: frequency, recency, T, monetary_value, n_unique_products, etc.
   - Model predictions: BG/NBD purchases, probability alive, Gamma-Gamma monetary
   - RFM scores: R_score, F_score, M_score
   - Encoded categorical: country (top N + "Other")

2. **Hyperparameter Tuning with Optuna**:
   - 20 trials of Bayesian optimization
   - Minimizes MAE on validation set
   - Searches: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, etc.

3. **Gradient Boosting**:
   - Builds trees sequentially
   - Each tree corrects errors of previous trees
   - Uses gradient descent to minimize loss function
   - Formula: F_m(x) = F_{m-1}(x) + η × h_m(x)
   
4. **Quantile Regression for Intervals**:
   - Trains separate models for 10th and 90th percentiles
   - Objective: `reg:quantileerror` with `quantile_alpha`
   - Provides 80% prediction intervals

5. **Feature Importance**:
   - SHAP values for model interpretability
   - Shows which features drive predictions

### Combining All Models

**Final CLV Calculation:**

```
Probabilistic CLV (12 months) =
    Monthly_Purchases × Avg_Profit × Time_Discount
    
where:
    Monthly_Purchases = BG/NBD_Predicted_90d / 90 × 30
    Avg_Profit = Gamma-Gamma_Predicted_Monetary × Margin (10%)
    Time_Discount = Σ (1 / (1 + r)^t) for t = 1 to 12 months
```

**ML CLV** = XGBoost predicts directly on holdout revenue

**Final Prediction** = Weighted average or use ML prediction (typically better)

---

## 📊 Key Results & Metrics

### Training Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | £864.00 | Mean Absolute Error on holdout |
| **RMSE** | £4,303.56 | Root Mean Squared Error |
| **R²** | 0.4739 | Variance explained (Ridge baseline) |
| **XGBoost R²** | 0.5483 | Best ML model performance |
| **Holdout Period** | 90 days | Future prediction window |
| **Observation Period** | 365 days | Historical features |

### Segment Distribution

| Segment | Customers | Revenue Share | Avg CLV |
|---------|-----------|---------------|---------|
| **Champions** | 613 (12%) | 52.22% | £11,847 |
| **Loyal** | 1,170 (22%) | 31.22% | £3,711 |
| **Potential Loyalists** | 1,067 (20%) | 7.51% | £979 |
| **Hibernating** | 1,513 (29%) | 4.81% | £442 |
| **At Risk** | 370 (7%) | 2.95% | £1,110 |
| **Needs Attention** | 279 (5%) | 0.71% | £353 |
| **New Customers** | 236 (4%) | 0.58% | £341 |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and navigate to project
cd clv-predictor

# Build and run with Docker Compose
docker-compose up -d

# Access services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (requires dataset in data/online_retail.csv)
python -m src.train --data data/online_retail.csv

# Start API
uvicorn api.main:app --reload

# In another terminal, start Dashboard
streamlit run app.py
```

### Option 3: Use Makefile

```bash
make install    # Install dependencies
make train      # Train models
make api        # Start API
make dashboard  # Start Streamlit
make docker-up  # Start with Docker
```

---

## 📁 Project Structure

```
clv-predictor/
├── data/                      # Data directory
│   └── online_retail.csv     # Online Retail II dataset (1M+ transactions)
├── src/                       # Source modules
│   ├── preprocess.py         # Data loading, cleaning, feature engineering
│   ├── rfm.py                # RFM analysis, scoring, segmentation
│   ├── clv_models.py         # CLV models (BG/NBD, Gamma-Gamma, ML)
│   └── train.py              # End-to-end training pipeline
├── api/                       # FastAPI application
│   └── main.py               # 15+ REST endpoints
├── models/                    # Trained models (generated after training)
│   ├── bgnbd_model.pkl       # BG/NBD probabilistic model
│   ├── gamma_gamma_model.pkl # Gamma-Gamma monetary model
│   ├── ml_model/             # XGBoost ensemble
│   ├── customers_with_predictions.csv
│   ├── rfm_segmentation.csv
│   ├── segment_stats.csv
│   └── plots/                # Generated visualizations
├── app.py                    # Streamlit dashboard (5 tabs)
├── notebook.ipynb            # Jupyter analysis notebook (11 sections)
├── requirements.txt          # Python dependencies
├── Dockerfile                # API container image
├── Dockerfile.dashboard      # Dashboard container image
├── docker-compose.yml        # Multi-service orchestration
├── Makefile                  # CLI shortcuts
├── setup.sh                  # One-command setup
├── test_api.py               # API test suite
├── nginx.conf                # Production reverse proxy
└── README.md                 # This file
```

---

## 📡 API Endpoints

### Core Prediction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and model status |
| `GET` | `/health` | Health check with model loading status |
| `POST` | `/predict/clv` | Predict CLV for a customer |
| `POST` | `/predict/batch` | Batch CLV predictions (multiple customers) |
| `GET` | `/segments` | RFM segment summary with actions |
| `GET` | `/customer/{id}` | Customer profile + full CLV analysis |

### Business Action Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/top-customers` | Top N customers by predicted CLV |
| `GET` | `/at-risk` | High CLV but low probability of being alive |
| `GET` | `/actions/vip` | VIP customers for loyalty programs |
| `GET` | `/actions/churn-risk` | Churn risk customers requiring attention |
| `GET` | `/actions/win-back` | Hibernating high-value customers |
| `GET` | `/actions/nurture` | New high-potential customers |
| `POST` | `/rfm` | Compute RFM from transaction history |

### Example Usage

```bash
# Predict CLV for a customer
curl -X POST "http://localhost:8000/predict/clv" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345",
    "frequency": 5,
    "recency": 30.0,
    "T": 180.0,
    "monetary_value": 150.0,
    "n_unique_products": 20,
    "country": "UK"
  }'

# Response:
{
  "customer_id": "12345",
  "predicted_clv_90d": 127.50,
  "predicted_clv_12m": 1530.00,
  "prob_alive": 0.8523,
  "expected_purchases_90d": 4.25,
  "rfm_segment": "Champions",
  "clv_segment": "High Value",
  "ci_lower": 45.90,
  "ci_upper": 3060.00,
  "recommended_action": "🎁 Loyalty rewards program - VIP treatment"
}

# Get segment summary
curl "http://localhost:8000/segments"

# Get VIP customers
curl "http://localhost:8000/actions/vip?limit=50"
```

---

## 📈 Dashboard Tabs

### Tab 1: Customer Lookup
- Search by Customer ID
- CLV gauge visualization (0 to max CLV)
- Segment badge (VIP/High/Medium/Low)
- Probability alive indicator with color coding
- Expected purchases next 90 days
- RFM scores radar chart
- Prediction interval bar
- Recommended action card (color-coded)

### Tab 2: CLV Predictor (New Customer)
- Input customer features: frequency, recency, T, monetary, country
- Real-time CLV prediction with confidence intervals
- Distribution comparison (where this customer falls)
- Compare to average customer

### Tab 3: Segment Overview
- 7 RFM segment cards with metrics
- Per segment: customer count, % of revenue, avg CLV
- Interactive RFM heatmap (Recency vs Frequency)
- CLV distribution by segment (violin plot)
- Segment action recommendations table
- Revenue at risk summary with progress bar

### Tab 4: Cohort Analysis
- Retention cohort heatmap (monthly)
- Monthly revenue cohort chart
- Average CLV by cohort
- Cohort size over time

### Tab 5: Model Performance
- BG/NBD diagnostic plots
- Actual vs Predicted CLV scatter
- ML model metrics table
- Feature importance chart
- SHAP summary plot (if available)
- Error distribution analysis

---

## 🧠 Model Details

### BG/NBD Model Parameters

```python
# Fitted parameters from training:
r     = 0.700706    # Shape of Gamma for purchase rate
alpha = 1.026672    # Scale of Gamma for purchase rate  
a     = 0.170720    # Beta parameter for dropout probability
b     = 1.051809    # Beta parameter for dropout probability
```

### Gamma-Gamma Model Parameters

```python
# Fitted parameters from training:
p     = 1.212256    # Shape of transaction value distribution
q     = 0.402399    # Scale of heterogeneity distribution
v     = 1.198953    # Mean of transaction value heterogeneity
```

### XGBoost Configuration

```python
# Optuna-tuned hyperparameters (best trial):
n_estimators     = 300
max_depth        = 6
learning_rate    = 0.05
subsample        = 0.8
colsample_bytree = 0.8
min_child_weight = 1
reg_alpha        = 0.1
reg_lambda       = 1.0
```

---

## 🛠️ Configuration

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `observation_days` | 365 | Days of history for features |
| `holdout_days` | 90 | Days to predict (target) |
| `time_horizon` | 12 | Months for CLV calculation |
| `discount_rate` | 0.01 | Monthly discount rate |
| `margin` | 0.10 | Profit margin (10%) |
| `n_trials` | 20 | Optuna optimization trials |

### Model Configuration

| Model | Parameter | Value | Reason |
|-------|-----------|-------|--------|
| BG/NBD | penalizer_coef | 0.1 | Prevents overfitting |
| Gamma-Gamma | penalizer_coef | 0.1 | Ensures convergence |
| XGBoost | n_estimators | 300 | Sufficient for convergence |
| XGBoost | max_depth | 6 | Prevents overfitting |
| XGBoost | learning_rate | 0.05 | Slow learning for stability |

---

## 📦 Dataset Information

**Online Retail II Dataset (UCI)**

| Attribute | Type | Description |
|-----------|------|-------------|
| Invoice | String | Transaction ID (C prefix = cancelled) |
| StockCode | String | Product ID |
| Description | String | Product name |
| Quantity | Integer | Number of items (negative = return) |
| InvoiceDate | DateTime | Transaction timestamp |
| Price | Float | Unit price (£) |
| Customer ID | Float | Unique customer identifier |
| Country | String | Customer's country |

**Source**: [Kaggle - Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

**Statistics**:
- Total Transactions: 1,067,371
- Unique Customers: 5,942
- Date Range: Dec 2009 - Dec 2011 (2 years)
- Countries: 38

---

## 🐛 Troubleshooting

### Models not loading
```bash
# Ensure training completed
python -m src.train --data data/online_retail.csv

# Check models directory
ls -la models/
```

### Dashboard not showing data
```bash
# Verify models are trained
python -c "import pandas as pd; df=pd.read_csv('models/customers_with_predictions.csv'); print(df.shape)"
```

### Docker issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## 📚 Resources & References

### Academic Papers
- [BG/NBD Model: "Counting Your Customers the Easy Way"](https://journals.sagepub.com/doi/10.1509/jmkr.2005.42.4.415)
- [Gamma-Gamma Model: "The Gamma-Gamma Model of Monetary Value"](https://www.tandfonline.com/doi/abs/10.1080/00036846.2013.771596)

### Documentation
- [Lifetimes Library](https://lifetimes.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP Values](https://shap.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)

---

## 🏆 Business Applications

### Marketing Budget Allocation

| Segment | Budget % | Strategy |
|---------|----------|----------|
| Champions | 30% | VIP treatment, loyalty rewards |
| Loyal | 25% | Personalized offers, upselling |
| Potential Loyalists | 20% | Loyalty enrollment, incentives |
| New Customers | 15% | Welcome campaigns, onboarding |
| At Risk | 25% | Win-back campaigns, special discounts |
| Hibernating | 10% | Re-engagement, seasonal offers |
| Lost | 5% | Low-cost reactivation attempts |

### Customer Acquisition ROI

```python
# If CLV:CAC ratio >= 3:1 → Profitable acquisition
# If CLV:CAC ratio 1-3:1 → Break-even or marginal
# If CLV:CAC ratio < 1:1 → Loss-making

Example:
- Average CLV: £3,000
- Typical CAC: £500
- CLV:CAC Ratio: 6:1 ✅ Healthy
```

### Retention Program Targeting

1. **Champions** (52% revenue): VIP loyalty program
2. **At Risk** (370 customers): Urgent retention outreach
3. **Hibernating** (1,513 customers): Re-engagement campaigns

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Credits

**Created by K MOKSHITH SRI VISHNU**

Special thanks to:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the Online Retail II dataset
- [Lifetimes Library](https://lifetimes.readthedocs.io/) for the probabilistic CLV models
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API framework
- [Streamlit](https://streamlit.io/) for the interactive dashboard

---

<div align="center">

**Built by K MOKSHITH SRI VISHNU**

</div>