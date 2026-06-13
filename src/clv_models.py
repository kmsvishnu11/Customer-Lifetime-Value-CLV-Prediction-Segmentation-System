"""
CLV Models Module
Implements BG/NBD, Gamma-Gamma probabilistic models and ML-based CLV prediction
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Probabilistic CLV Models
from lifetimes import BetaGeoFitter, GammaGammaFitter

# ML Models
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hyperparameter Optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Explainability
import shap

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Serialization
import joblib


class BGNBDModel:
    """
    BG/NBD (Beta-Geometric Negative Binomial Distribution) Model
    for predicting purchase frequency and customer activity probability.
    
    The BG/NBD model assumes:
    - While active, number of transactions follows a Poisson process
    - After any transaction, customer becomes inactive with probability p
    - Dropout time follows a geometric distribution
    """
    
    def __init__(self, penalizer_coef: float = 0.01):
        self.penalizer_coef = penalizer_coef
        self.model = None
        self.fitted = False
    
    def fit(self, customers_df: pd.DataFrame) -> 'BGNBDModel':
        """
        Fit the BG/NBD model.
        
        Parameters:
        -----------
        customers_df : pd.DataFrame
            Customer features with columns: frequency, recency, T
            
        Returns:
        --------
        self
        """
        self.model = BetaGeoFitter(penalizer_coef=self.penalizer_coef)
        
        # Prepare data
        frequency = customers_df['frequency'].values
        recency = customers_df['recency'].values
        T = customers_df['T'].values
        
        # Filter out invalid data
        valid_mask = (frequency >= 0) & (recency >= 0) & (T > 0)
        
        if valid_mask.sum() == 0:
            raise ValueError("No valid customers for BG/NBD fitting")
        
        # Fit the model
        self.model.fit(
            frequency[valid_mask],
            recency[valid_mask],
            T[valid_mask]
        )
        
        self.fitted = True
        print(f"BG/NBD model fitted. Parameters: {self.model.params_}")
        
        return self
    
    def predict_purchases(self, customers_df: pd.DataFrame, t: int = 90) -> pd.Series:
        """
        Predict expected number of purchases in next t days.
        
        Parameters:
        -----------
        customers_df : pd.DataFrame
            Customer features
        t : int
            Number of days to predict
            
        Returns:
        --------
        pd.Series
            Customer ID to predicted purchases
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        frequency = customers_df['frequency'].values
        recency = customers_df['recency'].values
        T = customers_df['T'].values
        
        predictions = self.model.conditional_expected_number_of_purchases_up_to_time(
            t, frequency, recency, T
        )
        
        return pd.Series(predictions, index=customers_df['Customer ID'])
    
    def predict_alive_probability(self, customers_df: pd.DataFrame) -> pd.Series:
        """
        Predict probability customer is still active.
        
        Parameters:
        -----------
        customers_df : pd.DataFrame
            Customer features
            
        Returns:
        --------
        pd.Series
            Customer ID to probability of being alive
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        frequency = customers_df['frequency'].values
        recency = customers_df['recency'].values
        T = customers_df['T'].values
        
        probabilities = self.model.conditional_probability_alive(
            frequency, recency, T
        )
        
        return pd.Series(probabilities, index=customers_df['Customer ID'])
    
    def plot_frequency_recency(self) -> plt.Figure:
        """
        Create frequency vs recency visualization from fitted model.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        
        fig = plt.figure(figsize=(12, 8))
        
        # Create a grid of frequency and recency values
        personal_frequency = np.arange(0, 50, 1)
        personal_recency = np.arange(0, 365, 5)
        
        Z = np.zeros((len(personal_recency), len(personal_frequency)))
        
        for i, recency in enumerate(personal_recency):
            for j, freq in enumerate(personal_frequency):
                if freq > 0:
                    Z[i, j] = self.model.conditional_probability_alive(freq, recency, recency + 1)
        
        plt.imshow(Z, cmap='viridis', aspect='auto', 
                   extent=[0, 50, 365, 0], origin='lower')
        plt.colorbar(label='Probability Alive')
        plt.xlabel('Frequency')
        plt.ylabel('Recency (days)')
        plt.title('BG/NBD: Probability Customer is Alive\n(Higher frequency + recent = more likely alive)')
        
        return fig
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.fitted = True


class GammaGammaModel:
    """
    Gamma-Gamma Model for predicting average transaction value.
    
    Assumes:
    - Transaction values follow a gamma distribution
    - The shape parameter varies across customers
    """
    
    def __init__(self, penalizer_coef: float = 0.01):
        self.penalizer_coef = penalizer_coef
        self.model = None
        self.fitted = False
    
    def fit(self, customers_df: pd.DataFrame) -> 'GammaGammaModel':
        """
        Fit the Gamma-Gamma model.
        
        Note: Requires customers with frequency > 0 (repeat purchases)
        
        Parameters:
        -----------
        customers_df : pd.DataFrame
            Customer features with columns: frequency, monetary_value
            
        Returns:
        --------
        self
        """
        # Filter to customers with repeat purchases
        repeat_customers = customers_df[customers_df['frequency'] > 0].copy()
        
        if len(repeat_customers) < 10:
            raise ValueError(f"Need at least 10 repeat customers, got {len(repeat_customers)}")
        
        self.model = GammaGammaFitter(penalizer_coef=self.penalizer_coef)
        
        # Fit on frequency and average transaction value
        self.model.fit(
            repeat_customers['frequency'],
            repeat_customers['monetary_value']
        )
        
        self.fitted = True
        print(f"Gamma-Gamma model fitted. Parameters: {self.model.params_}")
        
        return self
    
    def predict_average_profit(
        self, 
        customers_df: pd.DataFrame, 
        margin: float = 0.10
    ) -> pd.Series:
        """
        Predict expected average transaction value with margin applied.
        
        Parameters:
        -----------
        customers_df : pd.DataFrame
            Customer features
        margin : float
            Profit margin factor (default: 0.10 = 10%)
            
        Returns:
        --------
        pd.Series
            Customer ID to predicted average profit
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get conditional expected average transaction value
        predicted_value = self.model.conditional_expected_average_profit(
            customers_df['frequency'],
            customers_df['monetary_value']
        )
        
        # Apply margin
        predicted_profit = predicted_value * margin
        
        return pd.Series(predicted_profit, index=customers_df['Customer ID'])
    
    def compute_clv(
        self,
        bgnbd_model: BGNBDModel,
        customers_df: pd.DataFrame,
        time_months: int = 12,
        discount_rate: float = 0.01
    ) -> pd.Series:
        """
        Compute expected CLV using both BG/NBD and Gamma-Gamma models.
        
        CLV = Σ (discounted_purchases * discounted_profit)
        
        Parameters:
        -----------
        bgnbd_model : BGNBDModel
            Fitted BG/NBD model
        customers_df : pd.DataFrame
            Customer features
        time_months : int
            Time horizon in months
        discount_rate : float
            Monthly discount rate
            
        Returns:
        --------
        pd.Series
            Customer ID to predicted CLV
        """
        if not self.fitted:
            raise RuntimeError("Gamma-Gamma model not fitted.")
        
        # Get expected purchases per month from BG/NBD
        monthly_purchases = bgnbd_model.predict_purchases(customers_df, t=30)  # per month
        
        # Get expected profit per transaction from Gamma-Gamma
        monthly_profit = self.predict_average_profit(customers_df)
        
        # Calculate CLV with discounting
        clv = monthly_purchases * monthly_profit
        
        # Apply time value of money
        discount_factor = (1 + discount_rate) ** np.arange(time_months)
        time_discount = np.sum(1 / discount_factor)
        
        clv = clv * time_discount
        
        return clv
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.fitted = True


class MLCLVPredictor:
    """
    ML-based CLV Predictor using gradient boosting with uncertainty estimation.
    
    Features:
    - XGBoost with Optuna hyperparameter tuning
    - Quantile regression for prediction intervals
    - SHAP explainability
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 20
    ):
        self.random_state = random_state
        self.n_trials = n_trials
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.trained = False
        
    def prepare_features(
        self,
        customers_df: pd.DataFrame,
        bgnbd_predictions: Optional[pd.Series] = None,
        prob_alive: Optional[pd.Series] = None,
        gamma_gamma_monetary: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Prepare feature matrix for ML model.
        
        Parameters:
        -----------
        customers_df : pd.DataFrame
            Customer features dataframe
        bgnbd_predictions : pd.Series, optional
            BG/NBD predicted purchases
        prob_alive : pd.Series, optional
            Probability customer is alive
        gamma_gamma_monetary : pd.Series, optional
            Gamma-Gamma predicted monetary value
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        features_df = customers_df.copy()
        
        # Add BG/NBD features
        if bgnbd_predictions is not None:
            features_df['bgnbd_predicted_purchases'] = features_df['Customer ID'].map(bgnbd_predictions)
        else:
            features_df['bgnbd_predicted_purchases'] = 0
            
        if prob_alive is not None:
            features_df['prob_alive'] = features_df['Customer ID'].map(prob_alive)
        else:
            features_df['prob_alive'] = 1.0
            
        if gamma_gamma_monetary is not None:
            features_df['gamma_gamma_monetary'] = features_df['Customer ID'].map(gamma_gamma_monetary)
        else:
            features_df['gamma_gamma_monetary'] = features_df['monetary_value']
        
        # Encode country
        top_countries = features_df['country'].value_counts().head(10).index.tolist()
        features_df['country_encoded'] = features_df['country'].apply(
            lambda x: x if x in top_countries else 'Other'
        )
        
        if 'country_encoded' not in self.label_encoders:
            self.label_encoders['country'] = LabelEncoder()
            features_df['country_encoded_num'] = self.label_encoders['country'].fit_transform(
                features_df['country_encoded']
            )
        else:
            # Handle unseen categories
            known = set(self.label_encoders['country'].classes_)
            features_df['country_encoded'] = features_df['country_encoded'].apply(
                lambda x: x if x in known else 'Other'
            )
            features_df['country_encoded_num'] = self.label_encoders['country'].transform(
                features_df['country_encoded']
            )
        
        # Select feature columns
        self.feature_names = [
            'frequency', 'recency', 'T', 'monetary_value', 'total_revenue_obs',
            'avg_revenue_per_txn', 'std_revenue_per_txn', 'n_unique_products',
            'n_unique_months', 'avg_days_between_purchases', 'max_purchase_gap',
            'n_transactions', 'bgnbd_predicted_purchases', 'prob_alive',
            'gamma_gamma_monetary', 'country_encoded_num'
        ]
        
        X = features_df[self.feature_names].fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        return X
    
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for XGBoost."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        
        return mae
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'MLCLVPredictor':
        """
        Train ML models (XGBoost, LightGBM, Ridge).
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : np.ndarray
            Training targets
            
        Returns:
        --------
        self
        """
        from sklearn.model_selection import train_test_split
        
        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        X_val_scaled = self.scaler.transform(X_val)
        
        print("Training XGBoost with Optuna hyperparameter tuning...")
        
        # Optuna optimization for XGBoost
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(
            lambda trial: self._objective(trial, X_tr_scaled, y_tr, X_val_scaled, y_val),
            n_trials=self.n_trials,
            show_progress_bar=False
        )
        
        best_params = study.best_params
        print(f"Best XGBoost MAE: {study.best_value:.4f}")
        
        # Train XGBoost with best params
        self.models['xgboost'] = xgb.XGBRegressor(
            **best_params,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['xgboost'].fit(X_tr_scaled, y_tr)
        
        # Train LightGBM
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.random_state,
            verbose=-1
        )
        self.models['lightgbm'].fit(X_tr_scaled, y_tr)
        
        # Train Ridge (baseline)
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_tr_scaled, y_tr)
        
        # Train quantile models for prediction intervals
        print("Training quantile models for prediction intervals...")
        self.models['xgboost_lower'] = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.10,
            **best_params,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['xgboost_lower'].fit(X_tr_scaled, y_tr)
        
        self.models['xgboost_upper'] = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.90,
            **best_params,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['xgboost_upper'].fit(X_tr_scaled, y_tr)
        
        self.best_model = 'xgboost'
        self.trained = True
        
        print("ML models trained successfully!")
        
        return self
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """
        Evaluate models on test set.
        
        Returns:
        --------
        dict
            Metrics including MAE, RMSE, MAPE, R2, and feature importance
        """
        if not self.trained:
            raise RuntimeError("Models not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            if name in ['xgboost_lower', 'xgboost_upper']:
                continue
                
            preds = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            # MAPE (handle zeros)
            non_zero_mask = y_test > 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_test[non_zero_mask] - preds[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                mape = np.nan
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
        
        # Compute prediction intervals
        lower = self.models['xgboost_lower'].predict(X_test_scaled)
        upper = self.models['xgboost_upper'].predict(X_test_scaled)
        
        interval_coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100
        results['interval_coverage_80'] = interval_coverage
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict CLV with prediction intervals.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            predictions, lower_bound, upper_bound
        """
        if not self.trained:
            raise RuntimeError("Models not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Point predictions (use best model)
        predictions = self.models[self.best_model].predict(X_scaled)
        
        # Prediction intervals (quantile regression)
        lower = self.models['xgboost_lower'].predict(X_scaled)
        upper = self.models['xgboost_upper'].predict(X_scaled)
        
        # Ensure lower <= predictions <= upper
        lower = np.minimum(lower, predictions)
        upper = np.maximum(upper, predictions)
        
        return predictions, lower, upper
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        if not self.trained:
            raise RuntimeError("Models not trained.")
        
        importance = self.models['xgboost'].feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def plot_shap(self, X_sample: pd.DataFrame) -> plt.Figure:
        """Create SHAP summary plot."""
        if not self.trained:
            raise RuntimeError("Models not trained.")
        
        X_scaled = self.scaler.transform(X_sample)
        
        explainer = shap.TreeExplainer(self.models['xgboost'])
        shap_values = explainer.shap_values(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        
        return fig
    
    def segment_by_predicted_clv(self, predictions: np.ndarray) -> np.ndarray:
        """
        Segment customers based on predicted CLV.
        
        - VIP: top 10% (red carpet treatment)
        - High Value: 10-30% (priority support)
        - Medium Value: 30-60% (standard)
        - Low Value: bottom 40% (re-engage campaigns)
        
        Returns:
        --------
        np.ndarray
            Segment labels
        """
        percentiles = np.percentile(predictions, [40, 60, 90])
        
        segments = np.where(
            predictions >= percentiles[1],
            np.where(predictions >= percentiles[2], 'VIP', 'High Value'),
            np.where(predictions >= percentiles[0], 'Medium Value', 'Low Value')
        )
        
        return segments
    
    def save(self, path_prefix: str):
        """Save all models and scalers to disk."""
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
        joblib.dump(self.models, f"{path_prefix}_models.pkl")
        joblib.dump(self.feature_names, f"{path_prefix}_features.pkl")
        joblib.dump(self.label_encoders, f"{path_prefix}_encoders.pkl")
    
    def load(self, path_prefix: str):
        """Load models and scalers from disk."""
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        self.models = joblib.load(f"{path_prefix}_models.pkl")
        self.feature_names = joblib.load(f"{path_prefix}_features.pkl")
        self.label_encoders = joblib.load(f"{path_prefix}_encoders.pkl")
        self.trained = True


def compute_retention_cohorts(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Compute monthly cohort retention matrix.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe with CohortMonth and CohortIndex columns
        
    Returns:
    --------
    Tuple[pd.DataFrame, go.Figure]
        retention_matrix and heatmap figure
    """
    # Get unique customers per cohort and month
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()
    cohort_data.columns = ['CohortMonth', 'CohortIndex', 'n_customers']
    
    # Pivot to get cohort matrix
    cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='n_customers')
    
    # Get cohort sizes (month 0)
    cohort_sizes = cohort_pivot[0]
    
    # Calculate retention percentages
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=retention_matrix.values,
        x=[f'Month {i}' for i in retention_matrix.columns],
        y=[str(idx) for idx in retention_matrix.index],
        colorscale='RdYlGn',
        colorbar_title='Retention %',
        text=np.round(retention_matrix.values, 1),
        texttemplate='%{text:.1f}%',
        hovertemplate='Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monthly Cohort Retention Analysis',
        xaxis_title='Months Since First Purchase',
        yaxis_title='Cohort Month',
        width=900,
        height=600
    )
    
    return retention_matrix, fig


def compare_models(
    bgnbd_clv: pd.Series,
    ml_clv: np.ndarray,
    actual_holdout: Dict[int, float]
) -> pd.DataFrame:
    """
    Compare probabilistic (BG/NBD+GG) vs ML-based CLV predictions.
    
    Parameters:
    -----------
    bgnbd_clv : pd.Series
        BG/NBD+Gamma-Gamma CLV predictions
    ml_clv : np.ndarray
        ML-based CLV predictions
    actual_holdout : Dict[int, float]
        Actual holdout revenue
        
    Returns:
    --------
    pd.DataFrame
        Comparison metrics
    """
    # Convert to comparable format
    customers = list(actual_holdout.keys())
    actual = np.array([actual_holdout.get(c, 0) for c in customers])
    
    # BG/NBD predictions
    bgnbd_pred = np.array([bgnbd_clv.get(c, 0) for c in customers])
    
    # ML predictions (same order)
    
    results = []
    
    for name, preds in [('BG/NBD + Gamma-Gamma', bgnbd_pred), ('ML (XGBoost)', ml_clv)]:
        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        correlation = np.corrcoef(actual, preds)[0, 1]
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation
        })
    
    return pd.DataFrame(results)


def get_recommended_action(segment: str, prob_alive: float, predicted_clv: float) -> str:
    """
    Map segment + probability alive to business action.
    
    Parameters:
    -----------
    segment : str
        RFM segment
    prob_alive : float
        Probability customer is alive
    predicted_clv : float
        Predicted CLV
        
    Returns:
    --------
    str
        Recommended action
    """
    clv_percentile = np.percentile([predicted_clv], 80)  # Simplified
    
    if segment == 'Champions' and prob_alive > 0.8:
        return "🎁 Loyalty rewards program - VIP treatment"
    elif segment == 'Loyal' and prob_alive > 0.7:
        return "💝 Personalized offers and upselling"
    elif segment == 'At Risk' and prob_alive < 0.5:
        return "🚨 Urgent retention call - special discount"
    elif segment == 'Potential Loyalists':
        return "📧 Loyalty program enrollment - early incentives"
    elif segment == 'New Customers' and prob_alive > 0.9:
        return "👋 Welcome sequence - onboarding campaign"
    elif segment == 'Hibernating' and predicted_clv > 500:
        return "💤 Re-engagement campaign - seasonal offer"
    elif segment == 'Lost' and predicted_clv > 500:
        return "🔄 Win-back campaign - exclusive discount"
    elif segment == 'Needs Attention':
        return "🔍 Investigate - segment-specific campaign"
    else:
        return "📊 Monitor - standard engagement"


if __name__ == "__main__":
    print("Testing CLV models module...")
    
    # Create sample customer data
    np.random.seed(42)
    n_customers = 1000
    
    sample_customers = pd.DataFrame({
        'Customer ID': range(1, n_customers + 1),
        'frequency': np.random.randint(0, 20, n_customers),
        'recency': np.random.randint(1, 365, n_customers),
        'T': np.random.randint(30, 730, n_customers),
        'monetary_value': np.random.uniform(10, 500, n_customers),
        'country': np.random.choice(['UK', 'France', 'Germany', 'Spain'], n_customers)
    })
    
    print(f"Created {len(sample_customers)} sample customers")
    print(sample_customers.head())