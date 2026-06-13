"""
Training Pipeline for CLV Prediction System
Orchestrates data preprocessing, model training, and evaluation
"""

import os
import sys
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from .preprocess import (
    load_data, create_cohorts, split_observation_holdout,
    create_customer_features, compute_holdout_revenue
)
from .rfm import (
    compute_rfm, score_rfm, segment_customers, 
    compute_segment_stats, plot_rfm_heatmap, plot_segment_distribution
)
from .clv_models import (
    BGNBDModel, GammaGammaModel, MLCLVPredictor,
    compute_retention_cohorts, compare_models
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CLVTrainer:
    """
    End-to-end CLV model trainer.
    """
    
    def __init__(self, data_path: str, output_dir: str = "models"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        # Model artifacts
        self.raw_df = None
        self.observation_df = None
        self.holdout_df = None
        self.customers_df = None
        self.holdout_revenue = None
        
        self.bgnbd_model = None
        self.gamma_gamma_model = None
        self.ml_model = None
        
        self.rfm_df = None
        
        # Training results
        self.metrics = {}
        self.plots = {}
        
    def load_and_preprocess(self):
        """Load and preprocess the data."""
        print("\n" + "="*60)
        print("STEP 1: Loading and Preprocessing Data")
        print("="*60)
        
        # Load data
        self.raw_df = load_data(self.data_path)
        
        # Create cohorts
        self.raw_df = create_cohorts(self.raw_df)
        
        # Split observation/holdout
        self.observation_df, self.holdout_df = split_observation_holdout(
            self.raw_df,
            observation_days=365,
            holdout_days=90
        )
        
        # Create customer features
        self.customers_df = create_customer_features(self.observation_df)
        
        # Compute holdout revenue (target variable)
        self.holdout_revenue = compute_holdout_revenue(self.holdout_df)
        
        # Add holdout revenue to customers dataframe
        self.customers_df['holdout_revenue'] = self.customers_df['Customer ID'].map(
            self.holdout_revenue
        ).fillna(0)
        
        print(f"\nCustomer features shape: {self.customers_df.shape}")
        print(f"Customers with holdout revenue: {(self.customers_df['holdout_revenue'] > 0).sum()}")
        
        return self
    
    def compute_rfm(self):
        """Compute RFM segmentation."""
        print("\n" + "="*60)
        print("STEP 2: RFM Analysis")
        print("="*60)
        
        # Compute RFM
        self.rfm_df = compute_rfm(self.observation_df)
        
        # Score RFM
        self.rfm_df = score_rfm(self.rfm_df)
        
        # Segment customers
        self.rfm_df = segment_customers(self.rfm_df)
        
        # Merge with customer features
        self.customers_df = self.customers_df.merge(
            self.rfm_df[['Customer ID', 'R_score', 'F_score', 'M_score', 'RFM_score', 'segment']],
            on='Customer ID',
            how='left'
        )
        
        # Compute segment stats
        segment_stats = compute_segment_stats(self.rfm_df, self.observation_df)
        self.segment_stats = segment_stats
        
        # Save segment stats
        segment_stats.to_csv(self.output_dir / "segment_stats.csv", index=False)
        
        print("\nRFM computed successfully!")
        
        return self
    
    def fit_probabilistic_models(self):
        """Fit BG/NBD and Gamma-Gamma models."""
        print("\n" + "="*60)
        print("STEP 3: Fitting Probabilistic CLV Models")
        print("="*60)
        
        # Prepare data for BG/NBD - filter to valid customers
        # Must have frequency > 0 (repeat customers) for meaningful model
        bgnbd_customers = self.customers_df[self.customers_df['frequency'] > 0].copy()
        
        # Further filter for stability
        bgnbd_customers = bgnbd_customers[
            (bgnbd_customers['recency'] > 0) &
            (bgnbd_customers['recency'] <= bgnbd_customers['T'])
        ]
        
        print(f"\nFitting BG/NBD on {len(bgnbd_customers):,} repeat customers...")
        
        # Fit BG/NBD model with higher penalizer for stability
        self.bgnbd_model = BGNBDModel(penalizer_coef=0.1)
        self.bgnbd_model.fit(bgnbd_customers)
        
        # Predict purchases and probability alive for all customers
        # Use repeat customers for prediction
        self.customers_df['bgnbd_predicted_purchases'] = 0.0
        self.customers_df['prob_alive'] = 0.5  # Default for one-time buyers
        
        # Predict for customers with features
        predict_df = self.customers_df[
            (self.customers_df['frequency'] > 0) &
            (self.customers_df['recency'] > 0) &
            (self.customers_df['recency'] <= self.customers_df['T'])
        ].copy()
        
        if len(predict_df) > 0:
            self.customers_df.loc[predict_df.index, 'bgnbd_predicted_purchases'] = \
                self.bgnbd_model.predict_purchases(predict_df, t=90).reindex(predict_df.index).fillna(0).values
            self.customers_df.loc[predict_df.index, 'prob_alive'] = \
                self.bgnbd_model.predict_alive_probability(predict_df).reindex(predict_df.index).fillna(1).values
        
        # For one-time buyers, estimate 1 purchase per year
        self.customers_df.loc[self.customers_df['frequency'] == 0, 'bgnbd_predicted_purchases'] = \
            self.customers_df.loc[self.customers_df['frequency'] == 0, 'monetary_value'] * 0.1 / 365 * 90
        
        # Fit Gamma-Gamma model on repeat customers
        print("\nFitting Gamma-Gamma Model...")
        gg_customers = self.customers_df[
            (self.customers_df['frequency'] > 0) &
            (self.customers_df['monetary_value'] > 0)
        ].copy()
        
        self.gamma_gamma_model = GammaGammaModel(penalizer_coef=0.1)
        self.gamma_gamma_model.fit(gg_customers)
        
        # Predict monetary value - use actual monetary for one-time buyers
        self.customers_df['gamma_gamma_monetary'] = self.customers_df['monetary_value'] * 0.10  # Default margin
        
        # Predict for repeat customers
        gg_predict_df = self.customers_df[
            (self.customers_df['frequency'] > 0) &
            (self.customers_df['monetary_value'] > 0)
        ].copy()
        
        if len(gg_predict_df) > 0:
            predicted_monetary = self.gamma_gamma_model.predict_average_profit(gg_predict_df, margin=0.10)
            self.customers_df.loc[gg_predict_df.index, 'gamma_gamma_monetary'] = \
                predicted_monetary.reindex(gg_predict_df.index).fillna(
                    self.customers_df.loc[gg_predict_df.index, 'monetary_value'] * 0.10
                ).values
        
        # Compute probabilistic CLV: predicted_purchases * predicted_profit * 12 months
        self.customers_df['bgnbd_gg_clv'] = (
            self.customers_df['bgnbd_predicted_purchases'] / 90 * 30 * 12 *  # Monthly equivalent
            self.customers_df['gamma_gamma_monetary']
        )
        
        # Cap at reasonable values
        self.customers_df['bgnbd_gg_clv'] = self.customers_df['bgnbd_gg_clv'].clip(upper=100000)
        
        print("\nProbabilistic models fitted successfully!")
        
        return self
    
    def train_ml_model(self):
        """Train ML-based CLV predictor."""
        print("\n" + "="*60)
        print("STEP 4: Training ML CLV Predictor")
        print("="*60)
        
        # Initialize ML predictor
        self.ml_model = MLCLVPredictor(random_state=42, n_trials=20)
        
        # Prepare features
        print("\nPreparing features...")
        X = self.ml_model.prepare_features(
            self.customers_df,
            bgnbd_predictions=self.customers_df['bgnbd_predicted_purchases'],
            prob_alive=self.customers_df['prob_alive'],
            gamma_gamma_monetary=self.customers_df['gamma_gamma_monetary']
        )
        
        # Filter to customers with holdout revenue for training
        mask = self.customers_df['holdout_revenue'] > 0
        X_train = X[mask]
        y_train = self.customers_df.loc[mask, 'holdout_revenue'].values
        
        print(f"Training data: {len(X_train)} customers with holdout revenue")
        
        # Train
        self.ml_model.train(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating on holdout set...")
        metrics = self.ml_model.evaluate(X_train, y_train)
        self.metrics['ml'] = metrics
        
        for model_name, model_metrics in metrics.items():
            if isinstance(model_metrics, dict):
                print(f"\n{model_name}:")
                for metric, value in model_metrics.items():
                    print(f"  {metric}: {value:.4f}")
        
        return self
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "="*60)
        print("STEP 5: Generating Plots")
        print("="*60)
        
        # 1. Retention cohort heatmap
        print("Creating retention cohort heatmap...")
        try:
            retention_matrix, fig = compute_retention_cohorts(self.raw_df)
            fig.write_html(str(self.output_dir / "plots" / "retention_cohorts.html"))
            self.plots['retention_cohorts'] = fig
        except Exception as e:
            print(f"Could not create retention cohorts: {e}")
        
        # 2. RFM segment distribution
        print("Creating RFM segment distribution...")
        fig = plot_segment_distribution(self.rfm_df)
        fig.write_html(str(self.output_dir / "plots" / "rfm_segments.html"))
        self.plots['rfm_segments'] = fig
        
        # 3. RFM heatmap
        print("Creating RFM heatmap...")
        fig = plot_rfm_heatmap(self.rfm_df)
        fig.write_html(str(self.output_dir / "plots" / "rfm_heatmap.html"))
        self.plots['rfm_heatmap'] = fig
        
        # 4. CLV distribution by segment
        print("Creating CLV by segment distribution...")
        fig = go.Figure()
        
        segments_order = ['Champions', 'Loyal', 'Potential Loyalists', 'New Customers',
                         'At Risk', 'Hibernating', 'Lost', 'Needs Attention']
        
        for segment in segments_order:
            if segment in self.customers_df['segment'].values:
                segment_data = self.customers_df[
                    self.customers_df['segment'] == segment
                ]['bgnbd_gg_clv'].values
                
                fig.add_trace(go.Violin(
                    y=segment_data,
                    name=segment,
                    box_visible=True,
                    meanline_visible=True
                ))
        
        fig.update_layout(
            title='CLV Distribution by RFM Segment',
            yaxis_title='Predicted CLV (£)',
            showlegend=True
        )
        fig.write_html(str(self.output_dir / "plots" / "clv_by_segment.html"))
        self.plots['clv_by_segment'] = fig
        
        # 5. BG/NBD frequency-recency plot
        print("Creating BG/NBD diagnostic plot...")
        try:
            fig = self.bgnbd_model.plot_frequency_recency()
            plt.savefig(str(self.output_dir / "plots" / "bgnbd_diagnostic.png"), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create BG/NBD plot: {e}")
        
        # 6. Actual vs Predicted scatter
        print("Creating actual vs predicted scatter...")
        mask = self.customers_df['holdout_revenue'] > 0
        
        if 'bgnbd_gg_clv' in self.customers_df.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.customers_df.loc[mask, 'holdout_revenue'],
                y=self.customers_df.loc[mask, 'bgnbd_gg_clv'],
                mode='markers',
                name='BG/NBD + GG',
                marker=dict(opacity=0.5)
            ))
            
            # Add perfect prediction line
            max_val = max(self.customers_df.loc[mask, 'holdout_revenue'].max(),
                         self.customers_df.loc[mask, 'bgnbd_gg_clv'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Actual vs Predicted CLV (BG/NBD + Gamma-Gamma)',
                xaxis_title='Actual 90-Day Revenue (£)',
                yaxis_title='Predicted CLV (£)'
            )
            fig.write_html(str(self.output_dir / "plots" / "actual_vs_predicted.html"))
            self.plots['actual_vs_predicted'] = fig
        
        # 7. Feature importance
        print("Creating feature importance chart...")
        importance_df = self.ml_model.get_feature_importance()
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title='Feature Importance (XGBoost)',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600
        )
        fig.write_html(str(self.output_dir / "plots" / "feature_importance.html"))
        self.plots['feature_importance'] = fig
        
        # Save feature importance CSV
        importance_df.to_csv(self.output_dir / "feature_importance.csv", index=False)
        
        print("\nAll plots generated!")
        
        return self
    
    def save_models(self):
        """Save all model artifacts."""
        print("\n" + "="*60)
        print("STEP 6: Saving Models")
        print("="*60)
        
        # Save BG/NBD model
        self.bgnbd_model.save(str(self.output_dir / "bgnbd_model.pkl"))
        print("Saved BG/NBD model")
        
        # Save Gamma-Gamma model
        self.gamma_gamma_model.save(str(self.output_dir / "gamma_gamma_model.pkl"))
        print("Saved Gamma-Gamma model")
        
        # Save ML model
        self.ml_model.save(str(self.output_dir / "ml_model"))
        print("Saved ML model")
        
        # Save customer features with predictions
        self.customers_df.to_csv(self.output_dir / "customers_with_predictions.csv", index=False)
        print("Saved customer predictions")
        
        # Save RFM segmentation
        self.rfm_df.to_csv(self.output_dir / "rfm_segmentation.csv", index=False)
        print("Saved RFM segmentation")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'n_customers': len(self.customers_df),
            'n_features': len(self.ml_model.feature_names),
            'features': self.ml_model.feature_names,
            'metrics': self.metrics
        }
        
        with open(self.output_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        print("Saved metadata")
        
        print("\nAll models saved successfully!")
        
        return self
    
    def run_full_pipeline(self):
        """Execute the full training pipeline."""
        print("\n" + "="*60)
        print("🚀 CLV MODEL TRAINING PIPELINE")
        print("="*60)
        
        self.load_and_preprocess()
        self.compute_rfm()
        self.fit_probabilistic_models()
        self.train_ml_model()
        self.generate_all_plots()
        self.save_models()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModels saved to: {self.output_dir}")
        print(f"Plots saved to: {self.output_dir / 'plots'}")
        
        return self


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train CLV Prediction Models')
    parser.add_argument('--data', type=str, default='data/online_retail.csv',
                       help='Path to the dataset')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"\n❌ Data file not found: {args.data}")
        print("Please download the Online Retail II dataset from Kaggle")
        print("Link: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
        sys.exit(1)
    
    # Run training
    trainer = CLVTrainer(args.data, args.output)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()