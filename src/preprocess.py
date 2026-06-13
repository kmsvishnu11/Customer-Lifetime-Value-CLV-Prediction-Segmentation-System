"""
Data Preprocessing Module for CLV Prediction
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict


def load_data(path: str) -> pd.DataFrame:
    """
    Load Online Retail II dataset and perform initial cleaning.
    
    Parameters:
    -----------
    path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    # Load CSV with latin-1 encoding for special characters
    df = pd.read_csv(path, encoding='latin-1')
    
    # Remove cancelled orders (Invoice starts with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    # Remove rows with Quantity <= 0
    df = df[df['Quantity'] > 0]
    
    # Remove rows with Price <= 0
    df = df[df['Price'] > 0]
    
    # Parse InvoiceDate as datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Add Revenue column
    df['Revenue'] = df['Quantity'] * df['Price']
    
    # Remove extreme outliers (price > 99th percentile)
    price_99th = df['Price'].quantile(0.99)
    df = df[df['Price'] <= price_99th]
    
    # Remove rows with negative or null CustomerID (we need customer identity)
    df = df[df['Customer ID'].notna()]
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # Remove returns and adjustments (StockCode starting with 'A' or 'M')
    df = df[~df['StockCode'].astype(str).str.match(r'^[AM]', na=False)]
    
    print(f"Loaded {len(df):,} transactions from {df['Customer ID'].nunique():,} customers")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    return df


def create_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cohort columns based on customer's first purchase month.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with CohortMonth and CohortIndex columns
    """
    df = df.copy()
    
    # Find each customer's first purchase month
    first_purchase = df.groupby('Customer ID')['InvoiceDate'].min()
    
    # Create CohortMonth (first purchase month)
    df['CohortMonth'] = df['Customer ID'].map(first_purchase).dt.to_period('M')
    
    # Create InvoiceMonth (transaction month)
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    
    # Calculate CohortIndex (months since first purchase)
    df['CohortIndex'] = (
        (df['InvoiceMonth'].astype('int64') - df['CohortMonth'].astype('int64'))
    )
    
    return df


def split_observation_holdout(
    df: pd.DataFrame,
    observation_days: int = 365,
    holdout_days: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into observation and holdout periods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    observation_days : int
        Number of days for observation period (default: 365)
    holdout_days : int
        Number of days for holdout period (default: 90)
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        observation_df, holdout_df
    """
    df = df.copy()
    
    # Define observation and holdout periods
    max_date = df['InvoiceDate'].max()
    observation_end = max_date - timedelta(days=holdout_days)
    holdout_start = observation_end + timedelta(days=1)
    
    print(f"Date range: {df['InvoiceDate'].min()} to {max_date}")
    print(f"Observation period: {df['InvoiceDate'].min()} to {observation_end} ({observation_days} days)")
    print(f"Holdout period: {holdout_start} to {max_date} ({holdout_days} days)")
    
    # Split based on InvoiceDate
    observation_df = df[df['InvoiceDate'] <= observation_end].copy()
    holdout_df = df[df['InvoiceDate'] >= holdout_start].copy()
    
    print(f"Observation: {len(observation_df):,} transactions")
    print(f"Holdout: {len(holdout_df):,} transactions")
    
    return observation_df, holdout_df


def create_customer_features(observation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level features from observation period.
    
    Features computed:
    - frequency: number of repeat purchases (not first)
    - recency: days from first to last purchase
    - T: days from first purchase to observation end
    - monetary_value: average purchase value
    - total_revenue_obs: total revenue in observation period
    - first/last_purchase_date
    - avg_days_between_purchases
    - purchase_std: std of purchase amounts
    - country: most frequent country
    - n_unique_products: distinct products bought
    - n_unique_months: active months
    - max_purchase_gap: max days between purchases
    
    Parameters:
    -----------
    observation_df : pd.DataFrame
        Transaction dataframe from observation period
        
    Returns:
    --------
    pd.DataFrame
        Customer-level dataframe with all features
    """
    observation_df = observation_df.copy()
    
    # Reference date for T calculation
    max_date = observation_df['InvoiceDate'].max()
    
    # Aggregate by customer
    customer_stats = observation_df.groupby('Customer ID').agg({
        'InvoiceDate': ['min', 'max', 'count'],
        'Revenue': ['sum', 'mean', 'std'],
        'Quantity': 'sum',
        'StockCode': 'nunique',
        'Country': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    })
    
    # Flatten column names
    customer_stats.columns = [
        'first_purchase_date', 'last_purchase_date', 'n_transactions',
        'total_revenue_obs', 'avg_revenue_per_txn', 'std_revenue_per_txn',
        'total_quantity', 'n_unique_products', 'country'
    ]
    customer_stats = customer_stats.reset_index()
    
    # Fill NaN in std with 0 (for customers with single transaction)
    customer_stats['std_revenue_per_txn'] = customer_stats['std_revenue_per_txn'].fillna(0)
    
    # Calculate frequency (repeat purchases = total - 1)
    customer_stats['frequency'] = customer_stats['n_transactions'] - 1
    
    # Calculate recency (days from first to last purchase)
    customer_stats['recency'] = (
        customer_stats['last_purchase_date'] - customer_stats['first_purchase_date']
    ).dt.days
    
    # Calculate T (days from first purchase to observation end)
    customer_stats['T'] = (max_date - customer_stats['first_purchase_date']).dt.days
    
    # Calculate monetary_value (average purchase value)
    customer_stats['monetary_value'] = customer_stats['avg_revenue_per_txn']
    
    # Calculate average days between purchases
    def calc_avg_days_between_purchases(group):
        if len(group) < 2:
            return np.nan
        sorted_dates = group.sort_values()
        gaps = sorted_dates.diff().dt.days.dropna()
        return gaps.mean() if len(gaps) > 0 else np.nan
    
    avg_days = observation_df.groupby('Customer ID')['InvoiceDate'].apply(
        calc_avg_days_between_purchases
    )
    customer_stats['avg_days_between_purchases'] = customer_stats['Customer ID'].map(avg_days)
    
    # Calculate max purchase gap
    def calc_max_gap(group):
        if len(group) < 2:
            return np.nan
        sorted_dates = group.sort_values()
        gaps = sorted_dates.diff().dt.days.dropna()
        return gaps.max() if len(gaps) > 0 else np.nan
    
    max_gap = observation_df.groupby('Customer ID')['InvoiceDate'].apply(calc_max_gap)
    customer_stats['max_purchase_gap'] = customer_stats['Customer ID'].map(max_gap)
    
    # Calculate number of unique months active
    observation_df_temp = observation_df.copy()
    observation_df_temp['YearMonth'] = observation_df_temp['InvoiceDate'].dt.to_period('M')
    n_months = observation_df_temp.groupby('Customer ID')['YearMonth'].nunique()
    customer_stats['n_unique_months'] = customer_stats['Customer ID'].map(n_months)
    
    # Handle customers with no repeat purchases for avg_days calculation
    customer_stats['avg_days_between_purchases'] = customer_stats['avg_days_between_purchases'].fillna(0)
    customer_stats['max_purchase_gap'] = customer_stats['max_purchase_gap'].fillna(0)
    
    # Create additional features
    customer_stats['first_purchase_date'] = customer_stats['first_purchase_date'].astype(str)
    customer_stats['last_purchase_date'] = customer_stats['last_purchase_date'].astype(str)
    
    # Convert T to numeric (handle edge cases)
    customer_stats['T'] = customer_stats['T'].clip(lower=1)
    customer_stats['recency'] = customer_stats['recency'].clip(lower=0)
    
    print(f"Created features for {len(customer_stats):,} customers")
    
    return customer_stats


def compute_holdout_revenue(holdout_df: pd.DataFrame) -> Dict[int, float]:
    """
    Compute actual revenue for each customer in holdout period.
    
    Parameters:
    -----------
    holdout_df : pd.DataFrame
        Transaction dataframe from holdout period
        
    Returns:
    --------
    Dict[int, float]
        Dictionary mapping customer_id to holdout_revenue
    """
    holdout_revenue = holdout_df.groupby('Customer ID')['Revenue'].sum().to_dict()
    
    print(f"Computed holdout revenue for {len(holdout_revenue):,} customers")
    
    return holdout_revenue


def get_feature_columns() -> list:
    """
    Return list of feature columns used for ML model.
    """
    return [
        'frequency', 'recency', 'T', 'monetary_value', 'total_revenue_obs',
        'avg_revenue_per_txn', 'std_revenue_per_txn', 'n_unique_products',
        'n_unique_months', 'avg_days_between_purchases', 'max_purchase_gap',
        'n_transactions', 'total_quantity'
    ]


if __name__ == "__main__":
    # Test the preprocessing
    print("Testing preprocessing module...")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'Invoice': ['12345', 'C12345', '67890', '11111'],
        'InvoiceDate': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-05']),
        'Customer ID': [1.0, 2.0, 1.0, 3.0],
        'Quantity': [10, -5, 20, 5],
        'Price': [5.0, 10.0, 3.0, 8.0],
        'StockCode': ['ABC', 'ABC', 'DEF', 'GHI'],
        'Country': ['UK', 'UK', 'France', 'UK']
    })
    
    print("Sample data loaded successfully")
    print(f"Feature columns: {get_feature_columns()}")