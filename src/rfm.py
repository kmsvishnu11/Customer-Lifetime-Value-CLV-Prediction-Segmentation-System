"""
RFM Analysis Module for Customer Segmentation
Implements Recency, Frequency, Monetary analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute RFM scores for each customer.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    snapshot_date : pd.Timestamp, optional
        Reference date for Recency calculation.
        Default: max date + 1 day
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Customer ID, R, F, M scores
    """
    df = df.copy()
    
    # Default snapshot date is day after last transaction
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    print(f"Computing RFM with snapshot date: {snapshot_date}")
    
    # Aggregate transactions by customer
    customer_trans = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency
        'Revenue': 'mean'  # Monetary (average transaction value)
    })
    
    customer_trans.columns = ['Recency', 'Frequency', 'Monetary']
    customer_trans = customer_trans.reset_index()
    
    # Handle any infinite or NaN values
    customer_trans['Recency'] = customer_trans['Recency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    customer_trans['Monetary'] = customer_trans['Monetary'].fillna(0)
    
    print(f"Computed RFM for {len(customer_trans):,} customers")
    
    return customer_trans


def score_rfm(rfm_df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Score RFM components on a scale of 1-5.
    
    Scoring logic:
    - Recency: Lower is better (more recent = higher score)
    - Frequency: Higher is better
    - Monetary: Higher is better
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with Recency, Frequency, Monetary columns
    n_quantiles : int
        Number of quantiles for scoring (default: 5)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with R, F, M scores (1-5) and combined score
    """
    rfm_df = rfm_df.copy()
    
    # Score Recency (lower is better, so reverse the quantile)
    # Use qcut with labels reversed
    rfm_df['R_score'] = pd.qcut(
        rfm_df['Recency'], 
        q=n_quantiles, 
        labels=[n_quantiles, n_quantiles-1, 3, 2, 1],
        duplicates='drop'
    ).astype(int)
    
    # Score Frequency (higher is better)
    rfm_df['F_score'] = pd.qcut(
        rfm_df['Frequency'].rank(method='first'),
        q=n_quantiles,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    # Score Monetary (higher is better)
    rfm_df['M_score'] = pd.qcut(
        rfm_df['Monetary'].rank(method='first'),
        q=n_quantiles,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    # Combined RFM score (110-555 scale)
    rfm_df['RFM_score'] = rfm_df['R_score'] * 100 + rfm_df['F_score'] * 10 + rfm_df['M_score']
    
    # Handle cases where qcut might produce fewer bins due to duplicates
    # Re-score with rank-based method for robustness
    rfm_df['R_score'] = pd.cut(
        rfm_df['Recency'].rank(method='average'),
        bins=n_quantiles,
        labels=[5, 4, 3, 2, 1],
        duplicates='drop'
    ).astype(float).fillna(3).astype(int)
    
    rfm_df['F_score'] = pd.cut(
        rfm_df['Frequency'].rank(method='average'),
        bins=n_quantiles,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(float).fillna(3).astype(int)
    
    rfm_df['M_score'] = pd.cut(
        rfm_df['Monetary'].rank(method='average'),
        bins=n_quantiles,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(float).fillna(3).astype(int)
    
    # Recalculate combined score
    rfm_df['RFM_score'] = rfm_df['R_score'] * 100 + rfm_df['F_score'] * 10 + rfm_df['M_score']
    
    print(f"RFM scoring complete. Score range: {rfm_df['RFM_score'].min()}-{rfm_df['RFM_score'].max()}")
    
    return rfm_df


def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers based on RFM scores.
    
    Segment definitions:
    - Champions: R>=4, F>=4, M>=4 (Best customers)
    - Loyal: F>=4 (Frequently buying, regardless of recency)
    - Potential Loyalists: R>=3, F>=2 (Recent, moderate frequency)
    - New Customers: R>=4, F<=1 (Recent, first purchase)
    - At Risk: R<=2, F>=3 (Used to buy frequently but haven't returned)
    - Hibernating: R<=2, F<=2 (Low everything)
    - Lost: R=1, F=1 (Worst customers)
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with R_score, F_score, M_score columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with segment column added
    """
    rfm_df = rfm_df.copy()
    
    def assign_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif f >= 4:
            return 'Loyal'
        elif r >= 3 and f >= 2:
            return 'Potential Loyalists'
        elif r >= 4 and f <= 1:
            return 'New Customers'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        elif r == 1 and f == 1:
            return 'Lost'
        else:
            return 'Needs Attention'
    
    rfm_df['segment'] = rfm_df.apply(assign_segment, axis=1)
    
    # Print segment distribution
    segment_counts = rfm_df['segment'].value_counts()
    print("\nSegment Distribution:")
    print(segment_counts)
    
    return rfm_df


def compute_segment_stats(rfm_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each RFM segment.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with segment column
    original_df : pd.DataFrame
        Original transaction dataframe for revenue calculation
        
    Returns:
    --------
    pd.DataFrame
        Segment summary statistics
    """
    # Compute customer-level revenue
    customer_revenue = original_df.groupby('Customer ID')['Revenue'].sum().reset_index()
    customer_revenue.columns = ['Customer ID', 'total_revenue']
    
    # Merge with RFM
    rfm_with_revenue = rfm_df.merge(customer_revenue, on='Customer ID', how='left')
    
    # Total revenue for percentage calculation
    total_revenue = rfm_with_revenue['total_revenue'].sum()
    
    # Group by segment
    segment_stats = rfm_with_revenue.groupby('segment').agg({
        'Customer ID': 'count',
        'total_revenue': ['sum', 'mean'],
        'R_score': 'mean',
        'F_score': 'mean',
        'M_score': 'mean',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # Flatten column names
    segment_stats.columns = [
        'customer_count', 'segment_revenue', 'avg_revenue_per_customer',
        'avg_R_score', 'avg_F_score', 'avg_M_score',
        'avg_recency', 'avg_frequency', 'avg_monetary'
    ]
    segment_stats = segment_stats.reset_index()
    
    # Calculate revenue share
    segment_stats['revenue_share_pct'] = (
        segment_stats['segment_revenue'] / total_revenue * 100
    ).round(2)
    
    # Sort by revenue share
    segment_stats = segment_stats.sort_values('revenue_share_pct', ascending=False)
    
    print("\nSegment Statistics:")
    print(segment_stats.to_string())
    
    return segment_stats


def plot_rfm_heatmap(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap of Recency vs Frequency with Monetary as values.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with R_score, F_score, Monetary columns
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create pivot table for heatmap
    heatmap_data = rfm_df.groupby(['R_score', 'F_score'])['Monetary'].mean().unstack()
    
    # Ensure all combinations exist
    all_scores = [1, 2, 3, 4, 5]
    heatmap_data = heatmap_data.reindex(index=all_scores[::-1], columns=all_scores, fill_value=0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f'F={i}' for i in all_scores],
        y=[f'R={i}' for i in all_scores[::-1]],
        colorscale='RdYlGn',
        text=np.round(heatmap_data.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar_title="Avg Monetary (£)",
        hovertemplate='Recency: %{y}<br>Frequency: %{x}<br>Avg Monetary: £%{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='RFM Heatmap: Recency vs Frequency<br><sub>Cell values show average monetary value</sub>',
        xaxis_title='Frequency Score',
        yaxis_title='Recency Score',
        width=700,
        height=500
    )
    
    return fig


def plot_segment_distribution(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart of segment distribution.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with segment column
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    segment_counts = rfm_df['segment'].value_counts()
    
    # Color mapping for segments
    color_map = {
        'Champions': '#00C851',
        'Loyal': '#007E33',
        'Potential Loyalists': '#33B5E5',
        'New Customers': '#AA66CC',
        'At Risk': '#FF4444',
        'Hibernating': '#CC0000',
        'Lost': '#999999',
        'Needs Attention': '#FFBB33'
    }
    
    colors = [color_map.get(seg, '#888888') for seg in segment_counts.index]
    
    fig = go.Figure(data=go.Bar(
        x=segment_counts.index,
        y=segment_counts.values,
        marker_color=colors,
        text=segment_counts.values,
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Customer Distribution by RFM Segment',
        xaxis_title='Segment',
        yaxis_title='Number of Customers',
        showlegend=False,
        width=900,
        height=500
    )
    
    return fig


def get_segment_action_recommendations() -> dict:
    """
    Return action recommendations for each segment.
    
    Returns:
    --------
    dict
        Dictionary mapping segment to recommendation
    """
    return {
        'Champions': {
            'priority': 'High',
            'action': 'Loyalty rewards program, exclusive offers, VIP treatment',
            'budget_allocation': '30%',
            'kpi': 'Retain and increase purchase frequency'
        },
        'Loyal': {
            'priority': 'High',
            'action': 'Personalized recommendations, upselling opportunities',
            'budget_allocation': '25%',
            'kpi': 'Maintain engagement, prevent churn'
        },
        'Potential Loyalists': {
            'priority': 'Medium',
            'action': 'Targeted promotions, loyalty program enrollment',
            'budget_allocation': '20%',
            'kpi': 'Convert to Loyal segment'
        },
        'New Customers': {
            'priority': 'Medium',
            'action': 'Welcome campaigns, onboarding sequence',
            'budget_allocation': '15%',
            'kpi': 'Increase second purchase rate'
        },
        'At Risk': {
            'priority': 'High',
            'action': 'Win-back campaigns, special discounts, personalized outreach',
            'budget_allocation': '25%',
            'kpi': 'Reactivate before churn'
        },
        'Hibernating': {
            'priority': 'Medium',
            'action': 'Re-engagement offers, seasonal promotions',
            'budget_allocation': '15%',
            'kpi': 'Minimum cost to maintain'
        },
        'Lost': {
            'priority': 'Low',
            'action': 'Low-cost reactivation attempts, exit surveys',
            'budget_allocation': '5%',
            'kpi': 'Salvage if possible'
        },
        'Needs Attention': {
            'priority': 'Medium',
            'action': 'Segment-specific campaigns based on R/F/M pattern',
            'budget_allocation': '10%',
            'kpi': 'Move to higher segment'
        }
    }


if __name__ == "__main__":
    print("Testing RFM module...")
    
    # Create sample RFM data
    sample_rfm = pd.DataFrame({
        'Customer ID': [1, 2, 3, 4, 5],
        'Recency': [10, 30, 100, 200, 365],
        'Frequency': [10, 5, 3, 2, 1],
        'Monetary': [500, 200, 100, 50, 20]
    })
    
    # Score and segment
    scored = score_rfm(sample_rfm)
    segmented = segment_customers(scored)
    
    print("\nSegmented RFM:")
    print(segmented)
    
    print("\nRecommendations:")
    print(get_segment_action_recommendations())