"""
RFM Calculation Module
Computes Recency, Frequency, and Monetary metrics for each customer
"""

import pandas as pd
from datetime import datetime


def calculate_rfm(df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    Calculate RFM metrics for each customer.

    Args:
        df: Cleaned transaction dataframe
        reference_date: Snapshot date (default: day after last transaction)

    Returns:
        DataFrame with CustomerID, Recency, Frequency, Monetary
    """
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    print(f"\nReference date for analysis: {reference_date}")

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Add additional metrics for deeper analysis
    customer_stats = df.groupby('CustomerID').agg({
        'TotalPrice': ['mean', 'std', 'min', 'max'],
        'Quantity': 'sum',
        'InvoiceDate': ['min', 'max']
    }).reset_index()

    customer_stats.columns = [
        'CustomerID', 'AvgOrderValue', 'OrderValueStd', 'MinOrderValue',
        'MaxOrderValue', 'TotalQuantity', 'FirstPurchase', 'LastPurchase'
    ]

    rfm = rfm.merge(customer_stats, on='CustomerID')

    # Calculate customer lifetime (days between first and last purchase)
    rfm['CustomerLifetime'] = (rfm['LastPurchase'] - rfm['FirstPurchase']).dt.days

    print(f"\nRFM Metrics Summary:")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))

    return rfm, reference_date


def score_rfm(rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Assign RFM scores using quantile-based scoring.

    Scoring:
    - Recency: Lower = better (recent purchases), so reversed labels
    - Frequency: Higher = better
    - Monetary: Higher = better
    """
    print(f"\nAssigning {n_quantiles}-tier RFM scores...")

    # Recency: Lower days = higher score
    rfm['R_Score'] = pd.qcut(
        rfm['Recency'],
        n_quantiles,
        labels=list(range(n_quantiles, 0, -1))
    ).astype(int)

    # Frequency: Higher count = higher score
    rfm['F_Score'] = pd.qcut(
        rfm['Frequency'].rank(method='first'),
        n_quantiles,
        labels=list(range(1, n_quantiles + 1))
    ).astype(int)

    # Monetary: Higher spend = higher score
    rfm['M_Score'] = pd.qcut(
        rfm['Monetary'],
        n_quantiles,
        labels=list(range(1, n_quantiles + 1))
    ).astype(int)

    # Composite RFM Score
    rfm['RFM_Score'] = (
            rfm['R_Score'].astype(str) +
            rfm['F_Score'].astype(str) +
            rfm['M_Score'].astype(str)
    )

    # RFM combined score (simple average for ranking)
    rfm['RFM_Value'] = (rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']) / 3

    print(f"Score distribution:")
    print(f"R_Score: {rfm['R_Score'].value_counts().sort_index().to_dict()}")
    print(f"F_Score: {rfm['F_Score'].value_counts().sort_index().to_dict()}")
    print(f"M_Score: {rfm['M_Score'].value_counts().sort_index().to_dict()}")

    return rfm


if __name__ == "__main__":
    # Test
    df = pd.read_csv('data/processed/clean_data.csv', parse_dates=['InvoiceDate'])
    rfm, ref_date = calculate_rfm(df)
    rfm = score_rfm(rfm)
    print(rfm.head())