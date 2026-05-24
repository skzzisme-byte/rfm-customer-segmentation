"""
Data Cleaning Module for RFM Analysis
Handles real-world data quality issues in the Online Retail dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_retail_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the Online Retail dataset for RFM analysis.

    Args:
        filepath: Path to the raw Excel file

    Returns:
        Tuple of (cleaned_dataframe, returns_dataframe)
    """
    print("Loading raw data...")
    df = pd.read_excel(filepath)
    print(f"Original dataset: {len(df):,} rows")

    # 1. Handle missing CustomerIDs (guest checkouts - can't segment these)
    missing_before = df['CustomerID'].isna().sum()
    df = df.dropna(subset=['CustomerID']).copy()
    df['CustomerID'] = df['CustomerID'].astype(int)
    print(f"Removed {missing_before:,} rows with missing CustomerID")

    # 2. Remove exact duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {dupes:,} duplicate rows")

    # 3. Separate returns (negative quantities) for separate analysis
    returns = df[df['Quantity'] < 0].copy()
    df = df[df['Quantity'] > 0].copy()
    print(f"Captured {len(returns):,} return transactions")

    # 4. Remove invalid prices
    invalid_prices = (df['UnitPrice'] <= 0).sum()
    df = df[df['UnitPrice'] > 0].copy()
    print(f"Removed {invalid_prices:,} rows with invalid prices")

    # 5. Outlier handling using IQR method (more robust than percentile)
    def remove_outliers_iqr(dataframe, column, multiplier=3.0):
        """Remove extreme outliers using IQR method."""
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return dataframe[(dataframe[column] >= lower_bound) &
                         (dataframe[column] <= upper_bound)]

    rows_before = len(df)
    df = remove_outliers_iqr(df, 'Quantity')
    df = remove_outliers_iqr(df, 'UnitPrice')
    print(f"Removed {rows_before - len(df):,} outlier rows")

    # 6. Create derived features
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # 7. Filter to meaningful transactions
    df = df[df['TotalPrice'] > 0].copy()

    # 8. Add time-based features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour

    print(f"\nFinal cleaned dataset: {len(df):,} rows")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Total revenue: ${df['TotalPrice'].sum():,.2f}")

    return df, returns


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned data to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    # Test the cleaning
    clean_df, returns_df = clean_retail_data('data/raw/Online Retail.xlsx')
    save_cleaned_data(clean_df, 'data/processed/clean_data.csv')