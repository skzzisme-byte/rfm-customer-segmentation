"""
Main Pipeline Orchestrator
Runs the complete RFM analysis with churn prediction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_cleaning import clean_retail_data, save_cleaned_data
from rfm_calculation import calculate_rfm, score_rfm
from segmentation import segment_customers
from visualization import create_all_visualizations
from churn_prediction import (
    create_churn_labels, prepare_features, train_churn_model,
    predict_churn_risk, save_model
)

import pandas as pd


def main():
    print("=" * 70)
    print("RFM CUSTOMER SEGMENTATION & CHURN PREDICTION PIPELINE")
    print("=" * 70)

    # Step 1: Data Cleaning
    print("\n" + "=" * 70)
    print("STEP 1: DATA CLEANING")
    print("=" * 70)
    clean_df, returns_df = clean_retail_data('data/raw/Online Retail.xlsx')
    save_cleaned_data(clean_df, 'data/processed/clean_data.csv')

    # Step 2: RFM Calculation
    print("\n" + "=" * 70)
    print("STEP 2: RFM CALCULATION")
    print("=" * 70)
    rfm, snapshot_date = calculate_rfm(clean_df)
    rfm = score_rfm(rfm)

    # Step 3: Segmentation
    print("\n" + "=" * 70)
    print("STEP 3: CUSTOMER SEGMENTATION")
    print("=" * 70)
    rfm, segment_stats = segment_customers(rfm)

    # Step 4: Churn Prediction
    print("\n" + "=" * 70)
    print("STEP 4: CHURN PREDICTION")
    print("=" * 70)
    rfm = create_churn_labels(rfm, churn_threshold_days=90)
    X, y = prepare_features(rfm)

    # Train Random Forest
    rf_results = train_churn_model(X, y, 'random_forest')
    save_model(rf_results, 'outputs/')

    # Predict churn probability for all customers
    rfm = predict_churn_risk(rfm, rf_results['model'])

    # Step 5: Visualization
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZATION")
    print("=" * 70)
    create_all_visualizations(rfm, segment_stats, rf_results, 'outputs/figures/')

    # Step 6: Save Outputs
    print("\n" + "=" * 70)
    print("STEP 6: SAVING OUTPUTS")
    print("=" * 70)
    rfm.to_csv('outputs/rfm_segments.csv', index=False)
    segment_stats.to_csv('outputs/segment_summary.csv')

    # Generate report
    generate_report(rfm, segment_stats, snapshot_date, rf_results)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review outputs in outputs/ folder")
    print("2. Launch dashboard: streamlit run dashboard/app.py")
    print("3. Check README.md for project documentation")


def generate_report(rfm, segment_stats, snapshot_date, model_results):
    """Generate comprehensive analysis report."""

    top_segments = segment_stats.nlargest(3, 'Total_Revenue')
    top_3_revenue_share = top_segments['Revenue_Share'].sum()

    champions_pct = segment_stats.loc['Champions', 'Percentage'] if 'Champions' in segment_stats.index else 0
    champions_revenue = segment_stats.loc['Champions', 'Revenue_Share'] if 'Champions' in segment_stats.index else 0

    at_risk_count = segment_stats.loc['At Risk', 'Count'] if 'At Risk' in segment_stats.index else 0

    high_risk_count = len(rfm[rfm['Churn_Risk'] == 'High'])
    high_risk_revenue = rfm[rfm['Churn_Risk'] == 'High']['Monetary'].sum()

    report = f"""
{'=' * 70}
RFM CUSTOMER SEGMENTATION & CHURN PREDICTION ANALYSIS REPORT
{'=' * 70}

Analysis Date: {snapshot_date.strftime('%Y-%m-%d')}
Dataset: UCI Online Retail Dataset (Dec 2010 - Dec 2011)

EXECUTIVE SUMMARY
-----------------
Total Customers Analyzed: {len(rfm):,}
Total Revenue: ${rfm['Monetary'].sum():,.2f}
Churn Prediction Model AUC: {model_results['auc']:.3f}

KEY FINDINGS
------------
1. REVENUE CONCENTRATION
   Top 3 segments account for {top_3_revenue_share:.1f}% of total revenue

2. CHAMPIONS SEGMENT
   {champions_pct:.1f}% of customers generate {champions_revenue:.1f}% of revenue

3. AT-RISK CUSTOMERS
   {at_risk_count:,} customers in 'At Risk' segment need immediate attention

4. CHURN PREDICTION
   {high_risk_count:,} customers ({high_risk_count / len(rfm):.1%}) have high churn risk
   Revenue at risk: ${high_risk_revenue:,.2f}

SEGMENT BREAKDOWN
-----------------
{segment_stats[['Count', 'Percentage', 'Revenue_Share', 'Avg_Monetary']].to_string()}

CHURN RISK DISTRIBUTION
-----------------------
{rfm['Churn_Risk'].value_counts().to_string()}

RECOMMENDED ACTIONS
-------------------
1. IMMEDIATE (High Churn Risk):
   - Launch win-back campaign for {high_risk_count:,} high-risk customers
   - Focus on 'Cannot Lose Them' segment with personalized offers

2. SHORT-TERM (Segment Optimization):
   - VIP program for Champions
   - Loyalty rewards for Loyal Customers
   - Onboarding for New Customers

3. LONG-TERM (Retention):
   - Monitor 'About to Sleep' segment
   - Increase purchase frequency for 'Potential Loyalists'

MODEL PERFORMANCE
-----------------
Algorithm: Random Forest Classifier
AUC-ROC: {model_results['auc']:.3f}

Top Predictive Features:
{model_results['feature_importance'].head(5).to_string(index=False)}

{'=' * 70}
Report generated by RFM Analysis Pipeline
{'=' * 70}
"""

    with open('outputs/summary_report.txt', 'w') as f:
        f.write(report)

    print("Report saved to outputs/summary_report.txt")


if __name__ == "__main__":
    main()