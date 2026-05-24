"""
Customer Segmentation Module
Assigns business-meaningful segments based on RFM scores
"""

import pandas as pd


def segment_customers(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign customer segments based on RFM scores using established marketing logic.

    Segments based on:
    - Champions: Best customers (high R, F, M)
    - Loyal Customers: High frequency and monetary
    - Potential Loyalists: Recent with moderate activity
    - New Customers: Very recent, low frequency
    - Promising: Recent but low spend
    - Need Attention: Moderate across all
    - About to Sleep: Declining recency
    - At Risk: Used to be good but haven't purchased recently
    - Cannot Lose Them: Were top customers, now inactive
    - Hibernating: Low activity, some history
    - Lost: Minimal activity, long inactive
    """

    def get_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

        # Champions: Best of the best (555, 554, 544, 545, 454, 455, 445)
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'

        # Loyal Customers: High frequency and monetary, any recency
        elif f >= 4 and m >= 3:
            return 'Loyal Customers'

        # Potential Loyalists: Recent with moderate frequency/monetary
        elif r >= 4 and f >= 2 and m >= 2:
            return 'Potential Loyalists'

        # New Customers: Very recent but low frequency
        elif r >= 4 and f <= 2:
            return 'New Customers'

        # Promising: Recent but low spend and frequency
        elif r >= 3 and f <= 2 and m <= 2:
            return 'Promising'

        # Need Attention: Moderate recency, decent frequency/monetary
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Need Attention'

        # About to Sleep: Below average recency, moderate history
        elif r >= 2 and f >= 2 and m >= 2:
            return 'About to Sleep'

        # Cannot Lose Them: Were top customers (high F, M), now low R
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Cannot Lose Them'

        # At Risk: Historically good (moderate F, M), now low R
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'

        # Hibernating: Low recency, some history
        elif r <= 2 and f >= 2 and m >= 2:
            return 'Hibernating'

        # Lost: Low across all dimensions
        else:
            return 'Lost'

    print("\nAssigning customer segments...")
    rfm['Segment'] = rfm.apply(get_segment, axis=1)

    # Calculate segment statistics
    segment_stats = rfm.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum', 'std'],
        'AvgOrderValue': 'mean',
        'CustomerLifetime': 'mean'
    }).round(2)

    # Flatten column names
    segment_stats.columns = [
        'Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
        'Total_Revenue', 'Monetary_Std', 'Avg_Order_Value', 'Avg_Lifetime'
    ]

    segment_stats['Percentage'] = (
            segment_stats['Count'] / segment_stats['Count'].sum() * 100
    ).round(1)

    segment_stats['Revenue_Share'] = (
            segment_stats['Total_Revenue'] / segment_stats['Total_Revenue'].sum() * 100
    ).round(1)

    # Sort by total revenue (business priority)
    segment_stats = segment_stats.sort_values('Total_Revenue', ascending=False)

    print(f"\nSegment Distribution:")
    print(segment_stats[['Count', 'Percentage', 'Revenue_Share', 'Avg_Monetary']])

    return rfm, segment_stats


def get_segment_recommendations(segment: str) -> str:
    """
    Get business recommendation for a segment.
    """
    recommendations = {
        'Champions': 'Reward them. VIP program, early access, exclusive offers. They are your brand ambassadors.',
        'Loyal Customers': 'Upsell higher value products. Ask for reviews. Referral incentives.',
        'Potential Loyalists': 'Offer membership/loyalty program. Recommend related products. Increase frequency.',
        'New Customers': 'Welcome series. Onboarding emails. First-purchase follow-up. Build relationship.',
        'Promising': 'Free shipping offers. Limited-time discounts. Nurture to increase spend.',
        'Need Attention': 'Make limited time offers. Recommend based on past purchases. Re-engage.',
        'About to Sleep': 'Share valuable resources. Recommend popular products. Renew interest.',
        'At Risk': 'Send personalized reactivation campaigns. Special discounts. Win-back emails.',
        'Cannot Lose Them': 'Win them back at all costs. Personal outreach. Exclusive offers. Survey for feedback.',
        'Hibernating': 'Offer relevant products. Special reactivation deals. Recommend new arrivals.',
        'Lost': 'Cost-effective reactivation only. Remove from active marketing if no response.'
    }
    return recommendations.get(segment, 'No specific recommendation available.')


if __name__ == "__main__":
    # Test
    rfm = pd.read_csv('outputs/rfm_segments.csv')
    rfm, stats = segment_customers(rfm)
    print(stats)