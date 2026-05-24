"""
Visualization Module
Creates publication-quality charts for RFM analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def create_all_visualizations(rfm, segment_stats, model_results=None, output_dir='outputs/figures/'):
    """
    Create all visualizations for the analysis.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nCreating visualizations...")

    # 1. Segment Distribution
    _create_segment_pie(rfm, output_dir)

    # 2. RFM Distributions
    _create_rfm_distributions(rfm, output_dir)

    # 3. Segment Revenue
    _create_segment_revenue(segment_stats, output_dir)

    # 4. RFM Heatmap
    _create_rfm_heatmap(rfm, output_dir)

    # 5. Scatter Plot
    _create_scatter_plot(rfm, output_dir)

    # 6. Treemap-style
    _create_treemap(segment_stats, output_dir)

    # 7. Churn prediction visualizations (if available)
    if model_results:
        _create_churn_visualizations(rfm, model_results, output_dir)

    print(f"All visualizations saved to {output_dir}")


def _create_segment_pie(rfm, output_dir):
    """Create segment distribution pie chart."""
    fig, ax = plt.subplots(figsize=(12, 8))

    segment_counts = rfm['Segment'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))

    wedges, texts, autotexts = ax.pie(
        segment_counts,
        labels=segment_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )

    plt.setp(autotexts, size=9, weight="bold")
    ax.set_title('Customer Segment Distribution', fontsize=18, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}segment_distribution.png', bbox_inches='tight')
    plt.close()


def _create_rfm_distributions(rfm, output_dir):
    """Create RFM metric distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['Recency', 'Frequency', 'Monetary']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    titles = ['Recency (Days Since Last Purchase)',
              'Frequency (Number of Orders)',
              'Monetary (Total Spend)']

    for ax, metric, color, title in zip(axes, metrics, colors, titles):
        ax.hist(rfm[metric], bins=50, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')

        # Add mean line
        mean_val = rfm[metric].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.legend()

    plt.suptitle('RFM Metrics Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}rfm_distributions.png', bbox_inches='tight')
    plt.close()


def _create_segment_revenue(segment_stats, output_dir):
    """Create segment revenue bar chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    segment_revenue = segment_stats.sort_values('Total_Revenue', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(segment_revenue)))

    bars = ax.barh(segment_revenue.index, segment_revenue['Total_Revenue'], color=colors)

    ax.set_xlabel('Total Revenue ($)', fontsize=12, fontweight='bold')
    ax.set_title('Revenue Contribution by Customer Segment', fontsize=16, fontweight='bold')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        pct = segment_revenue.iloc[i]['Revenue_Share']
        ax.text(width + 1000, bar.get_y() + bar.get_height() / 2,
                f'${width:,.0f} ({pct}%)', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}segment_revenue.png', bbox_inches='tight')
    plt.close()


def _create_rfm_heatmap(rfm, output_dir):
    """Create RFM profile heatmap by segment."""
    fig, ax = plt.subplots(figsize=(12, 10))

    heatmap_data = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()

    # Normalize for better color visualization
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    sns.heatmap(
        heatmap_normalized,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Normalized Score (0=Low, 1=High)'},
        linewidths=0.5
    )

    ax.set_title('Average RFM Profile by Segment\n(Green=High, Red=Low)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}rfm_heatmap.png', bbox_inches='tight')
    plt.close()


def _create_scatter_plot(rfm, output_dir):
    """Create Frequency vs Monetary scatter plot."""
    fig, ax = plt.subplots(figsize=(14, 10))

    segments = rfm['Segment'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))

    for i, segment in enumerate(segments):
        segment_data = rfm[rfm['Segment'] == segment]
        ax.scatter(
            segment_data['Frequency'],
            segment_data['Monetary'],
            c=[colors[i]],
            label=segment,
            alpha=0.6,
            s=60,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('Frequency (Number of Orders)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monetary Value ($)', fontsize=12, fontweight='bold')
    ax.set_title('Customer Segments: Frequency vs Monetary Value', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # Set limits to exclude extreme outliers for better visualization
    ax.set_xlim(0, rfm['Frequency'].quantile(0.99))
    ax.set_ylim(0, rfm['Monetary'].quantile(0.99))

    plt.tight_layout()
    plt.savefig(f'{output_dir}frequency_monetary_scatter.png', bbox_inches='tight')
    plt.close()


def _create_treemap(segment_stats, output_dir):
    """Create treemap-style visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))

    segment_stats_sorted = segment_stats.sort_values('Total_Revenue', ascending=False)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(segment_stats_sorted)))

    bars = ax.bar(
        range(len(segment_stats_sorted)),
        segment_stats_sorted['Total_Revenue'],
        color=colors,
        edgecolor='black',
        linewidth=1
    )

    ax.set_xticks(range(len(segment_stats_sorted)))
    ax.set_xticklabels(segment_stats_sorted.index, rotation=45, ha='right')
    ax.set_ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
    ax.set_title('Revenue by Segment (Descending Order)', fontsize=16, fontweight='bold')

    # Add percentage labels
    total_rev = segment_stats_sorted['Total_Revenue'].sum()
    for bar in bars:
        height = bar.get_height()
        pct = height / total_rev * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + total_rev * 0.01,
            f'${height:,.0f}\n({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(f'{output_dir}revenue_treemap.png', bbox_inches='tight')
    plt.close()


def _create_churn_visualizations(rfm, model_results, output_dir):
    """Create churn prediction visualizations."""

    # 1. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = model_results['feature_importance'].head(10)

    bars = ax.barh(importance['Feature'], importance['Importance'], color='steelblue')
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Churn Prediction Features', fontsize=16, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{output_dir}feature_importance.png', bbox_inches='tight')
    plt.close()

    # 2. Churn Risk by Segment
    fig, ax = plt.subplots(figsize=(12, 6))

    churn_by_segment = pd.crosstab(rfm['Segment'], rfm['Churn_Risk'], normalize='index') * 100
    churn_by_segment.plot(kind='bar', stacked=True, ax=ax,
                          color=['#2ecc71', '#f39c12', '#e74c3c'])

    ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Churn Risk Distribution by Segment', fontsize=16, fontweight='bold')
    ax.legend(title='Churn Risk', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}churn_by_segment.png', bbox_inches='tight')
    plt.close()

    # 3. Churn Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(rfm['Churn_Probability'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(rfm['Churn_Probability'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {rfm["Churn_Probability"].mean():.2f}')
    ax.set_xlabel('Churn Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Churn Probabilities', fontsize=16, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}churn_probability_dist.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test
    rfm = pd.read_csv('outputs/rfm_segments.csv')
    stats = pd.read_csv('outputs/segment_summary.csv', index_col=0)
    create_all_visualizations(rfm, stats)