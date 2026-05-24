"""
Streamlit Dashboard - RFM Customer Segmentation Analysis
Professional interactive dashboard for portfolio presentation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'utils'))

from data_loader import load_rfm_data, load_segment_summary, load_clean_data
from segmentation import get_segment_recommendations

# Page configuration
st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .segment-card {
        background-color: #ffffff;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">📊 RFM Customer Segmentation Dashboard</p>',
                unsafe_allow_html=True)

    # Load data
    try:
        rfm = load_rfm_data()
        segment_stats = load_segment_summary()
        clean_data = load_clean_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run the main analysis pipeline first: `python main.py`")
        return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Segment filter
    selected_segments = st.sidebar.multiselect(
        "Select Segments",
        options=rfm['Segment'].unique(),
        default=rfm['Segment'].unique()
    )

    # RFM Score filter
    min_rfm = st.sidebar.slider("Minimum RFM Value",
                                float(rfm['RFM_Value'].min()),
                                float(rfm['RFM_Value'].max()),
                                float(rfm['RFM_Value'].min()))

    # Churn risk filter (if available)
    if 'Churn_Risk' in rfm.columns:
        selected_risk = st.sidebar.multiselect(
            "Churn Risk Level",
            options=['Low', 'Medium', 'High'],
            default=['Low', 'Medium', 'High']
        )

    # Filter data
    filtered_rfm = rfm[rfm['Segment'].isin(selected_segments)]
    filtered_rfm = filtered_rfm[filtered_rfm['RFM_Value'] >= min_rfm]

    if 'Churn_Risk' in rfm.columns and 'selected_risk' in locals():
        filtered_rfm = filtered_rfm[filtered_rfm['Churn_Risk'].isin(selected_risk)]

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Customers", f"{len(filtered_rfm):,}")
    with col2:
        st.metric("Total Revenue", f"${filtered_rfm['Monetary'].sum():,.0f}")
    with col3:
        st.metric("Avg Order Value", f"${filtered_rfm['AvgOrderValue'].mean():.2f}")
    with col4:
        st.metric("Avg Recency", f"{filtered_rfm['Recency'].mean():.0f} days")
    with col5:
        if 'Churn_Probability' in filtered_rfm.columns:
            st.metric("Avg Churn Risk", f"{filtered_rfm['Churn_Probability'].mean():.1%}")
        else:
            st.metric("Segments", f"{filtered_rfm['Segment'].nunique()}")

    st.markdown("---")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Segment Explorer", "⚠️ Churn Prediction"])

    with tab1:
        show_overview(filtered_rfm, segment_stats)

    with tab2:
        show_segment_explorer(filtered_rfm)

    with tab3:
        if 'Churn_Probability' in rfm.columns:
            show_churn_analysis(filtered_rfm)
        else:
            st.info("Churn prediction model not trained yet. Run `python main.py` to generate predictions.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Built with Python, Pandas, Scikit-learn, and Streamlit | Data Analyst Portfolio Project</p>
    </div>
    """, unsafe_allow_html=True)


def show_overview(rfm, segment_stats):
    """Display overview visualizations."""
    st.subheader("Customer Segment Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Segment distribution pie chart
        segment_counts = rfm['Segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Distribution by Segment",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Revenue by segment
        segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=True)
        fig = px.bar(
            x=segment_revenue.values,
            y=segment_revenue.index,
            orientation='h',
            title="Revenue by Segment",
            labels={'x': 'Total Revenue ($)', 'y': 'Segment'},
            color=segment_revenue.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    # RFM 3D scatter (or 2D with color)
    st.subheader("RFM Segment Map")

    fig = px.scatter(
        rfm,
        x='Frequency',
        y='Monetary',
        color='Segment',
        size='Recency',
        hover_data=['CustomerID', 'R_Score', 'F_Score', 'M_Score'],
        title="Frequency vs Monetary Value (Size = Recency)",
        labels={'Frequency': 'Number of Orders', 'Monetary': 'Total Spend ($)'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(marker=dict(opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

    # Segment summary table
    st.subheader("Segment Summary")
    st.dataframe(
        segment_stats[['Count', 'Percentage', 'Revenue_Share', 'Avg_Monetary', 'Avg_Frequency', 'Avg_Recency']],
        use_container_width=True
    )


def show_segment_explorer(rfm):
    """Interactive segment explorer."""
    st.subheader("🔍 Segment Deep Dive")

    # Segment selector
    selected_segment = st.selectbox(
        "Choose a segment to explore",
        options=rfm['Segment'].unique()
    )

    segment_data = rfm[rfm['Segment'] == selected_segment]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {selected_segment}")
        st.markdown(f"**Customers:** {len(segment_data):,}")
        st.markdown(f"**Revenue:** ${segment_data['Monetary'].sum():,.2f}")
        st.markdown(f"**Avg Recency:** {segment_data['Recency'].mean():.1f} days")
        st.markdown(f"**Avg Frequency:** {segment_data['Frequency'].mean():.1f} orders")
        st.markdown(f"**Avg Monetary:** ${segment_data['Monetary'].mean():.2f}")

        # Recommendation
        st.markdown("---")
        st.markdown("### 💡 Recommended Action")
        st.info(get_segment_recommendations(selected_segment))

    with col2:
        # RFM profile for this segment
        fig = go.Figure()

        avg_rfm = segment_data[['Recency', 'Frequency', 'Monetary']].mean()
        overall_avg = rfm[['Recency', 'Frequency', 'Monetary']].mean()

        # Normalize for radar chart
        categories = ['Recency (Lower=Better)', 'Frequency', 'Monetary']
        segment_values = [
            1 - (avg_rfm['Recency'] / overall_avg['Recency']),  # Invert recency
            avg_rfm['Frequency'] / overall_avg['Frequency'],
            avg_rfm['Monetary'] / overall_avg['Monetary']
        ]

        fig.add_trace(go.Scatterpolar(
            r=segment_values + [segment_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=selected_segment
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(segment_values) * 1.2])),
            showlegend=False,
            title=f"{selected_segment} RFM Profile (vs Average)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Customer table
    st.subheader(f"Top Customers in {selected_segment}")
    top_customers = segment_data.nlargest(10, 'Monetary')[
        ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue']]
    st.dataframe(top_customers, use_container_width=True)


def show_churn_analysis(rfm):
    """Churn prediction analysis."""
    st.subheader("⚠️ Churn Risk Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        high_risk = len(rfm[rfm['Churn_Risk'] == 'High'])
        st.metric("High Risk Customers", f"{high_risk:,}",
                  delta=f"{high_risk / len(rfm):.1%} of total")

    with col2:
        medium_risk = len(rfm[rfm['Churn_Risk'] == 'Medium'])
        st.metric("Medium Risk Customers", f"{medium_risk:,}")

    with col3:
        at_risk_revenue = rfm[rfm['Churn_Risk'] == 'High']['Monetary'].sum()
        st.metric("At-Risk Revenue", f"${at_risk_revenue:,.0f}")

    # Churn risk by segment
    st.subheader("Churn Risk by Segment")

    churn_by_segment = pd.crosstab(rfm['Segment'], rfm['Churn_Risk'], normalize='index') * 100

    fig = px.bar(
        churn_by_segment,
        barmode='group',
        title="Churn Risk Distribution by Segment (%)",
        labels={'value': 'Percentage', 'Churn_Risk': 'Risk Level'},
        color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # High risk customers table
    st.subheader("🚨 Priority: High Risk Customers")
    high_risk_customers = rfm[rfm['Churn_Risk'] == 'High'].nlargest(20, 'Monetary')

    display_cols = ['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary',
                    'Churn_Probability', 'AvgOrderValue']
    st.dataframe(high_risk_customers[display_cols], use_container_width=True)

    # Download button
    csv = high_risk_customers.to_csv(index=False)
    st.download_button(
        label="📥 Download High Risk Customers CSV",
        data=csv,
        file_name='high_risk_customers.csv',
        mime='text/csv'
    )


if __name__ == "__main__":
    main()