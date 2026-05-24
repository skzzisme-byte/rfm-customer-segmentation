"""
Data loading utilities for the Streamlit dashboard
"""

import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_rfm_data():
    """Load RFM segmented data."""
    df = pd.read_csv('outputs/rfm_segments.csv')
    return df


@st.cache_data
def load_segment_summary():
    """Load segment summary statistics."""
    df = pd.read_csv('outputs/segment_summary.csv', index_col=0)
    return df


@st.cache_data
def load_clean_data():
    """Load cleaned transaction data."""
    df = pd.read_csv('data/processed/clean_data.csv', parse_dates=['InvoiceDate'])
    return df