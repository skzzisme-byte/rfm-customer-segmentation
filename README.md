# 🎯 RFM Customer Segmentation & Churn Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rfm-customer-segmentation-3adudxm7xac3p2afpe5yyl.streamlit.app/)

## Project Overview
A comprehensive customer analytics project that segments customers using RFM 
(Recency, Frequency, Monetary) analysis and predicts churn risk using machine 
learning. Built with production-ready Python code and an interactive Streamlit 
dashboard.

## 🚀 Live Dashboard
[Click here to view the live dashboard](https://rfm-customer-segmentation-3adudxm7xac3p2afpe5yyl.streamlit.app/)

## 📊 Business Problem
An online retail company needs to:
- Identify high-value customers for retention
- Detect at-risk customers before they churn
- Optimize marketing spend by segment
- Increase customer lifetime value

## 🛠️ Tech Stack
| Category | Tools |
|----------|-------|
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (Random Forest, Logistic Regression) |
| Dashboard | Streamlit |
| Environment | PyCharm, Python 3.9+ |

## 📁 Project Structure
rfm-customer-segmentation/
├── src/                    # Core analysis modules
│   ├── data_cleaning.py    # Data quality & preprocessing
│   ├── rfm_calculation.py  # RFM metric computation
│   ├── segmentation.py     # Business segment assignment
│   ├── visualization.py    # Chart generation
│   └── churn_prediction.py # ML churn prediction
├── dashboard/              # Streamlit application
│   ├── app.py             # Main dashboard
│   └── utils/             # Data loading utilities
├── data/                   # Raw & processed data
├── outputs/               # Results & visualizations
└── main.py                # Pipeline orchestrator


## 🔑 Key Features
- **Data Cleaning**: Handles 135K missing IDs, returns, duplicates, outliers
- **RFM Analysis**: 11 meaningful customer segments with business logic
- **Churn Prediction**: Random Forest model with 85%+ AUC
- **Interactive Dashboard**: Filter, explore, and download customer lists
- **Actionable Insights**: Segment-specific marketing recommendations

## 📈 Key Results
| Metric | Value |
|--------|-------|
| Total Customers | 4,228 |
| Revenue Analyzed | $5.3M |
| Churn Prediction AUC | 1.000 |
| High-Risk Customers | 1,400 (33%) |

## 🚀 How to Run

### 1. Clone & Setup
```bash
git clone https://github.com/skzzisme-byte/rfm-customer-segmentation.git
cd rfm-customer-segmentation
pip install -r requirements.txt
```
### 2. Run Analysis Pipeline
```bash
python main.py
```
### 3. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

💡 Business Recommendations

    Champions (22.5%): VIP program, early access, exclusive offers
    At Risk (1.5%): Immediate win-back campaign with 20% discount
    New Customers (4.5%): Welcome series, onboarding emails

📧 Contact
[Sk Ashadu - skzzisme@gmail.com - www.linkedin.com/in/sk15azz]
