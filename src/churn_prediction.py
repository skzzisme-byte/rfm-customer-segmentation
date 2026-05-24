"""
Churn Prediction Module
Predicts which customers are at risk of churning using RFM features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def create_churn_labels(rfm: pd.DataFrame, churn_threshold_days: int = 90) -> pd.DataFrame:
    """
    Create churn labels based on recency threshold.
    Customers who haven't purchased in X days are considered churned.
    """
    print(f"\nCreating churn labels (threshold: {churn_threshold_days} days)...")

    # Label: 1 = Churned (high recency), 0 = Active (low recency)
    rfm['Churned'] = (rfm['Recency'] > churn_threshold_days).astype(int)

    churn_rate = rfm['Churned'].mean()
    print(f"Overall churn rate: {churn_rate:.1%}")
    print(f"Churned customers: {rfm['Churned'].sum():,}")
    print(f"Active customers: {(rfm['Churned'] == 0).sum():,}")

    return rfm


def prepare_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target variable for modeling.
    """
    feature_cols = [
        'Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
        'OrderValueStd', 'MinOrderValue', 'MaxOrderValue',
        'TotalQuantity', 'CustomerLifetime', 'R_Score', 'F_Score', 'M_Score'
    ]

    # Handle any NaN values
    X = rfm[feature_cols].fillna(0)
    y = rfm['Churned']

    return X, y


def train_churn_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> dict:
    """
    Train and evaluate churn prediction model.

    Args:
        X: Feature matrix
        y: Target variable
        model_type: 'random_forest' or 'logistic_regression'

    Returns:
        Dictionary with model, metrics, and predictions
    """
    print(f"\nTraining {model_type} model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)  # RF doesn't need scaling
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"AUC-ROC: {auc:.3f}")
    print(f"Precision: {report['1']['precision']:.3f}")
    print(f"Recall: {report['1']['recall']:.3f}")
    print(f"F1-Score: {report['1']['f1-score']:.3f}")

    # Feature importance
    if model_type == 'random_forest':
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)

    print(f"\nTop 5 Important Features:")
    print(importance.head())

    return {
        'model': model,
        'scaler': scaler,
        'auc': auc,
        'report': report,
        'confusion_matrix': cm,
        'feature_importance': importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_prob': y_prob
    }


def predict_churn_risk(rfm: pd.DataFrame, model, scaler=None) -> pd.DataFrame:
    """
    Predict churn probability for all customers.
    """
    feature_cols = [
        'Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
        'OrderValueStd', 'MinOrderValue', 'MaxOrderValue',
        'TotalQuantity', 'CustomerLifetime', 'R_Score', 'F_Score', 'M_Score'
    ]

    X = rfm[feature_cols].fillna(0)

    if scaler:
        X = scaler.transform(X)

    rfm['Churn_Probability'] = model.predict_proba(X)[:, 1]
    rfm['Churn_Risk'] = pd.cut(
        rfm['Churn_Probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return rfm


def save_model(model_dict: dict, output_dir: str = 'outputs/') -> None:
    """Save trained model and scaler."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model_dict['model'], f'{output_dir}churn_model.pkl')
    if model_dict.get('scaler'):
        joblib.dump(model_dict['scaler'], f'{output_dir}churn_scaler.pkl')
    print(f"Model saved to {output_dir}churn_model.pkl")


def load_model(model_path: str = 'outputs/churn_model.pkl'):
    """Load trained model."""
    return joblib.load(model_path)


if __name__ == "__main__":
    # Full pipeline test
    rfm = pd.read_csv('outputs/rfm_segments.csv')
    rfm = create_churn_labels(rfm, churn_threshold_days=90)
    X, y = prepare_features(rfm)

    # Train both models
    rf_results = train_churn_model(X, y, 'random_forest')
    lr_results = train_churn_model(X, y, 'logistic_regression')

    # Save best model (RF usually performs better)
    save_model(rf_results)

    # Predict on full dataset
    rfm = predict_churn_risk(rfm, rf_results['model'])
    print(rfm[['CustomerID', 'Churn_Probability', 'Churn_Risk']].head(10))