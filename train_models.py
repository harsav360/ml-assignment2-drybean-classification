"""
train_models.py
---------------
Training script for ML Assignment 2 - Dry Bean Dataset Classification
This script:
1. Loads the Dry Bean dataset from a local CSV file
2. Preprocesses the data (scaling, encoding)
3. Trains 5 classification models
4. Evaluates each model on test data
5. Saves trained models, scaler, label encoder, and test data
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)


def load_and_preprocess_data(csv_path):
    """Load dataset and perform preprocessing."""
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target classes: {df['Class'].unique()}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    print()

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save test data as CSV (with original labels for readability)
    test_df = X_test.copy()
    test_df['Class'] = le.inverse_transform(y_test)
    test_df.to_csv(
        os.path.join(os.path.dirname(csv_path), 'test_data.csv'),
        index=False
    )
    print(f"Test data saved: {test_df.shape[0]} rows")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


def evaluate_model(model, X_test, y_test, n_classes):
    """Calculate all required evaluation metrics."""
    y_pred = model.predict(X_test)

    # For AUC, we need probability estimates
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
        # Convert decision function to probability-like scores for multi-class
        from scipy.special import softmax
        y_proba = softmax(y_scores, axis=1)
    else:
        y_proba = None

    metrics = {}
    metrics['Accuracy'] = round(accuracy_score(y_test, y_pred), 4)

    # AUC Score (One-vs-Rest for multi-class)
    if y_proba is not None:
        try:
            metrics['AUC'] = round(
                roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'),
                4
            )
        except Exception:
            metrics['AUC'] = 'N/A'
    else:
        metrics['AUC'] = 'N/A'

    # Precision, Recall, F1 (weighted average for multi-class)
    metrics['Precision'] = round(
        precision_score(y_test, y_pred, average='weighted', zero_division=0), 4
    )
    metrics['Recall'] = round(
        recall_score(y_test, y_pred, average='weighted', zero_division=0), 4
    )
    metrics['F1'] = round(
        f1_score(y_test, y_pred, average='weighted', zero_division=0), 4
    )
    metrics['MCC'] = round(matthews_corrcoef(y_test, y_pred), 4)

    return metrics


def train_and_save_models(X_train, X_test, y_train, y_test, scaler, le, model_dir):
    """Train all 5 models, evaluate, and save them."""

    n_classes = len(le.classes_)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=15, min_samples_split=5
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='minkowski'
        ),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(
            kernel='rbf', C=10, gamma='scale', random_state=42,
            probability=True, decision_function_shape='ovr'
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, n_classes)
        results[name] = metrics

        # Save model
        safe_name = name.lower().replace(' ', '_')
        model_path = os.path.join(model_dir, f'{safe_name}.joblib')
        joblib.dump(model, model_path)
        print(f"  Saved to {model_path}")
        print(f"  Metrics: {metrics}")
        print()

    # Save scaler and label encoder
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))

    # Save results as JSON for the app to use
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save feature names
    with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
        json.dump(list(le.classes_.tolist()), f)

    return results


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE")
    print("=" * 90)

    header = f"{'Model':<25} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}"
    print(header)
    print("-" * 90)

    for model_name, metrics in results.items():
        row = f"{model_name:<25}"
        for metric in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
            val = metrics.get(metric, 'N/A')
            if isinstance(val, float):
                row += f" {val:>10.4f}"
            else:
                row += f" {str(val):>10}"
        print(row)

    print("=" * 90)


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'Dry_Bean_Dataset.csv')
    model_dir = os.path.join(base_dir, 'model')

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Check if dataset exists
    if not os.path.exists(csv_path):
        # Auto-download from UCI if not found locally
        print("Dry_Bean_Dataset.csv not found locally. Downloading from UCI Repository...")
        try:
            import urllib.request, zipfile
            zip_tmp = os.path.join(base_dir, 'dry_bean_temp.zip')
            url = 'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip'
            urllib.request.urlretrieve(url, zip_tmp)
            with zipfile.ZipFile(zip_tmp, 'r') as z:
                z.extractall(base_dir)
            if os.path.exists(zip_tmp):
                os.remove(zip_tmp)
            xlsx_p = os.path.join(base_dir, 'DryBeanDataset', 'Dry_Bean_Dataset.xlsx')
            if os.path.exists(xlsx_p):
                df_temp = pd.read_excel(xlsx_p)
                df_temp.to_csv(csv_path, index=False)
                print(f"Dataset successfully downloaded and converted to CSV ({len(df_temp)} rows)")
        except Exception as e:
            print(f"Automatic download failed: {e}")
            print(f"Please place Dry_Bean_Dataset.csv at {csv_path}")
            return


    print("=" * 60)
    print("Dry Bean Dataset - Multi-Class Classification")
    print("=" * 60)
    print()

    # Load and preprocess
    X_train, X_test, y_train, y_test, scaler, le = load_and_preprocess_data(csv_path)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {list(le.classes_)}")
    print()

    # Train and evaluate all models
    results = train_and_save_models(X_train, X_test, y_train, y_test, scaler, le, model_dir)

    # Print comparison table
    print_comparison_table(results)

    print("\nAll models trained and saved successfully!")
    print(f"Model directory: {model_dir}")
    print(f"Test data saved to: {os.path.join(base_dir, 'test_data.csv')}")


if __name__ == '__main__':
    main()
