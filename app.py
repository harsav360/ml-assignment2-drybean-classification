"""
app.py - Streamlit Web Application
-----------------------------------
ML Assignment 2: Dry Bean Dataset Classification
Interactive app for demonstrating 5 classification models.

Features:
- CSV test data upload
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix visualization
- Classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ──────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Dry Bean Classifier - ML Assignment 2",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ──────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────

@st.cache_resource
def load_model(model_name):
    """Load a trained model from disk."""
    safe_name = model_name.lower().replace(' ', '_')
    model_path = os.path.join(MODEL_DIR, f'{safe_name}.joblib')
    return joblib.load(model_path)

@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler."""
    return joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))

@st.cache_resource
def load_label_encoder():
    """Load the fitted LabelEncoder."""
    return joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))

@st.cache_data
def load_training_results():
    """Load pre-computed training results."""
    results_path = os.path.join(MODEL_DIR, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Compute all evaluation metrics."""
    metrics = {}
    metrics['Accuracy'] = round(accuracy_score(y_true, y_pred), 4)

    # AUC Score
    if y_proba is not None:
        try:
            metrics['AUC'] = round(
                roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'),
                4
            )
        except Exception:
            metrics['AUC'] = 'N/A'
    else:
        metrics['AUC'] = 'N/A'

    metrics['Precision'] = round(
        precision_score(y_true, y_pred, average='weighted', zero_division=0), 4
    )
    metrics['Recall'] = round(
        recall_score(y_true, y_pred, average='weighted', zero_division=0), 4
    )
    metrics['F1'] = round(
        f1_score(y_true, y_pred, average='weighted', zero_division=0), 4
    )
    metrics['MCC'] = round(matthews_corrcoef(y_true, y_pred), 4)

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Create and return a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────

def main():
    # Title and description
    st.title("🫘 Dry Bean Dataset - Classification Models")
    st.markdown("""
    **ML Assignment 2** | M.Tech (AIML) - BITS Pilani WILP  
    This app demonstrates 5 classification models trained on the 
    [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) 
    from the UCI Machine Learning Repository.
    """)
    st.markdown("---")

    # ── Sidebar ──
    st.sidebar.header("⚙️ Configuration")

    # Model names
    model_names = [
        'Logistic Regression',
        'Decision Tree',
        'KNN',
        'Naive Bayes',
        'Support Vector Machine'
    ]

    selected_model = st.sidebar.selectbox(
        "🔍 Select Classification Model",
        model_names,
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.markdown("""
    - **Source:** UCI ML Repository  
    - **Instances:** 13,611  
    - **Features:** 16 (all numerical)  
    - **Classes:** 7 bean varieties  
    - **Task:** Multi-class classification
    """)

    # Feature descriptions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Features")
    st.sidebar.markdown("""
    1. Area  
    2. Perimeter  
    3. MajorAxisLength  
    4. MinorAxisLength  
    5. AspectRation  
    6. Eccentricity  
    7. ConvexArea  
    8. EquivDiameter  
    9. Extent  
    10. Solidity  
    11. roundness  
    12. Compactness  
    13. ShapeFactor1  
    14. ShapeFactor2  
    15. ShapeFactor3  
    16. ShapeFactor4  
    """)

    # ── File Upload ──
    st.subheader("📁 Upload Test Data (CSV)")
    st.info(
        "Upload a CSV file containing all 16 input features and the **Class** "
        "target column. The app will use the saved trained model to predict "
        "and evaluate — no retraining is performed."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded data: {df.shape[0]} rows × {df.shape[1]} columns")

            # Show data preview
            with st.expander("👀 Preview Uploaded Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # Validate columns
            expected_features = [
                'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
                'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
                'Extent', 'Solidity', 'roundness', 'Compactness',
                'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
            ]

            if 'Class' not in df.columns:
                st.error("❌ The CSV must contain a **Class** column (target).")
                return

            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                st.error(f"❌ Missing features in CSV: {missing_features}")
                return

            # Separate features and target
            X = df[expected_features]
            y_true_labels = df['Class']

            # Load preprocessing objects
            scaler = load_scaler()
            le = load_label_encoder()

            # Encode true labels
            try:
                y_true = le.transform(y_true_labels)
            except ValueError as e:
                st.error(f"❌ Unknown class label found: {e}")
                return

            # Scale features
            X_scaled = scaler.transform(X)

            # ── Results for Selected Model ──
            st.markdown("---")
            st.subheader(f"📈 Results: {selected_model}")

            model = load_model(selected_model)

            # Predict
            y_pred = model.predict(X_scaled)

            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_scaled)
            elif hasattr(model, 'decision_function'):
                from scipy.special import softmax
                y_scores = model.decision_function(X_scaled)
                y_proba = softmax(y_scores, axis=1)
            else:
                y_proba = None

            # Compute metrics
            metrics = compute_metrics(y_true, y_pred, y_proba, len(le.classes_))

            # Display metrics in columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("AUC Score", f"{metrics['AUC']}" if metrics['AUC'] != 'N/A' else 'N/A')
            col3.metric("Precision", f"{metrics['Precision']:.4f}")
            col4.metric("Recall", f"{metrics['Recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['F1']:.4f}")
            col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

            # Layout: Confusion Matrix + Classification Report
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### 🔲 Confusion Matrix")
                fig = plot_confusion_matrix(y_true, y_pred, le.classes_)
                st.pyplot(fig)
                plt.close(fig)

            with col_right:
                st.markdown("#### 📋 Classification Report")
                report = classification_report(
                    y_true, y_pred,
                    target_names=le.classes_,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(
                    report_df.style.format({
                        'precision': '{:.4f}',
                        'recall': '{:.4f}',
                        'f1-score': '{:.4f}',
                        'support': '{:.0f}'
                    }),
                    use_container_width=True
                )

            # ── Compare All Models ──
            st.markdown("---")
            st.subheader("📊 Compare All Models on Uploaded Test Data")

            all_results = {}
            for name in model_names:
                m = load_model(name)
                yp = m.predict(X_scaled)
                if hasattr(m, 'predict_proba'):
                    ypr = m.predict_proba(X_scaled)
                elif hasattr(m, 'decision_function'):
                    from scipy.special import softmax
                    ys = m.decision_function(X_scaled)
                    ypr = softmax(ys, axis=1)
                else:
                    ypr = None
                all_results[name] = compute_metrics(y_true, yp, ypr, len(le.classes_))

            comparison_df = pd.DataFrame(all_results).T
            comparison_df.index.name = 'Model'

            # Highlight best values
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )

            # Bar chart comparison
            st.markdown("#### 📊 Visual Comparison")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            numeric_cols = [c for c in comparison_df.columns if comparison_df[c].dtype != 'object']
            comparison_df[numeric_cols].plot(kind='bar', ax=ax2, rot=15)
            ax2.set_ylabel('Score')
            ax2.set_title('Model Performance Comparison')
            ax2.legend(loc='lower right')
            ax2.set_ylim(0, 1.05)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    else:
        # Show training results if available
        st.markdown("---")
        st.subheader("📊 Training Results (Pre-computed)")

        training_results = load_training_results()
        if training_results:
            st.markdown("These results are from the training/test split (80/20). "
                        "Upload your own test CSV above to see live predictions.")

            results_df = pd.DataFrame(training_results).T
            results_df.index.name = 'Model'

            st.dataframe(
                results_df.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )

            # Bar chart
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            numeric_cols = [c for c in results_df.columns if results_df[c].dtype != 'object']
            results_df[numeric_cols].plot(kind='bar', ax=ax3, rot=15)
            ax3.set_ylabel('Score')
            ax3.set_title('Model Performance Comparison (Training Results)')
            ax3.legend(loc='lower right')
            ax3.set_ylim(0, 1.05)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.warning("⚠️ No pre-computed results found. Please upload test data above.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
        "ML Assignment 2 | M.Tech AIML - BITS Pilani WILP | "
        "Dry Bean Dataset (UCI ML Repository)"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
