# Dry Bean Dataset - Multi-Class Classification

## a. Problem Statement

The objective of this project is to classify **dry beans** into one of **7 varieties** (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) based on their **geometric and shape features** extracted from images. Five different machine learning classification models are implemented and compared on their performance using standard evaluation metrics.

This is a **multi-class classification problem** with 7 classes and 16 numerical input features.

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Dataset Name** | Dry Bean Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| **Total Instances** | 13,611 |
| **Number of Features** | 16 (all numerical) |
| **Target Variable** | Class (7 bean varieties) |
| **Missing Values** | None |
| **Train/Test Split** | 80/20 (stratified), random_state=42 |

### Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | Area | The area of a bean zone and the number of pixels within its boundaries |
| 2 | Perimeter | Bean circumference, defined as the length of its border |
| 3 | MajorAxisLength | The distance between the ends of the longest line that can be drawn from a bean |
| 4 | MinorAxisLength | The longest line that can be drawn from the bean while standing perpendicular to the main axis |
| 5 | AspectRation | Defines the relationship between MajorAxisLength and MinorAxisLength |
| 6 | Eccentricity | Eccentricity of the ellipse having the same moments as the region |
| 7 | ConvexArea | Number of pixels in the smallest convex polygon that can contain the area of a bean seed |
| 8 | EquivDiameter | The diameter of a circle having the same area as a bean seed area |
| 9 | Extent | The ratio of the pixels in the bounding box to the bean area |
| 10 | Solidity | Also known as convexity. The ratio of the pixels in the convex shell to those found in beans |
| 11 | roundness | Calculated with the formula: (4*pi*Area)/(Perimeter^2) |
| 12 | Compactness | Measures the roundness of an object: Ed/MajorAxisLength |
| 13 | ShapeFactor1 | Shape factor 1 |
| 14 | ShapeFactor2 | Shape factor 2 |
| 15 | ShapeFactor3 | Shape factor 3 |
| 16 | ShapeFactor4 | Shape factor 4 |

### Target Classes Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| DERMASON | 3,546 | 26.1% |
| SIRA | 2,636 | 19.4% |
| SEKER | 2,027 | 14.9% |
| HOROZ | 1,928 | 14.2% |
| CALI | 1,630 | 12.0% |
| BARBUNYA | 1,322 | 9.7% |
| BOMBAY | 522 | 3.8% |

## c. GitHub Repository Link

**GitHub Repository:** [https://github.com/harsav360/ml-assignment2-drybean-classification](https://github.com/harsav360/ml-assignment2-drybean-classification)

**Live Streamlit App:** [Add your Streamlit Cloud link here after deployment]

### Repository Structure

```
project-folder/
│── app.py                          # Streamlit web application
│── train_models.py                 # Model training script
│── requirements.txt                # Python dependencies
│── README.md                       # This file
│── test_data.csv                   # Test data (20% split)
│── Dry_Bean_Dataset.csv            # Full dataset
│── model/
│   ├── logistic_regression.joblib  # Trained Logistic Regression model
│   ├── decision_tree.joblib        # Trained Decision Tree model
│   ├── knn.joblib                  # Trained KNN model
│   ├── naive_bayes.joblib          # Trained Naive Bayes model
│   ├── support_vector_machine.joblib # Trained SVM model
│   ├── scaler.joblib               # Fitted StandardScaler
│   ├── label_encoder.joblib        # Fitted LabelEncoder
│   └── results.json                # Pre-computed evaluation results
```

## d. Models Used

### Preprocessing Steps
1. **Label Encoding**: Target variable (Class) encoded using `LabelEncoder`
2. **Feature Scaling**: All 16 numerical features standardized using `StandardScaler`
3. **Train/Test Split**: 80/20 stratified split with `random_state=42`

### Model Configurations

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | `max_iter=1000, solver='lbfgs'` |
| Decision Tree | `max_depth=15, min_samples_split=5` |
| KNN | `n_neighbors=7, weights='distance', metric='minkowski'` |
| Naive Bayes | Gaussian Naive Bayes (default parameters) |
| SVM | `kernel='rbf', C=10, gamma='scale', probability=True` |

### Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8994 | 0.9424 | 0.8994 | 0.8994 | 0.8993 | 0.8784 |
| KNN | 0.9166 | 0.9839 | 0.9173 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9243 | 0.9933 | 0.9243 | 0.9243 | 0.9243 | 0.9085 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieves the second-best overall performance (92.14% accuracy). Despite being a linear model, it performs very well on this dataset likely because the features (geometric shape descriptors) have good linear separability between bean classes. The AUC of 0.9934 is the highest, indicating excellent class discrimination. The model is also computationally efficient with fast training and inference times. |
| Decision Tree | Lowest performer among the five models (89.94% accuracy) though still reasonably accurate. The AUC of 0.9424 is significantly lower than other models, suggesting the tree-based decision boundaries may not capture the continuous nature of shape features well. The model likely overfits on some classes despite the `max_depth=15` constraint, while struggling with classes that have overlapping feature distributions (e.g., SIRA vs DERMASON). |
| KNN | Good performance (91.66% accuracy) as a non-parametric instance-based learner. Benefits from the `weights='distance'` parameter which gives closer neighbors more influence. Works well because bean varieties that look similar tend to cluster together in the 16-dimensional feature space. The slightly lower AUC (0.9839) compared to LR and SVM suggests some boundary cases are harder for nearest-neighbor decisions. |
| Naive Bayes | Achieves the lowest accuracy (89.79%) among all models, which is expected given the strong assumption of feature independence that Gaussian Naive Bayes makes. Many of the geometric features (e.g., Area, ConvexArea, EquivDiameter) are highly correlated, violating this assumption. However, its AUC of 0.9902 is surprisingly high, indicating that the probability estimates still provide good ranking/discrimination even if the hard predictions are less accurate. |
| Support Vector Machine | **Best overall performer** with the highest accuracy (92.43%), precision (0.9243), recall (0.9243), F1 score (0.9243), and MCC (0.9085). The RBF kernel effectively captures non-linear relationships between features, and the `C=10` regularization provides good generalization. SVM excels at finding optimal decision boundaries in high-dimensional spaces, which is beneficial for this 16-feature dataset with 7 classes. |
| **Overall Winner** | **Support Vector Machine (SVM)** is the clear winner for this dataset, achieving the best scores across all metrics (accuracy: 92.43%, MCC: 0.9085). It slightly outperforms Logistic Regression due to its ability to model non-linear boundaries via the RBF kernel, while maintaining good generalization. |

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (if retraining is needed)
```bash
python train_models.py
```

### 3. Run Streamlit App Locally
```bash
streamlit run app.py
```

### 4. Deploy to Streamlit Community Cloud
1. Push the repository to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App" → Select this repository → Choose `app.py` → Deploy

## References
- Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
- UCI Machine Learning Repository: [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
