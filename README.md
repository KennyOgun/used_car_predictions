# Used Car Sales Prediction using Machine Learning

![image](https://github.com/user-attachments/assets/483db985-29c5-4a1f-99a1-7404700d8a1a)


## Overview

This project focuses on predicting the prices of used cars using machine learning techniques. By leveraging advanced feature engineering, dimensionality reduction, and multiple machine learning models, the goal is to create a reliable prediction system to help buyers and sellers in the used car market.

---

## Table of Contents
1. [Features and Dataset](#features-and-dataset)
2. [Methodology](#methodology)
3. [Results and Insights](#results-and-insights)
4. [Models Used](#models-used)
5. [SHAP Interpretability](#shap-interpretability)
6. [Installation](#installation)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Features and Dataset

### Dataset
The dataset includes features relevant to predicting used car prices:
- **Categorical Features**: `car_colour`, `car_brand`, `body_type`, `fuel_type`, `car_model`, `crossover`
- **Numerical Features**: `car_age`, `mileage`
- **Target Variable**: Car Price (in dollars)

### Key Features
- **Categorical**: One-hot encoding is applied to features like `car_colour`, `car_brand`, etc.
- **Numerical**: Standardization and scaling were performed for features such as `car_age` and `mileage`.

---

## Methodology

### 1. **Feature Engineering**
   - Applied automated feature selection using `SelectKBest`.
   - Selected top features: `['car_colour', 'car_brand', 'body_type', 'fuel_type', 'car_age', 'mileage']`.

### 2. **Dimensionality Reduction**
   - Principal Component Analysis (PCA) was used to reduce dimensionality while preserving variance.
   - **Explained Variance Ratio**:
     - Top 4 components explained 68.45% of the variance.
   - **Loadings**: Highlighted the influence of each feature on the principal components.

### 3. **Model Selection and Training**
   - Models Trained:
     - Linear Regression
     - Random Forest Regressor
     - XGBoost Regressor
   - Hyperparameter Tuning:
     - Applied GridSearchCV for optimizing parameters.

---

## Results and Insights

### Model Performance:
| Model               | Train R²  | Test R²  |
|---------------------|-----------|----------|
| **XGBoost**         | 0.907583  | 0.868468 |
| **Random Forest**   | 0.980402  | 0.850564 |
| **Linear Regression** | 0.467682  | 0.458997 |

- **Best Model**: XGBoost achieved the highest performance with a Test R² of 0.868.

### Prediction Examples:
| Actual Price | Linear Regression | Random Forest | XGBoost |
|--------------|--------------------|---------------|---------|
| 6599         | 6547.36           | 6783.18       | 8035.96 |
| 9000         | 12845.77          | 8515.57       | 8701.37 |

---

## SHAP Interpretability

SHAP values were used to interpret the impact of features on model predictions:
### Summary of Feature Importance:
1. **Top Features** (XGBoost):
   - `car_brand` (most influential)
   - `car_age`
   - `body_type`
   - `mileage`

2. **Mean SHAP Values** (Linear Regression, Random Forest, and XGBoost):
   - Consistently important features include `car_age`, `car_brand`, and `mileage`.

---

## Installation and Usage

### Prerequisites
- Python 3.7 or higher
- Libraries: `numpy`, `pandas`, `sklearn`, `xgboost`, `matplotlib`, `shap`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/used-car-sales-prediction.git

**Install required libraries:**
pip install -r requirements.txt

**Conclusion**
This project demonstrates the application of machine learning in predicting used car prices. XGBoost emerged as the best-performing model, and SHAP analysis
provided valuable interpretability insights, highlighting **car_brand, car_age, and mileage** as the most influential factors. These insights can assist 
both sellers and buyers in making informed decisions in the used car market.

**References**
  * SHAP: Explainable AI [https://github.com/slundberg/shap]
  * Scikit-learn Documentation [https://scikit-learn.org]
  * XGBoost Documentation [https://xgboost.readthedocs.io]

