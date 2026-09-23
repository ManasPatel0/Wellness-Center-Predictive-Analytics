# Wellness Center Predictive Analytics

A machine learning project for analyzing wellness center data and applying predictive analytics techniques for Doctor Count prediction, Wellness Center Category classification, clustering, and dimensionality reduction.

## Project Overview

This project implements an end-to-end machine learning workflow on wellness center data. The workflow includes data preprocessing, exploratory analysis, regression, classification, clustering, Principal Component Analysis (PCA), neural network modeling, and cross-validation.

The project evaluates multiple machine learning algorithms using appropriate performance metrics.

## Objectives

- Analyze the distribution of doctors across wellness centers.
- Explore relationships between numerical and categorical features.
- Predict `DoctorCount` using regression models.
- Classify wellness centers into different categories.
- Compare multiple classification algorithms.
- Perform K-Means clustering to identify data groups.
- Apply PCA for dimensionality reduction and visualization.
- Evaluate model performance using cross-validation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Machine Learning Techniques

### Regression

- Linear Regression
- Random Forest Regression

### Classification

- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest Classifier
- Multi-Layer Perceptron (MLP)

### Unsupervised Learning

- K-Means Clustering
- Elbow Method
- Principal Component Analysis (PCA)

### Model Evaluation

- R² Score
- RMSE
- Accuracy
- Precision
- Recall
- F1 Score
- 5-Fold Cross-Validation

## Dataset

The project uses a wellness center dataset containing location, category, doctor count, and other center-related attributes.

The dataset is stored in:

```text
data/wellness_center_data.csv
