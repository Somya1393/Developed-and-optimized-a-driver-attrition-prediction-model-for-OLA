# Developed and optimized a driver attrition prediction model for OLA
 Ola Driver Attrition prediction

 # Driver Attrition Prediction

This project aims to predict driver attrition based on various features such as demographics, quarterly ratings, income, and other relevant factors. The data is preprocessed and analyzed using Python, and machine learning models are built to classify whether a driver is likely to leave.

## Table of Contents

1. [Introduction]
2. [Installation]
3. [Data Preprocessing]
4. [Exploratory Data Analysis]
5. [Model Training]
6. [Results]


## Introduction

This project utilizes a dataset of drivers to predict attrition. It includes steps for data preprocessing, handling missing values, feature engineering, and implementing machine learning models with evaluation metrics.

## Installation

To run this project, ensure you have the following Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

Data Preprocessing
Load Data:

Load the ola_driver_scaler.csv file.

Inspect the shape and basic information of the DataFrame.
Clean Data:

Drop unnecessary columns such as Unnamed: 0.
Check for missing values and fill or impute where needed.
Convert date-related columns like MMM-YY, Dateofjoining, and LastWorkingDate to datetime format.
Remove duplicates.

Imputation:

Use KNN Imputer to fill missing values for features like Age.
Visualize distributions of categorical variables such as Gender and Education_Level.

Feature Engineering:

Create new columns to capture trends in quarterly ratings, income growth, and business value.
Implement rating growth based on quarterly rating trends.

Exploratory Data Analysis

Visualize Categorical Variables:

Visualize distributions of categorical features like Gender, Education_Level, and City.
Plot count plots and bar plots for categorical features.

Visualize Continuous Variables:

Plot density and box plots for continuous variables such as Age, Income, and Total Business Value.
Analyze the relationship between continuous variables and attrition.

Bivariate Analysis:

Examine the relationship of categorical and continuous variables with the target variable, Attrited.
Model Training

Train-Test Split:

Split the data into train and test sets.
Use SMOTE to handle class imbalance in the target variable.

Decision Tree Classifier:

Perform hyperparameter tuning using GridSearchCV for max_depth, criterion, and min_samples_leaf.
Evaluate the model with metrics such as accuracy, precision, recall, and F1 score.

Random Forest Classifier:

Implement Random Forest with hyperparameter tuning on n_estimators, max_depth, and min_samples_leaf.
Evaluate the Random Forest model on test data, and calculate metrics including ROC-AUC score.

Decision Tree:
Training Accuracy: Approximately 85%
Test Accuracy: Approximately 75%
Evaluation Metrics: Precision, Recall, F1 Score
Random Forest:
Achieved higher accuracy and robustness against overfitting.
ROC-AUC score for Random Forest model provided an improved measure of model performance.
