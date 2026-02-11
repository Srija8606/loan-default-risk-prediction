# loan-default-risk-prediction
## Overview

This project builds a machine learning system to predict the probability of loan default using applicant financial and demographic features. The goal is to support risk-aware lending decisions in financial institutions.

## Problem Statement

Loan defaults lead to significant financial losses. This project aims to develop a classification model that identifies high-risk applicants while balancing false positives and false negatives.

## Dataset

Credit risk dataset containing applicant demographics, employment information, loan characteristics, and repayment status.

## Methodology

- Preprocessing using ColumnTransformer
- Median imputation for missing values
- OneHotEncoding for categorical variables
- Separate pipelines for Logistic Regression and Decision Tree
- Evaluation using ROC–AUC
- 5-Fold Cross-Validation for robustness

## Results

- Decision Tree ROC–AUC: ~0.878
- Logistic Regression ROC–AUC: ~0.868
- Final model selected based on cross-validated performance

## Key Skills Demonstrated

- Machine Learning Pipelines
- Handling Class Imbalance
- Model Comparison & Selection
- Cross-Validation
- Financial Risk Modeling
