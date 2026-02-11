# Loan Default Risk Prediction System
## Overview

This project builds a machine learning system to predict the probability of loan default using applicant financial and demographic information. The objective is to support risk-aware lending decisions in financial institutions.

## Problem Statement

Loan defaults lead to significant financial losses. Financial institutions must balance between approving profitable customers and avoiding high-risk applicants. This project aims to develop a classification model that identifies potential defaulters while maintaining balanced precision and recall.

## Dataset

The dataset contains applicant information including:

 -Demographic details
 -Employment history
 -Loan characteristics
 -Credit history
 -Loan repayment status (target variable)

## Methodology

Data preprocessing using ColumnTransformer

 - Median imputation for missing values
 - OneHotEncoding for categorical variables
 - Separate pipelines for:
   - Logistic Regression
   - Decision Tree Classifier
 - Evaluation using ROC–AUC
 - 5-Fold Cross-Validation for model stability

## Results
### Model	,              Test ROC–AUC	 ,   CV ROC–AUC
- Logistic Regression ,	  ~0.873 ,          ~0.868
- Decision Tree	,          ~0.878 ,	          ~0.878

Final model selected: Decision Tree Classifier based on slightly higher cross-validated performance and balanced precision–recall trade-off.

## Key Skills Demonstrated

 - Machine Learning Pipelines
 - Handling Class Imbalance
 - Model Comparison & Selection
 - Cross-Validation
 - Financial Risk Modeling

## Future Improvements

 - Implement ensemble methods (Random Forest, Gradient Boosting)
 - Add feature importance visualization
 - Deploy model as a web application (Flask / FastAPI)
