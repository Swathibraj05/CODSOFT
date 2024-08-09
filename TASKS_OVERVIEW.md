_TASK_02_

**Credit Card Fraud Detection**

_Task Description_

This project aims to develop a machine learning model to detect fraudulent credit card transactions. Using a dataset of credit card transactions, we train and evaluate several classification algorithms, including Logistic Regression, Decision Trees, and Random Forests, to classify transactions as fraudulent or legitimate.

_Steps and Implementation_

1. Data Loading and Cleaning
   
Loaded the fraudTrain.csv and fraudTest.csv datasets.

Dropped unnecessary columns (Unnamed: 0, trans_num) and converted date columns to datetime format.

2. Feature Engineering
   
Extracted new features from the date columns, such as transaction hour and day of the week.

Dropped the original date columns after feature extraction.

3. Data Preprocessing
   
Split the data into features (X_train, X_test) and labels (y_train, y_test).

Defined preprocessing pipelines for numerical (imputation, scaling) and categorical (imputation, one-hot encoding) data.

Applied transformations using a column transformer.

4. Model Training and Evaluation
   
Trained and evaluated three models:

Logistic Regression: Used balanced class weights and evaluated with classification report and ROC AUC score. Plotted ROC curve.

Decision Tree: Evaluated with classification report and ROC AUC score. Plotted confusion matrix.

Random Forest: Trained with balanced class weights, evaluated with classification report and ROC AUC score, and plotted feature importances.

5. Model Saving
   
Saved the preprocessed data and the best-performing model for future use.

_Deliverables_

Preprocessed data files.

Trained models with evaluation metrics.

Visualizations: ROC curve, confusion matrix, and feature importances.

Best model file (best_model_balanced.pkl).


_TASK_04_

**Customer Churn Prediction**

In this task, I developed a model to predict customer churn for a subscription-based service using historical customer data. The goal was to identify customers who are likely to leave the service, based on features such as usage behavior and demographics.

Steps Performed:

_Data Preprocessing:_

Loaded and inspected the dataset.

Handled categorical features (Geography, Gender) using Label Encoding.

Dropped non-essential columns (RowNumber, CustomerId, Surname) and set Exited as the target variable.


_Feature Scaling:_

Applied Standard Scaling to normalize features.


_Model Training:_

Implemented three classification algorithms:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

Trained each model on the training set.


_Model Evaluation:_

Evaluated model performance using accuracy, precision, recall, F1-score, and AUC-ROC.

Displayed evaluation metrics in bar plots for easy comparison.


_Visualization:_

Plotted performance metrics for each model to visualize and compare their effectiveness.
