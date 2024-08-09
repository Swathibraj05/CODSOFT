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
