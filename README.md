_TASK_02_

**Credit Card Fraud Detection**

_Task Description_

This task aims to develop a machine learning model to detect fraudulent credit card transactions. Using a dataset of credit card transactions, we train and evaluate several classification algorithms, including Logistic Regression, Decision Trees, and Random Forests, to classify transactions as fraudulent or legitimate.

_Steps and Implementation_

1. Data Loading and Cleaning
   
1. Loaded the fraudTrain.csv and fraudTest.csv datasets.

2. Dropped unnecessary columns (Unnamed: 0, trans_num) and converted date columns to datetime format.

2. Feature Engineering
   
1. Extracted new features from the date columns, such as transaction hour and day of the week.

2. Dropped the original date columns after feature extraction.

3. Data Preprocessing
   
1. Split the data into features (X_train, X_test) and labels (y_train, y_test).

2. Defined preprocessing pipelines for numerical (imputation, scaling) and categorical (imputation, one-hot encoding) data.

3. Applied transformations using a column transformer.
   

4. Model Training and Evaluation
   
 Trained and evaluated three models:

1. Logistic Regression: Used balanced class weights and evaluated with classification report and ROC AUC score. Plotted ROC curve.

2. Decision Tree: Evaluated with classification report and ROC AUC score. Plotted confusion matrix.

3. Random Forest: Trained with balanced class weights, evaluated with classification report and ROC AUC score, and plotted feature importances.

5. Model Saving
   
   Saved the preprocessed data and the best-performing model for future use.

_Deliverables_

1. Preprocessed data files.

2. Trained models with evaluation metrics.

3. Visualizations: ROC curve, confusion matrix, and feature importances.

4. Best model file (best_model_balanced.pkl).


_TASK_03_

**Customer Churn Prediction**

In this task, I developed a model to predict customer churn for a subscription-based service using historical customer data. The goal was to identify customers who are likely to leave the service, based on features such as usage behavior and demographics.

Steps Performed:

_Data Preprocessing:_

1. Loaded and inspected the dataset.

2. Handled categorical features (Geography, Gender) using Label Encoding.

3. Dropped non-essential columns (RowNumber, CustomerId, Surname) and set Exited as the target variable.


_Feature Scaling:_

   Applied Standard Scaling to normalize features.


_Model Training:_

Implemented three classification algorithms:

1. Logistic Regression

2. Random Forest Classifier

3. Gradient Boosting Classifier

Trained each model on the training set.


_Model Evaluation:_

Evaluated model performance using accuracy, precision, recall, F1-score, and AUC-ROC.

Displayed evaluation metrics in bar plots for easy comparison.


_Visualization:_

Plotted performance metrics for each model to visualize and compare their effectiveness.


_TASK_04_

**Spam SMS Detection**

In this task, an AI model is built to classify SMS messages as either spam or legitimate (ham). The task involves using text classification techniques and various machine learning algorithms to identify spam messages accurately.

_Data Preprocessing_

1. Dataset: The dataset used is spam.csv, which contains SMS messages labeled as 'spam' or 'ham'.

2. Processing:

1. Removed unnecessary columns from the dataset.
   
2. Renamed columns for clarity.
   
3. Mapped categorical labels ('spam', 'ham') to binary values (1, 0).

_Feature Extraction_

1. Utilized TF-IDF Vectorizer to transform SMS messages into numerical features while ignoring common English stop words.

_Model Training and Evaluation_

1. Split the dataset into training and test sets.
   
2. Trained three different models:
   
1. Naive Bayes: MultinomialNB
   
2. Logistic Regression: LogisticRegression
   
3. Support Vector Machine (SVM): SVC
   
3. Evaluated model performance using accuracy, classification report, and confusion matrix.
   
_Visualization_

1. Plotted confusion matrices for each model to visualize classification performance.
   
2. Created a bar chart comparing the accuracy of the models.
   
_Model Saving_

1. Saved the best-performing model's accuracy score using joblib.
   
_This approach provides a comprehensive method for detecting spam messages using text classification techniques and evaluates different models to determine the most effective one._
