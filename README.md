# Telecoms Customer Churn Prediction — Documentation

**Overview**

I built a binary classification model to predict whether a telecom customer will churn, using the public IBM Telco Customer Churn dataset. I ran it end to end from raw CSV to a serialized model ready for inference, originally developed in Google Colab against a file stored in Google Drive.

**Data preparation**

I loaded the dataset and dropped the customerID column since it carries no predictive signal. I checked the TotalCharges column for blank string values (a known quirk of this dataset where a handful of new customers have no billing history yet), replaced those blanks with "0.0", and cast the column to float. I then reviewed the unique values in every column to understand category structure before modeling.

**Exploratory analysis**

I visualized tenure, MonthlyCharges, and TotalCharges through histograms with mean and median markers and through boxplots to spot outliers and skew. I also plotted a correlation heatmap covering the three numeric fields, and gave every categorical column, including SeniorCitizen treated as categorical, a count plot to show class balance within each feature.

**Target and feature encoding**

I mapped the Churn target from Yes/No to 1/0. I label encoded all remaining object-typed columns and saved the fitted encoders to encoders.pkl so I could replay the same transformation on new data at inference time.

**Train-test split and class imbalance**

I split features and target into an 80/20 train-test split with a fixed random seed. Because churn is imbalanced, I applied SMOTE oversampling to the training set only, generating synthetic minority-class examples so my models trained on a balanced set without leaking synthetic data into the test set.

**Model selection**

I compared three classifiers with default hyperparameters using 5-fold cross-validation on the SMOTE-balanced training data: Decision Tree, Random Forest, and XGBoost. I selected Random Forest as my final model based on its cross-validation accuracy.

**Evaluation**

I fit the chosen Random Forest on the balanced training data and evaluated it on the untouched test set using accuracy score, a confusion matrix, and a full classification report covering precision, recall, and F1 by class.

**Persistence and inference**

I bundled the trained model and the list of feature names into a dictionary and pickled it to customer_churn_model.pkl. I then demonstrated loading that pickle back, along with my saved encoders, constructing a single-row example customer as a dataframe, encoding its categorical fields with the saved encoders, and producing a churn prediction with class probabilities.

**Dependencies**
numpy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn (for SMOTE), xgboost, and pickle from the standard library.
