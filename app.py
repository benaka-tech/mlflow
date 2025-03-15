# Install required packages (run this in your terminal or environment)
# pip install prefect mlflow scikit-learn pandas matplotlib seaborn flask

# ------------------------------ #
#         Import Libraries       #
# ------------------------------ #
import os
import datetime
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from prefect import task, flow  # Prefect 2 API
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify

# Global dictionary to store pipeline details for API access (Sub-Objective 3)
pipeline_details = {}

# ------------------------------ #
#  Sub-Objective 1: Data Pipeline
#    1.1 Business Understanding,
#        - Business Problem: Credit Card Fraud Detection.
#          This project addresses the high-impact issue of detecting fraudulent credit card transactions,
#          which is critical for minimizing financial losses and protecting consumers.
#    1.2 Data Ingestion,
#    1.3 Data Pre-processing,
#    1.4 Exploratory Data Analysis (EDA),
#    1.5 DataOps 
# ------------------------------ #

@task
def load_data():
    """
    [Sub-Objective 1.2: Data Ingestion]
    Loads the Credit Card Fraud Detection dataset.
    Expects a file named 'creditcard.csv' to be in the working directory.
    """
    file_path = 'creditcard_2023.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError("File 'creditcard.csv' not found. Please ensure it exists in the working directory.")
    df = pd.read_csv(file_path)
    print("Data loaded successfully with shape:", df.shape)
    return df

@task
def display_statistics(df):
    """
    [Sub-Objective 1.3: Data Pre-processing]
    Displays summary statistics and data types of the dataset to help understand its structure.
    """
    print("----- Data Info -----")
    print(df.info())
    print("\n----- Summary Statistics -----")
    print(df.describe())
    return True

@task
def preprocess_data(df):
    """
    [Sub-Objective 1.3: Data Pre-processing]
    Preprocesses the dataset by:
      - Checking for missing numeric values and imputing them with the median.
      - Converting the target ('Class') to integer.
      - Normalizing the 'Time' and 'Amount' columns if present.
    """
    # Check for missing values and impute numeric columns if needed
    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        print("Missing values imputed for numeric columns.")
    else:
        print("No missing values found.")
    
    # Ensure target is integer
    df['Class'] = df['Class'].astype(int)
    
    # Normalize "Time" and "Amount" if they exist
    scaler = StandardScaler()
    cols_to_normalize = []
    if 'Time' in df.columns:
        cols_to_normalize.append('Time')
    if 'Amount' in df.columns:
        cols_to_normalize.append('Amount')
    
    if cols_to_normalize:
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        print(f"Normalized columns: {cols_to_normalize}")
    else:
        print("No columns to normalize.")
    
    print("Preprocessing completed.")
    return df
    """
    [Sub-Objective 1.3: Data Pre-processing]
    Preprocesses the dataset by:
      - Checking for missing numeric values and imputing them with the median.
      - Converting the target ('Class') to integer.
      - Normalizing the 'Time' and 'Amount' columns using StandardScaler.
    """
    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        print("Missing values imputed for numeric columns.")
    else:
        print("No missing values found.")
    
    df['Class'] = df['Class'].astype(int)
    
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    print("Preprocessing completed.")
    return df

@task
def perform_eda(df):
    """
    [Sub-Objective 1.4: Exploratory Data Analysis (EDA)]
    Performs EDA by:
      - Plotting and saving a correlation heatmap.
      - Plotting and saving a histogram for the 'Amount' feature.
    """
    # Create correlation heatmap
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    heatmap_file = "correlation_heatmap.png"
    plt.savefig(heatmap_file)
    plt.close()
    print("Correlation heatmap saved as:", heatmap_file)
    
    # Create histogram for 'Amount'
    plt.figure(figsize=(8,6))
    sns.histplot(df['Amount'], bins=30, kde=True)
    plt.title("Distribution of Transaction Amounts")
    amount_hist_file = "amount_histogram.png"
    plt.savefig(amount_hist_file)
    plt.close()
    print("Amount histogram saved as:", amount_hist_file)
    
    return heatmap_file, amount_hist_file

@task
def train_models(df):
    """
    [Sub-Objective 2: Machine Learning Pipeline]
    Splits the data into training and testing sets (70/30), trains two models (RandomForestClassifier and LogisticRegression),
    evaluates them using multiple metrics, logs parameters and metrics to MLflow, and generates a feature importance plot.
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    mlflow.set_experiment("CreditCard_Fraud_Detection_Experiment")
    model_results = {}
    
    # RandomForest Model
    with mlflow.start_run(run_name="RandomForest") as run_rf:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        metrics_rf = {
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "precision": precision_score(y_test, y_pred_rf),
            "recall": recall_score(y_test, y_pred_rf),
            "f1_score": f1_score(y_test, y_pred_rf),
            "auc": roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
        }
        mlflow.log_param("model_rf", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        for key, value in metrics_rf.items():
            mlflow.log_metric(key, value)
        
        feature_importances = rf_model.feature_importances_
        features = X.columns
        plt.figure(figsize=(10,6))
        sns.barplot(x=feature_importances, y=features)
        plt.title("Feature Importances (RandomForest)")
        fi_file = "feature_importance.png"
        plt.savefig(fi_file)
        plt.close()
        mlflow.log_artifact(fi_file)
        
        model_results["RandomForest"] = {
            "run_id": run_rf.info.run_id,
            **metrics_rf,
            "feature_importance_plot": fi_file
        }
        print("RandomForest training completed. MLflow Run ID:", run_rf.info.run_id)
    
    # Logistic Regression Model
    with mlflow.start_run(run_name="LogisticRegression") as run_lr:
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        
        metrics_lr = {
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "precision": precision_score(y_test, y_pred_lr),
            "recall": recall_score(y_test, y_pred_lr),
            "f1_score": f1_score(y_test, y_pred_lr),
            "auc": roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1])
        }
        mlflow.log_param("model_lr", "LogisticRegression")
        for key, value in metrics_lr.items():
            mlflow.log_metric(key, value)
        
        model_results["LogisticRegression"] = {
            "run_id": run_lr.info.run_id,
            **metrics_lr
        }
        print("LogisticRegression training completed. MLflow Run ID:", run_lr.info.run_id)
    
    # Update global pipeline details for API access (Sub-Objective 3)
    pipeline_details.update({
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'models': model_results
    })
    return model_results

# ------------------------------ #
#         Prefect Flow           #
# ------------------------------ #

# [Sub-Objective 1.5: DataOps]
# In Prefect 2, scheduling is typically managed via deployments.
# Here we define a flow function for local execution.
@flow(name="CreditCard_Fraud_Detection_Pipeline")
def pipeline_flow():
    df = load_data()
    _ = display_statistics(df)
    df_clean = preprocess_data(df)
    eda_files = perform_eda(df_clean)
    model_results = train_models(df_clean)
    return model_results

if __name__ == "__main__":
    # Execute the flow
    pipeline_results = pipeline_flow()
    
    # ------------------------------ #
    #      API Access with Flask     #
    #  Sub-Objective 3: API Access
    # ------------------------------ #
    app = Flask(__name__)
    
    @app.route("/pipeline_details", methods=["GET"])
    def get_pipeline_details():
        """
        API endpoint to retrieve key pipeline details such as:
          - MLflow run IDs for each model,
          - Model performance metrics (accuracy, precision, recall, F1 score, AUC),
          - Feature importance plot file,
          - Timestamp of the pipeline run.
        """
        return jsonify(pipeline_details)
    
    # Run the Flask API server
    app.run(host="0.0.0.0", port=5000)
