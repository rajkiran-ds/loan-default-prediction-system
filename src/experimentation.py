import logging
import time
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# ============================================================
# LOGGING SETUP
# ============================================================

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    filename="logs/loan_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.Formatter.converter = time.localtime


# ============================================================
# SAVE MODEL FUNCTION
# ============================================================

def save_model(model, path):
    logging.info("Saving model as pickle file")

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved at: {path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_loan_pipeline():

    logging.info("Pipeline started")

    # 1. Load data
    logging.info("Loading dataset")

    df = pd.read_csv("artifacts/loan_guard_ai_dataset.csv")

    print("\nDataset Loaded")
    print("Shape:", df.shape)

    # 2. Basic preprocessing (matching iris style simplicity)

    logging.info("Handling missing values")

    for col in df.columns:
        
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Target selection
    target = "Has_Personal_Loan"

    logging.info("Splitting features and target")

    X = df.drop(columns=[target])
    y = df[target]

    # 4. Encoding categorical variables

    logging.info("Encoding categorical variables")

    X = pd.get_dummies(X, drop_first=True)

    # 5. Train-test split

    logging.info("Train-test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # 6. Model training

    logging.info("Training Random Forest model")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 7. Predictions

    logging.info("Making predictions")

    y_pred = model.predict(X_test)

    # 8. Evaluation

    logging.info("Classification report")

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================\n")

    print(classification_report(y_test, y_pred))

    # 9. Save model

    model_path = "artifacts/loan_rf_model.pkl"
    save_model(model, model_path)

    logging.info("Pipeline completed successfully")

    return model


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_loan_pipeline()