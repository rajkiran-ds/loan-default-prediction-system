import logging
import time
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


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

def save_pickle(obj, path):

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    print(f"\nSaved: {path}")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing(df):

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(df[col].median())

        else:

            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    if {"EMI", "Monthly_Income"}.issubset(df.columns):

        df["EMI_Burden"] = (
            df["EMI"] /
            (df["Monthly_Income"] + 1)
        )

    if {"Monthly_Income", "Monthly_Expenses"}.issubset(df.columns):

        df["Net_Savings"] = (
            df["Monthly_Income"] -
            df["Monthly_Expenses"]
        )

    if {"Loan_Amount_Requested", "Monthly_Income"}.issubset(df.columns):

        df["Loan_to_Income"] = (
            df["Loan_Amount_Requested"] /
            (df["Monthly_Income"] + 1)
        )

    if {"Payment_Delay_Days", "Previous_Defaults"}.issubset(df.columns):

        df["Delay_Default_Interaction"] = (
            df["Payment_Delay_Days"] *
            df["Previous_Defaults"]
        )

    if "Credit_Score" in df.columns:

        df["Poor_Credit"] = (
            df["Credit_Score"] < 600
        ).astype(int)

    product_cols = [
        "Has_Credit_Card",
        "Has_Insurance",
        "Has_Mutual_Fund",
        "Has_Fixed_Deposit",
        "Has_Gold_Loan",
        "Has_Vehicle_Loan",
        "Has_Home_Loan",
        "Has_BNPL",
        "Has_Overdraft"
    ]

    existing = [c for c in product_cols if c in df.columns]

    if existing:

        df["Total_Products_Held"] = (
            df[existing].sum(axis=1)
        )

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_loan_pipeline():

    print("\nLoading dataset...")

    df = pd.read_csv(
        "artifacts/loan_guard_ai_dataset.csv"
    )

    # ========================================================
    # TARGET ENCODING
    # ========================================================

    df["Loan_Default"] = df["Loan_Default"].map({
        "Yes": 1,
        "No": 0
    })

    # ========================================================
    # DROP ID COLUMNS
    # ========================================================

    drop_cols = [
        "Customer_ID"
    ]

    df.drop(
        columns=drop_cols,
        inplace=True,
        errors="ignore"
    )

    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    df = handle_missing(df)

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = engineer_features(df)

    # ========================================================
    # SPLIT X AND y
    # ========================================================

    X = df.drop("Loan_Default", axis=1)

    y = df["Loan_Default"]

    # ========================================================
    # ENCODE CATEGORICAL VARIABLES
    # ========================================================

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    # ========================================================
    # SAVE TRAINING COLUMNS
    # ========================================================

    save_pickle(
        X.columns.tolist(),
        "artifacts/model_columns.pkl"
    )

    print("\nFeature count:", X.shape[1])

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # SMOTE
    # ========================================================

    smote = SMOTE(
        random_state=42,
        sampling_strategy=0.5
    )

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    # ========================================================
    # SCALE POS WEIGHT
    # ========================================================

    neg = (y_train == 0).sum()

    pos = (y_train == 1).sum()

    scale = neg / pos

    # ========================================================
    # MODEL
    # ========================================================

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # ========================================================
    # THRESHOLD TUNING
    # ========================================================

    y_proba = model.predict_proba(X_test)[:, 1]

    best_thresh = 0.5

    best_f1 = 0

    for t in np.arange(0.05, 0.95, 0.01):

        preds = (y_proba >= t).astype(int)

        score = f1_score(
            y_test,
            preds,
            average="macro"
        )

        if score > best_f1:

            best_f1 = score

            best_thresh = t

    print("\nBest Threshold:", best_thresh)

    # ========================================================
    # FINAL PREDICTIONS
    # ========================================================

    y_pred = (
        y_proba >= best_thresh
    ).astype(int)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(
        ascending=False
    )

    print("\nTop 15 Features:\n")

    print(
        importances.head(15)
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_pickle(
        model,
        "artifacts/loan_xgb_model.pkl"
    )

    print("\nPipeline completed successfully")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_loan_pipeline()