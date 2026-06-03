# ============================================================
# LOAN DEFAULT PREDICTION - EXPERIMENTATION PIPELINE
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):
    df = pd.read_csv(path)
    print(f"\n[DATA LOADED] Shape: {df.shape}")
    return df


# ============================================================
# 2. MISSING VALUE HANDLING
# ============================================================

def handle_missing(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


# ============================================================
# 3. WINSORIZATION
# ============================================================

def winsorize(df, num_cols):
    df = df.copy()
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        lower = q1 - 1.5 * (q3 - q1)
        upper = q3 + 1.5 * (q3 - q1)
        df[col] = np.clip(df[col], lower, upper)
    return df


# ============================================================
# 4. LOG TRANSFORMATION
# ============================================================

def log_transform(df, num_cols):
    df = df.copy()
    for col in num_cols:
        if (df[col] >= 0).all():
            df[col] = np.log1p(df[col])
    return df


# ============================================================
# 5. ENCODE — Label Encode to avoid column explosion
# ============================================================

def encode(df, target):
    df = df.copy()
    X = df.drop(columns=[target])
    y = df[target].map({"No": 0, "Yes": 1})

    for col in X.select_dtypes(include=["object", "category", "string"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.astype(float)
    return X, y


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ============================================================
# 7. FEATURE SELECTION
# ============================================================

def feature_selection(X, y):
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    selected_cols = X.columns[model.feature_importances_ >= np.mean(model.feature_importances_)]
    print(f"[FEATURE SELECTION] {len(selected_cols)} / {X.shape[1]} features selected")
    return X[selected_cols]


# ============================================================
# 8. SMOTE
# ============================================================

def smote_balance(X_train, y_train):
    sm = SMOTE(sampling_strategy=0.67, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ============================================================
# 9. MODELS
# ============================================================

def get_models():
    return {
        "KNN":                 KNeighborsClassifier(n_jobs=-1),
        "Naive Bayes":         GaussianNB(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost":            AdaBoostClassifier(random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42)
    }


# ============================================================
# 10. EVALUATE
# ============================================================

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "Accuracy":  round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, pred, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)
                     if hasattr(model, "predict_proba") else None
    }


# ============================================================
# 11. RUN EXPERIMENT
# ============================================================

def run_experiment(label, X_train, X_test, y_train, y_test):
    print(f"\n{'='*50}")
    print(f"  EXPERIMENT: {label}")
    print(f"{'='*50}")

    results = []
    for name, model in get_models().items():
        print(f"  Training {name}...")
        res = evaluate(model, X_train, X_test, y_train, y_test)
        res["Model"] = name
        results.append(res)

    df_res = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
    df_res = df_res.sort_values("F1", ascending=False).reset_index(drop=True)
    print(f"\n{df_res.to_string(index=False)}")
    return df_res


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def run_pipeline(path):

    target = "Loan_Default"

    df = load_data(path)
    df = handle_missing(df)

    num_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
        and col != target
    ]

    df_win = winsorize(df, num_cols)
    df_log = log_transform(df, num_cols)

    # EXPERIMENT 1: BASE
    X, y = encode(df, target)
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"[COLUMNS] {X.shape[1]}")
    base_df = run_experiment("BASE DATA", X_train, X_test, y_train, y_test)

    # EXPERIMENT 2: WINSORIZED
    X, y = encode(df_win, target)
    X_train, X_test, y_train, y_test = split(X, y)
    win_df = run_experiment("WINSORIZED DATA", X_train, X_test, y_train, y_test)

    # EXPERIMENT 3: LOG TRANSFORM
    X, y = encode(df_log, target)
    X_train, X_test, y_train, y_test = split(X, y)
    log_df = run_experiment("LOG TRANSFORMED DATA", X_train, X_test, y_train, y_test)

    # EXPERIMENT 4: FEATURE SELECTION + SMOTE
    X, y = encode(df, target)
    X = feature_selection(X, y)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, y_train = smote_balance(X_train, y_train)
    smote_df = run_experiment("FEATURE SELECTION + SMOTE", X_train, X_test, y_train, y_test)

    # FINAL SUMMARY
    print(f"\n{'='*50}")
    print("  FINAL SUMMARY — BEST F1 PER EXPERIMENT")
    print(f"{'='*50}")
    summary = pd.DataFrame({
        "Experiment": ["Base Data", "Winsorized", "Log Transform", "Feature + SMOTE"],
        "Best Model": [d.iloc[0]["Model"] for d in [base_df, win_df, log_df, smote_df]],
        "Best F1":    [d.iloc[0]["F1"]    for d in [base_df, win_df, log_df, smote_df]]
    })
    print(f"\n{summary.to_string(index=False)}")

    return base_df, win_df, log_df, smote_df


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    run_pipeline("artifacts/loan_guard_ai_dataset.csv")