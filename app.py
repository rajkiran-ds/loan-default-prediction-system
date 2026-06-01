import streamlit as st
import pandas as pd
import pickle


# ============================================================
# LOAD MODEL
# ============================================================

model = pickle.load(
    open("artifacts/loan_xgb_model.pkl", "rb")
)

columns = pickle.load(
    open("artifacts/model_columns.pkl", "rb")
)


# ============================================================
# TITLE
# ============================================================

st.title("Loan Default Prediction App")

st.write("Enter customer details below")


# ============================================================
# USER INPUTS
# ============================================================

age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0
)

monthly_expenses = st.number_input(
    "Monthly Expenses",
    min_value=0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

emi = st.number_input(
    "EMI",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount Requested",
    min_value=0
)

delay_days = st.number_input(
    "Payment Delay Days",
    min_value=0
)

previous_defaults = st.number_input(
    "Previous Defaults",
    min_value=0
)

has_insurance = st.selectbox(
    "Has Insurance",
    [0, 1]
)

has_credit_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

has_bnpl = st.selectbox(
    "Has BNPL",
    [0, 1]
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_data = {

    "Age": age,

    "Gender": gender,

    "Marital_Status": marital_status,

    "Monthly_Income": monthly_income,

    "Monthly_Expenses": monthly_expenses,

    "Credit_Score": credit_score,

    "EMI": emi,

    "Loan_Amount_Requested": loan_amount,

    "Payment_Delay_Days": delay_days,

    "Previous_Defaults": previous_defaults,

    "Has_Insurance": has_insurance,

    "Has_Credit_Card": has_credit_card,

    "Has_BNPL": has_bnpl
}

input_df = pd.DataFrame([input_data])


# ============================================================
# FEATURE ENGINEERING
# ============================================================

input_df["EMI_Burden"] = (
    input_df["EMI"] /
    (input_df["Monthly_Income"] + 1)
)

input_df["Net_Savings"] = (
    input_df["Monthly_Income"] -
    input_df["Monthly_Expenses"]
)

input_df["Loan_to_Income"] = (
    input_df["Loan_Amount_Requested"] /
    (input_df["Monthly_Income"] + 1)
)

input_df["Delay_Default_Interaction"] = (
    input_df["Payment_Delay_Days"] *
    input_df["Previous_Defaults"]
)

input_df["Poor_Credit"] = (
    input_df["Credit_Score"] < 600
).astype(int)

input_df["Total_Products_Held"] = (
    input_df["Has_Insurance"] +
    input_df["Has_Credit_Card"] +
    input_df["Has_BNPL"]
)


# ============================================================
# DUMMY ENCODING
# ============================================================

input_df = pd.get_dummies(
    input_df,
    drop_first=True
)


# ============================================================
# MATCH TRAINING COLUMNS
# ============================================================

input_df = input_df.reindex(
    columns=columns,
    fill_value=0
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Loan Default Risk"):

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:

        st.error(
            f"High Risk of Default\n\n"
            f"Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"Low Risk of Default\n\n"
            f"Probability: {probability:.2%}"
        )

