# Loan Default Prediction System

## Problem Statement

XYZ Financial Firm provides different types of loans such as:

* Business Loans
* Personal Loans
* Educational Loans

Currently, the loan approval process relies heavily on manual verification and the company struggles to identify potential loan defaulters accurately.

This project builds a Machine Learning based Loan Default Prediction System that predicts whether a customer is likely to default on a loan.


## Business Objective

* Reduce financial risk
* Improve loan approval process
* Identify high-risk applicants
* Automate loan risk analysis


## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SMOTE
* Streamlit
* Docker


## Machine Learning Workflow

1. Data Loading
2. Missing Value Handling
3. Feature Engineering
4. Encoding
5. Train-Test Split
6. SMOTE Oversampling
7. XGBoost Training
8. Threshold Optimization
9. Prediction
10. Streamlit Deployment


## Feature Engineering

Additional features created:

* EMI Burden
* Net Savings
* Loan-to-Income Ratio
* Delay-Default Interaction
* Poor Credit Indicator
* Total Products Held


## Model Highlights

* Imbalanced dataset handled using SMOTE
* XGBoost Classifier used
* Threshold tuning performed
* Macro F1-score optimization used


## Run Project

Install requirements:

pip install -r requirements.txt

Run pipeline:

python main.py

Run Streamlit App:

streamlit run app.py


## Docker Support

Build Docker Image:

docker build -t loan-default-app .

Run Docker Container:

docker run -p 8501:8501 loan-default-app
