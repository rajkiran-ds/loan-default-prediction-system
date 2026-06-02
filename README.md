# Loan Default Prediction System

## Live Demo

🚀 Streamlit Application:
https://rajkiran-ds-loan-default-prediction-system-app-vchisg.streamlit.app/

## Problem Statement

XYZ Financial Firm provides multiple loan services such as:

* Business Loans
* Personal Loans
* Educational Loans

Currently, the loan approval process relies heavily on manual verification and the company struggles to accurately identify potential loan defaulters.

This project builds a Machine Learning based Loan Default Prediction System that predicts whether a customer is likely to default on a loan.


## Business Objective

* Reduce financial risk
* Improve loan approval efficiency
* Identify high-risk applicants
* Automate loan risk analysis
* Improve minority class prediction performance


## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Random Forest
* SMOTE
* Streamlit
* Docker
* Git & GitHub


## Machine Learning Workflow

1. Data Loading
2. Missing Value Handling
3. Feature Engineering
4. Encoding Categorical Variables
5. Train-Test Split
6. SMOTE Oversampling
7. XGBoost Model Training
8. Threshold Optimization
9. Prediction Pipeline
10. Streamlit Deployment
11. Docker Containerization

## Feature Engineering

Additional features engineered for better model performance:

* EMI Burden
* Net Savings
* Loan-to-Income Ratio
* Delay-Default Interaction
* Poor Credit Indicator
* Total Products Held

## Model Highlights

* Imbalanced dataset handled using SMOTE
* XGBoost Classifier used for final prediction
* Random Forest used for experimentation
* Threshold tuning performed
* Macro F1-score optimization used
* Feature engineering improved prediction capability

## Project Structure

Loan_Default_Prediction/
│
├── artifacts/
├── images/
├── logs/
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   └── experimentation.py
│
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore


## Run Project

### Install Requirements

pip install -r requirements.txt

### Run Training Pipeline

python main.py

### Run Streamlit Application

streamlit run app.py


## Docker Support

### Build Docker Image


docker build -t loan-default-app .

### Run Docker Container

docker run loan-default-app


## Application Screenshots

### Streamlit Homepage


### Prediction Result


### Docker Deployment


## Future Improvements

* Model monitoring
* CI/CD integration
* Cloud deployment
* Advanced feature engineering
* Explainable AI integration


## Author

Raj Kiran Reddy
