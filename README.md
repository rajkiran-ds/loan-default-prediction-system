# 🏦 Loan Default Prediction System


A machine learning system that predicts whether a loan applicant is likely to default — helping financial institutions reduce risk and automate loan approval decisions.

🚀 **Live Demo:** [Launch Streamlit App](https://rajkiran-ds-loan-default-prediction-system-app-vchisg.streamlit.app/)

---

## Problem Statement

XYZ Financial Firm offers Business, Personal, and Educational loans. Their current approval process is entirely manual, making it slow and prone to missed defaulters. This system replaces that with an ML pipeline that flags high-risk applicants automatically.

---

## Business Impact

| Goal | Approach |
|------|----------|
| Reduce financial risk | Predict defaulters before approval |
| Handle class imbalance | SMOTE oversampling |
| Improve minority class detection | Macro F1-score optimization |
| Automate risk analysis | End-to-end prediction pipeline |
| Production-ready deployment | Streamlit + Docker |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| ML Models | XGBoost, Scikit-learn |
| Imbalance Handling | SMOTE |
| App Framework | Streamlit |
| Containerization | Docker |
| Version Control | Git & GitHub |

---

## ML Pipeline

```
Data Loading → Missing Value Handling → Feature Engineering
→ Encoding → Train-Test Split → SMOTE Oversampling
→ XGBoost Training → Threshold Optimization → Prediction Pipeline
→ Streamlit Deployment → Docker Containerization
```

---

## Feature Engineering

Six custom features were engineered to improve prediction performance:

| Feature | Description |
|---------|-------------|
| EMI Burden | Monthly EMI as a fraction of income |
| Net Savings | Income minus total loan obligations |
| Loan-to-Income Ratio | Total loan amount relative to annual income |
| Delay-Default Interaction | Combined signal of payment delays and past defaults |
| Poor Credit Indicator | Binary flag for poor credit history |
| Total Products Held | Number of financial products held by applicant |

---

## Model Highlights

- **XGBoost Classifier** used for final prediction
- **Random Forest** used during experimentation
- **SMOTE** applied to handle class imbalance
- **Threshold tuning** performed for optimal recall/precision trade-off
- **Macro F1-score** used as primary evaluation metric

---

## Project Structure

```
Loan_Default_Prediction/
│
├── artifacts/              # Saved models and encoders
├── images/                 # Screenshots and visuals
├── logs/                   # Training and experimentation logs
├── src/
│   ├── __init__.py
│   ├── pipeline.py         # Training pipeline
│   └── experimentation.py  # Model experimentation
│
├── app.py                  # Streamlit application
├── main.py                 # Pipeline entry point
├── Dockerfile
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
python main.py
```

### 3. Launch Streamlit App

```bash
streamlit run app.py
```

---

## Docker Support

### Build Image

```bash
docker build -t loan-default-app .
```

### Run Container

```bash
docker run loan-default-app
```

---

## Future Improvements

- [ ] Model monitoring & drift detection
- [ ] CI/CD pipeline integration
- [ ] Cloud deployment (AWS/Azure)
- [ ] Advanced feature engineering
- [ ] Explainable AI (SHAP values)

---

## Author

**Raj Kiran Reddy**  
B.Tech Data Science | MLRITM 
📍 Hyderabad, Telangana, India
