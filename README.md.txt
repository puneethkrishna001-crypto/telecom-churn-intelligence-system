# 📉 Customer Churn Prediction System

## 📌 Project Overview

This project is an end-to-end Machine Learning application designed to predict telecom customer churn using customer behavior and service usage data. The system helps identify high-risk customers and provides actionable business insights for customer retention strategies.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- SMOTE balancing
- Machine Learning modeling
- SHAP Explainable AI
- Power BI Dashboard
- Streamlit Web Application

---

# 🚀 Features

✅ Exploratory Data Analysis (EDA)  
✅ Data Cleaning & Preprocessing  
✅ Feature Engineering  
✅ SMOTE Class Balancing  
✅ Multiple ML Model Comparison  
✅ Hyperparameter Tuning  
✅ SHAP Explainability  
✅ Interactive Power BI Dashboard  
✅ Streamlit Web Application  
✅ Churn Risk Prediction System  

---

# 🧠 Machine Learning Models Used

| Model | ROC-AUC Score |
|---|---|
| Logistic Regression | 0.8373 |
| Random Forest | 0.8168 |
| XGBoost | 0.8149 |
| Tuned XGBoost | 0.8218 |

### ✅ Best Model
Logistic Regression achieved the best generalization performance with ROC-AUC score of **0.8373**.

---

# 📊 Key Business Insights

- Customers with **low tenure** are more likely to churn.
- **Fiber optic users** showed higher churn behavior.
- **Electronic check payment method** customers had increased churn probability.
- Long-term contract customers were significantly more stable.
- The system identified **231 high-risk customers** for proactive retention campaigns.

---

# 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to identify the most influential churn-driving features.

### Top Churn Drivers
- Number of services used
- Internet service type
- Tenure
- Payment method
- Contract type
- Monthly charges

---

# 🛠️ Technologies Used

## Programming & ML
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Imbalanced-learn

## Visualization & Dashboard
- Matplotlib
- Seaborn
- Power BI

## Deployment
- Streamlit

---

# 📂 Project Structure

```bash
customer-churn-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── telco_churn.csv
│   ├── predictions.csv
│   └── predictions_dashboard.csv
│
├── models/
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── xgb_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── dashboard/
│
├── screenshots/
│
└── README.md