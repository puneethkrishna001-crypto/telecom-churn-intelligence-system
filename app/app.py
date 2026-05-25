import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# =========================
# LOAD MODEL FILES
# =========================

# Get absolute paths for deployment compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'xgb_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'scaler.pkl')
FEATURE_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'feature_names.pkl')

model = pickle.load(
    open(MODEL_PATH, 'rb')
)

scaler = pickle.load(
    open(SCALER_PATH, 'rb')
)

features = pickle.load(
    open(FEATURE_PATH, 'rb')
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: white;
}

.stButton > button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #ff2b2b;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📊 About Project")

st.sidebar.info(
    """
    This application predicts telecom customer churn risk
    using Machine Learning models.

    ### Models Used
    - Logistic Regression
    - Random Forest
    - XGBoost

    ### Features
    - Churn Probability Prediction
    - Risk Classification
    - Business Recommendations
    - Real-Time ML Inference
    """
)

st.sidebar.success(
    "Built using Streamlit, Scikit-learn, XGBoost and SHAP"
)

# =========================
# TITLE
# =========================

st.title("📉 Customer Churn Risk Predictor")

st.markdown(
    """
    Predict whether a telecom customer is likely to churn
    using Machine Learning.
    """
)

st.markdown("---")

# =========================
# INPUT SECTION
# =========================

col1, col2 = st.columns(2)

with col1:

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    monthly = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        65.0
    )

    total = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        800.0
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    dependents = st.selectbox(
        "Dependents",
        [0, 1]
    )

with col2:

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

# =========================
# PREDICTION BUTTON
# =========================

if st.button("🔍 Predict Churn Risk"):

    # Create empty dataframe
    input_data = pd.DataFrame(
        np.zeros((1, len(features))),
        columns=features
    )

    # =========================
    # NUMERICAL FEATURES
    # =========================

    if 'tenure' in features:
        input_data['tenure'] = tenure

    if 'MonthlyCharges' in features:
        input_data['MonthlyCharges'] = monthly

    if 'TotalCharges' in features:
        input_data['TotalCharges'] = total

    if 'SeniorCitizen' in features:
        input_data['SeniorCitizen'] = senior

    if 'Dependents' in features:
        input_data['Dependents'] = dependents

    # =========================
    # ENGINEERED FEATURES
    # =========================

    if 'avg_monthly_spend' in features:
        input_data['avg_monthly_spend'] = (
            total / (tenure + 1)
        )

    if 'is_high_value' in features:
        input_data['is_high_value'] = int(
            monthly > 70
        )

    # =========================
    # CONTRACT FEATURES
    # =========================

    if contract == "One year":
        if 'Contract_One year' in features:
            input_data['Contract_One year'] = 1

    elif contract == "Two year":
        if 'Contract_Two year' in features:
            input_data['Contract_Two year'] = 1

    # =========================
    # INTERNET FEATURES
    # =========================

    if internet == "Fiber optic":
        if 'InternetService_Fiber optic' in features:
            input_data['InternetService_Fiber optic'] = 1

    elif internet == "No":
        if 'InternetService_No' in features:
            input_data['InternetService_No'] = 1

    # =========================
    # PAYMENT METHOD FEATURES
    # =========================

    if payment == "Electronic check":
        if 'PaymentMethod_Electronic check' in features:
            input_data['PaymentMethod_Electronic check'] = 1

    elif payment == "Mailed check":
        if 'PaymentMethod_Mailed check' in features:
            input_data['PaymentMethod_Mailed check'] = 1

    elif payment == "Credit card (automatic)":
        if 'PaymentMethod_Credit card (automatic)' in features:
            input_data['PaymentMethod_Credit card (automatic)'] = 1

    # =========================
    # PAPERLESS BILLING
    # =========================

    if paperless == "Yes":
        if 'PaperlessBilling' in features:
            input_data['PaperlessBilling'] = 1

    # =========================
    # SCALE INPUT
    # =========================

    scaled_input = scaler.transform(input_data)

    # =========================
    # PREDICTION
    # =========================

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(
        scaled_input
    )[0][1]

    st.markdown("---")

    # =========================
    # RESULT SECTION
    # =========================

    st.subheader("Prediction Result")

    if probability > 0.7:

        st.error(
            f"⚠️ High Churn Risk: {probability*100:.2f}%"
        )

        st.markdown("""
        ### Recommended Business Actions
        - Offer loyalty discounts
        - Provide premium customer support
        - Recommend long-term contract
        - Contact customer retention team
        """)

    elif probability > 0.4:

        st.warning(
            f"🟡 Medium Churn Risk: {probability*100:.2f}%"
        )

        st.markdown("""
        ### Recommended Business Actions
        - Send promotional offers
        - Improve engagement
        - Monitor customer satisfaction
        """)

    else:

        st.success(
            f"✅ Low Churn Risk: {probability*100:.2f}%"
        )

        st.markdown("""
        ### Customer Status
        - Customer likely to stay
        - Low retention risk
        - Stable customer profile
        """)

    # =========================
    # PROGRESS BAR
    # =========================

    st.progress(int(probability * 100))

    # =========================
    # METRICS
    # =========================

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

    with col4:
        st.metric(
            "Monthly Charges",
            f"${monthly:.2f}"
        )

    with col5:
        st.metric(
            "Tenure",
            f"{tenure} Months"
        )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Customer Churn Prediction System | Built with Streamlit & Machine Learning"
)
