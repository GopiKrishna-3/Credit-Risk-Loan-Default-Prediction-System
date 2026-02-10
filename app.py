import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load ML Model and Columns
# -------------------------------
rf = joblib.load("loan_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -------------------------------
# App Title
# -------------------------------
st.title("🏦 Bank-Style Loan Approval Prediction System")

# -------------------------------
# Customer Details
# -------------------------------
st.subheader("Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# -------------------------------
# Financial Details
# -------------------------------
st.subheader("Financial Details")
app_income = st.number_input("Applicant Monthly Income", min_value=0)
coapp_income = st.number_input("Coapplicant Monthly Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Term (Months)", [60, 120, 180, 240, 360])
existing_emis = st.number_input("Existing EMI Payments", min_value=0)

# -------------------------------
# Credit Score Input
# -------------------------------
st.subheader("Credit Behaviour")
credit_score = st.number_input("Enter your Credit Score (300-900)", min_value=300, max_value=900)

# -------------------------------
# Property Area
# -------------------------------
st.subheader("Property Area")
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("Predict Loan Approval"):

    # -------------------------------
    # Map Credit Score to Good/Bad
    # -------------------------------
    if credit_score >= 700:
        credit_history_enc = 1  # Good
    else:
        credit_history_enc = 0  # Bad

    # -------------------------------
    # Feature Calculations
    # -------------------------------
    total_income = app_income + coapp_income
    monthly_loan_payment = loan_amount / loan_term if loan_term > 0 else 0
    emi_income_ratio = (existing_emis + monthly_loan_payment) / total_income if total_income > 0 else 0
    loan_to_income_ratio = loan_amount / total_income if total_income > 0 else 0

    # -------------------------------
    # Bank-Style Rule Checks
    # -------------------------------
    reasons = []
    approve_flag = True

    # Rule 1: Unrealistic loan
    if loan_amount > total_income * 50:
        reasons.append("Requested loan is unrealistically high compared to your income")
        approve_flag = False

    # Rule 2: EMI too high
    if emi_income_ratio > 0.5:
        reasons.append("Total EMIs exceed 50% of your income")
        approve_flag = False

    # Rule 3: Bad credit
    if credit_history_enc == 0:
        if loan_to_income_ratio > 15:
            reasons.append("Loan amount too high for your income with bad credit")
            approve_flag = False
        elif emi_income_ratio > 0.3:
            reasons.append("Monthly payment burden too high with bad credit")
            approve_flag = False

    # -------------------------------
    # ML Prediction if basic rules pass
    # -------------------------------
    if approve_flag:
        # Encode categorical features
        gender_enc = 1 if gender == "Male" else 0
        married_enc = 1 if married == "Yes" else 0
        education_enc = 1 if education == "Graduate" else 0
        self_employed_enc = 1 if self_employed == "Yes" else 0
        prop_semi = 1 if property_area == "Semiurban" else 0
        prop_urban = 1 if property_area == "Urban" else 0
        prop_new = 0

        # Prepare input for ML model
        input_data = pd.DataFrame([[
            gender_enc, married_enc, dependents, education_enc, self_employed_enc,
            app_income, coapp_income, loan_amount, loan_term,
            credit_history_enc, 30, existing_emis,
            prop_semi, prop_urban, prop_new,
            total_income, loan_to_income_ratio, emi_income_ratio
        ]], columns=[
            'Gender','Married','Dependents','Education','Self_Employed',
            'ApplicantIncome','CoapplicantIncome','LoanAmount',
            'Loan_Amount_Term','Credit_History','Age','Existing EMIs',
            'Property_Area_Semiurban','Property_Area_Urban','Property_Area_New',
            'Total_Income','DTI','EMI_Income_Ratio'
        ])

        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        prediction = rf.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
            st.info("Approved based on your financial profile and credit score")
        else:
            st.error("❌ Loan Rejected")
            st.warning("Rejected based on model prediction")
    else:
        st.error("❌ Loan Rejected")
        for r in reasons:
            st.warning(f"⚠ {r}")

