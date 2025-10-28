import streamlit as st

st.set_page_config(page_title="Healthcare Cost Dashboard", layout="wide")
st.title("Predictive Healthcare Cost Dashboard")

st.write("Hello! Your app is running. Next, we’ll load model.joblib exported from Colab.")



# ======================================
# STEP 2 — STREAMLIT DASHBOARD + MODEL
# ======================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import date

st.set_page_config(page_title="Healthcare Cost Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_claims_data.csv", parse_dates=["servicedate"], infer_datetime_format=True)
    df['claim_month'] = pd.to_datetime(df['servicedate']).dt.to_period('M').astype(str)
    df['claim_week'] = pd.to_datetime(df['servicedate']).dt.to_period('W').astype(str)
    df['claim_day'] = pd.to_datetime(df['servicedate']).dt.date
    df['year_quarter'] = pd.to_datetime(df['servicedate']).dt.to_period('Q').astype(str)
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Daily View",
    "Weekly View",
    "Monthly Overview",
    "Quarterly Reports",
    "Predictive Modeling"
])

# ======================================================
# 1️⃣ DAILY DASHBOARD — Claims Filed Today, Fraud Flags
# ======================================================
if page == "Daily View":
    st.header("📅 Daily View — Claims Activity & Fraud Detection")
    today = df['claim_day'].max()
    df_today = df[df['claim_day'] == today]

    st.metric("Claims Filed Today", len(df_today))
    st.metric("Total Claim Cost (Today)", f"${df_today['claim_cost'].sum():,.0f}")

    # Fraud Detection — simple outlier based
    threshold = df['claim_cost'].quantile(0.99)
    frauds = df_today[df_today['claim_cost'] > threshold]
    st.metric("Potential Fraudulent Claims", len(frauds))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Cost Conditions Today")
        top_cond = df_today.groupby('diagnosis1')['claim_cost'].sum().nlargest(5)
        st.bar_chart(top_cond)
    with col2:
        st.subheader("Payer Coverage Summary")
        payer_summary = df_today.groupby('payer_name')['claim_cost'].sum().nlargest(5)
        st.bar_chart(payer_summary)

# ======================================================
# 2️⃣ WEEKLY DASHBOARD — Trends & Provider Performance
# ======================================================
elif page == "Weekly View":
    st.header("📆 Weekly Performance Dashboard")

    weekly = df.groupby('claim_week', as_index=False).agg(
        total_claims=('id','count'),
        total_cost=('claim_cost','sum'),
        avg_cost=('claim_cost','mean')
    )
    st.line_chart(weekly.set_index('claim_week')[['total_cost','avg_cost']])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Providers by Claim Volume")
        top_prov = df.groupby('providerid')['id'].count().nlargest(5)
        st.bar_chart(top_prov)
    with col2:
        st.subheader("Top 5 Payers by Volume")
        top_pay = df.groupby('payer_name')['id'].count().nlargest(5)
        st.bar_chart(top_pay)

# ======================================================
# 3️⃣ MONTHLY STRATEGY DASHBOARD — Budget vs Spend
# ======================================================
elif page == "Monthly Overview":
    st.header("📈 Monthly Strategy & Finance Overview")

    monthly = df.groupby('claim_month', as_index=False).agg(
        total_claims=('id','count'),
        total_cost=('claim_cost','sum'),
        avg_cost=('claim_cost','mean')
    )
    st.line_chart(monthly.set_index('claim_month')[['total_cost','avg_cost']])

    st.subheader("Top Conditions by Monthly Spend")
    cond_month = df.groupby(['claim_month','diagnosis1'])['claim_cost'].sum().reset_index()
    fig = px.bar(cond_month, x='claim_month', y='claim_cost', color='diagnosis1', title="Condition-Level Monthly Cost")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 4️⃣ QUARTERLY DASHBOARD — Long-term Cost & Risk
# ======================================================
elif page == "Quarterly Reports":
    st.header("📊 Quarterly Cost & Risk Analysis")

    quarter = df.groupby('year_quarter', as_index=False).agg(
        total_claims=('id','count'),
        total_cost=('claim_cost','sum'),
        avg_cost=('claim_cost','mean')
    )
    st.area_chart(quarter.set_index('year_quarter')[['total_cost','avg_cost']])

    st.subheader("Top 10 High-Risk Conditions (Total Cost)")
    high_risk = df.groupby('diagnosis1')['claim_cost'].sum().nlargest(10)
    st.bar_chart(high_risk)

# ======================================================
# 5️⃣ PREDICTION — Random Forest Model
# ======================================================
elif page == "Predictive Modeling":
    st.header("🧠 Predict Future Claim Cost")

    # Load model
    try:
        model = joblib.load("model.joblib")
        st.success("Model loaded successfully!")
    except:
        st.error("Upload model.joblib trained in Colab.")
        st.stop()

    # Input form
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["M","F"])
        race = st.text_input("Race", "Unknown")
    with col2:
        total_enc = st.number_input("Total Encounters", 0, 100, 5)
        total_proc = st.number_input("Total Procedures", 0, 50, 3)
    with col3:
        total_meds = st.number_input("Total Medications", 0, 50, 4)
        dialysis_sessions = st.number_input("Dialysis Sessions", 0, 30, 2)

    row = pd.DataFrame([{
        "gender": gender, "race": race, "ethnicity": "Unknown",
        "total_encounters": total_enc, "inpatient_encounters": 0,
        "outpatient_encounters": 0, "emergency_encounters": 0,
        "total_procedures": total_proc, "dialysis_sessions": dialysis_sessions,
        "total_meds": total_meds, "unique_medications": total_meds
    }])
    if st.button("Predict Cost"):
        pred = model.predict(row)[0]
        st.metric("Predicted Claim Cost", f"${pred:,.2f}")
