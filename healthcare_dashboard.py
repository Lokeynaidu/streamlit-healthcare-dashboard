import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Analytics Dashboard for Insurance and Claims Managers")
st.markdown("**Group 6 — Aparna Manukonda | Sharan Kothuru | Nandini Bandla | Lokesh Goalla**")
st.write("This dashboard helps insurance managers forecast budgets, set premiums, and analyze costs for chronic conditions like Diabetes and Dialysis.")

# -----------------------------
# LOAD DATA (Simulated for Demo)
# -----------------------------
np.random.seed(42)
years = np.arange(2015, 2025)
conditions = ['Diabetes', 'Dialysis']
cities = ['Los Angeles', 'San Diego', 'San Jose', 'Sacramento', 'Fresno']
payers = [f'Payer_{i}' for i in range(1, 8)]

data = pd.DataFrame({
    'Year': np.random.choice(years, 1000),
    'City': np.random.choice(cities, 1000),
    'Payer': np.random.choice(payers, 1000),
    'ConditionType': np.random.choice(conditions, 1000),
    'TotalCost': np.random.randint(2000, 50000, 1000),
    'Patients': np.random.randint(50, 500, 1000),
    'CoveragePercent': np.random.uniform(60, 95, 1000)
})

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔍 Filter Options")

year_filter = st.sidebar.multiselect("Select Year", sorted(data['Year'].unique()), default=[2024])
condition_filter = st.sidebar.multiselect("Select Condition Type", conditions, default=conditions)
city_filter = st.sidebar.multiselect("Select City", cities, default=cities)

filtered_data = data[
    (data['Year'].isin(year_filter)) &
    (data['ConditionType'].isin(condition_filter)) &
    (data['City'].isin(city_filter))
]

# -----------------------------
# KPI METRICS
# -----------------------------
total_patients = int(filtered_data['Patients'].sum())
avg_cost_per_patient = round(filtered_data['TotalCost'].mean(), 2)
total_cost = round(filtered_data['TotalCost'].sum()/1e6, 2)  # in millions
payer_coverage = round(filtered_data['CoveragePercent'].mean(), 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", f"{total_patients:,}")
col2.metric("Avg Cost per Patient ($)", f"{avg_cost_per_patient:,}")
col3.metric("Total Cost (M$)", f"{total_cost}")
col4.metric("Avg Coverage %", f"{payer_coverage}%")

st.markdown("---")

# -----------------------------
# OVERVIEW CHARTS
# -----------------------------
st.subheader("📊 Overview")

# Total Cost by City
fig_city = px.bar(
    filtered_data.groupby('City', as_index=False)['TotalCost'].sum(),
    x='City', y='TotalCost', color='City',
    title="Total Cost by City"
)

# Total Cost Over Years
fig_year = px.line(
    filtered_data.groupby('Year', as_index=False)['TotalCost'].sum(),
    x='Year', y='TotalCost', title="Total Cost Over Years", markers=True
)

col5, col6 = st.columns(2)
col5.plotly_chart(fig_city, use_container_width=True)
col6.plotly_chart(fig_year, use_container_width=True)

# -----------------------------
# DEMOGRAPHICS / CONDITION INSIGHT
# -----------------------------
st.subheader("👥 Demographics and Condition Insights")

fig_condition = px.pie(
    filtered_data, names='ConditionType', values='TotalCost',
    title="Cost Distribution by Condition Type", color_discrete_sequence=px.colors.qualitative.Set2
)

fig_payer = px.bar(
    filtered_data.groupby('Payer', as_index=False)['TotalCost'].sum(),
    x='Payer', y='TotalCost', title="Total Cost by Payer", color='Payer'
)

col7, col8 = st.columns(2)
col7.plotly_chart(fig_condition, use_container_width=True)
col8.plotly_chart(fig_payer, use_container_width=True)

# -----------------------------
# FINANCIAL & PREDICTIVE ANALYSIS
# -----------------------------
st.subheader("💰 Financial & Predictive Analysis")

growth_rate = st.slider("Adjust Cost Growth (%)", 0, 20, 5)
adjusted_data = filtered_data.copy()
adjusted_data['AdjustedCost'] = adjusted_data['TotalCost'] * (1 + growth_rate/100)
adjusted_total_cost = adjusted_data['AdjustedCost'].sum()/1e6

col9, col10 = st.columns(2)
col9.metric("Adjusted Total Cost (M$)", f"{adjusted_total_cost:.2f}")
col10.metric("Growth Rate Applied", f"{growth_rate}%")

fig_growth = px.line(
    adjusted_data.groupby('Year', as_index=False)['AdjustedCost'].sum(),
    x='Year', y='AdjustedCost',
    title=f"Projected Claim Cost Trend ({growth_rate}% Growth)",
    markers=True, color_discrete_sequence=['#009688']
)
st.plotly_chart(fig_growth, use_container_width=True)

st.markdown("---")

# -----------------------------
# NOTES / DOCUMENTATION
# -----------------------------
st.subheader("📝 Notes & Project Documentation")

st.write("""
**Objective:**  
To help insurance managers forecast healthcare budgets and simulate premium adjustments for chronic conditions.

**Pages Included:**  
- **Overview:** KPIs, Cost by City, and Cost Trends  
- **Demographics:** Cost breakdown by condition and payer  
- **Financial & Predictive:** 'What-If' cost simulation  

**Outcome:**  
This dashboard transforms raw healthcare data into meaningful financial insights for decision-making.
""")
