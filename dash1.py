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

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
    <h2 style='text-align: center; color: #1565c0;'>🏥 Healthcare Analytics Dashboard</h2>
    <h4 style='text-align: center; color: #5c6bc0;'>For Insurance and Claims Managers</h4>
""", unsafe_allow_html=True)

st.write("")
st.markdown("This dashboard helps insurance managers analyze healthcare claims, readmissions, and forecast budgets for chronic conditions like **Diabetes** and **Dialysis**.")

# -----------------------------
# SAMPLE DATA (Synthetic for Demo)
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
    'CoveragePercent': np.random.uniform(60, 95, 1000),
    'Readmissions': np.random.randint(0, 100, 1000)
})

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔍 Filter Data")

year_filter = st.sidebar.multiselect("Select Year", sorted(data['Year'].unique()), default=[2024])
condition_filter = st.sidebar.multiselect("Select Condition", conditions, default=conditions)
city_filter = st.sidebar.multiselect("Select City", cities, default=cities)

filtered_data = data[
    (data['Year'].isin(year_filter)) &
    (data['ConditionType'].isin(condition_filter)) &
    (data['City'].isin(city_filter))
]

# -----------------------------
# KPI METRICS SECTION
# -----------------------------
st.markdown("### 📈 Key Performance Indicators")

total_patients = int(filtered_data['Patients'].sum())
avg_cost_per_patient = round(filtered_data['TotalCost'].mean(), 2)
total_cost = round(filtered_data['TotalCost'].sum()/1e6, 2)
avg_coverage = round(filtered_data['CoveragePercent'].mean(), 2)
total_readmissions = int(filtered_data['Readmissions'].sum())
readmission_rate = round((total_readmissions / total_patients) * 100, 2)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("👥 Total Patients", f"{total_patients:,}")
col2.metric("💰 Total Cost (M$)", f"{total_cost}")
col3.metric("💵 Avg Cost per Patient", f"${avg_cost_per_patient:,}")
col4.metric("📊 Avg Coverage %", f"{avg_coverage}%")
col5.metric("🔄 Total Readmissions", f"{total_readmissions:,}")
col6.metric("📈 Readmission Rate", f"{readmission_rate}%")

st.markdown("---")

# -----------------------------
# COST OVERVIEW & TRENDS
# -----------------------------
st.markdown("### 🌍 Cost Overview & Trends")

col5, col6 = st.columns(2)

# Total Cost by City
city_cost = filtered_data.groupby('City', as_index=False)['TotalCost'].sum()
fig_city = px.bar(
    city_cost, x='City', y='TotalCost',
    title="Total Cost by City",
    color='City',
    color_discrete_sequence=px.colors.sequential.Blues
)
fig_city.update_layout(template="simple_white")

# Total Cost Over Years
yearly_cost = filtered_data.groupby('Year', as_index=False)['TotalCost'].sum()
fig_year = px.line(
    yearly_cost, x='Year', y='TotalCost',
    title="Total Cost Over Years",
    markers=True,
    color_discrete_sequence=['#1E88E5']
)
fig_year.update_layout(template="simple_white")

col5.plotly_chart(fig_city, use_container_width=True)
col6.plotly_chart(fig_year, use_container_width=True)

st.markdown("---")

# -----------------------------
# CONDITION ANALYSIS
# -----------------------------
st.markdown("### 🩺 Condition Insights")

col7, col8 = st.columns(2)

# Pie chart for Cost Distribution
fig_condition = px.pie(
    filtered_data, names='ConditionType', values='TotalCost',
    title="Cost Distribution by Condition",
    color_discrete_sequence=px.colors.sequential.Blues
)
fig_condition.update_traces(textinfo='percent+label')

# Bar chart for Patient Count
patient_condition = filtered_data.groupby('ConditionType', as_index=False)['Patients'].sum()
fig_patients = px.bar(
    patient_condition, x='ConditionType', y='Patients',
    title="Total Patients by Condition",
    text='Patients',
    color='ConditionType',
    color_discrete_sequence=px.colors.sequential.Blues_r
)
fig_patients.update_layout(template="simple_white")

col7.plotly_chart(fig_condition, use_container_width=True)
col8.plotly_chart(fig_patients, use_container_width=True)

st.markdown("---")

# -----------------------------
# READMISSIONS ANALYSIS
# -----------------------------
st.markdown("### 🔄 Readmissions Analysis")

col9, col10 = st.columns(2)

# Readmissions by City
read_city = filtered_data.groupby('City', as_index=False)['Readmissions'].sum()
fig_read_city = px.bar(
    read_city, x='City', y='Readmissions',
    title="Readmissions by City",
    color='City',
    color_discrete_sequence=px.colors.sequential.Blues_r
)
fig_read_city.update_layout(template="simple_white")

# Readmission Rate by Condition
read_condition = filtered_data.groupby('ConditionType', as_index=False)['Readmissions'].sum()
fig_read_condition = px.pie(
    read_condition, names='ConditionType', values='Readmissions',
    title="Readmissions by Condition",
    color_discrete_sequence=px.colors.sequential.Blues
)
fig_read_condition.update_traces(textinfo='percent+label')

col9.plotly_chart(fig_read_city, use_container_width=True)
col10.plotly_chart(fig_read_condition, use_container_width=True)

st.markdown("---")

# -----------------------------
# FINANCIAL & PREDICTIVE SECTION
# -----------------------------
st.markdown("### 💹 Predictive Cost Simulation")

growth_rate = st.slider("Simulate Cost Growth (%)", 0, 20, 5)
adjusted_data = filtered_data.copy()
adjusted_data['AdjustedCost'] = adjusted_data['TotalCost'] * (1 + growth_rate/100)
adjusted_total = adjusted_data['AdjustedCost'].sum()/1e6

col11, col12 = st.columns(2)
col11.metric("Projected Total Cost (M$)", f"{adjusted_total:.2f}")
col12.metric("Applied Growth Rate", f"{growth_rate}%")

# Projected Trend
fig_proj = px.line(
    adjusted_data.groupby('Year', as_index=False)['AdjustedCost'].sum(),
    x='Year', y='AdjustedCost',
    title=f"Projected Claim Cost Trend ({growth_rate}% Growth)",
    markers=True,
    color_discrete_sequence=['#1565C0']
)
fig_proj.update_layout(template="simple_white")
st.plotly_chart(fig_proj, use_container_width=True)

st.markdown("---")

# -----------------------------
# ADDITIONAL INSIGHTS (HEATMAP + SUMMARY)
# -----------------------------
st.markdown("### 📊 Additional Insights")

col13, col14 = st.columns(2)

# Heatmap: Average Cost by City and Year
heatmap_data = filtered_data.groupby(['City', 'Year'])['TotalCost'].mean().reset_index()
fig_heatmap = px.density_heatmap(
    heatmap_data, x='Year', y='City', z='TotalCost',
    color_continuous_scale='Blues',
    title="Average Claim Cost by City and Year"
)
fig_heatmap.update_layout(template="simple_white")

# Summary Table
summary = filtered_data.groupby(['Payer', 'ConditionType'])[['TotalCost', 'Patients', 'Readmissions']].sum().reset_index()
col13.plotly_chart(fig_heatmap, use_container_width=True)
col14.dataframe(summary, use_container_width=True, height=420)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:#5c6bc0;'>
Built by Group 6 — Healthcare Analytics Project (AA 5960)
</div>
""", unsafe_allow_html=True)
