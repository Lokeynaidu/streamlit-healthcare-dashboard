import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="Healthcare Analytics GUI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the theme colors (based on the blue/teal from the Power BI screenshots)
ACCENT_COLOR = '#007bff' # A bright blue for highlights (used for bars/lines)
SECONDARY_COLOR = '#1f77b4' # Plotly default blue, good for charts (used for headers)
BACKGROUND_COLOR = '#FFFFFF'

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    /* General Styling and Font (Trying to emulate Segoe UI/clean corporate look) */
    html, body, [class*="stApp"] {
        font-family: 'Segoe UI', Inter, sans-serif;
        color: #333333;
        background-color: #f0f2f6; /* Light gray background for a clean feel */
    }
    
    /* Header Styling */
    h1 {
        font-weight: 700;
        color: #1f77b4; /* Dark blue/teal */
        border-bottom: 2px solid #007bff;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    
    h2, h3 {
        color: #333333;
        font-weight: 600;
    }

    /* Custom Card Style for KPIs/Text Boxes (Similar to Power BI cards) */
    .stCustomCard {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%; 
    }
    
    /* KPI Value Style */
    .kpi-value {
        font-size: 2.2em; 
        font-weight: bold;
        color: #1f77b4;
        margin-top: 10px;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    
    /* KPI Label Style */
    .kpi-label {
        font-size: 0.85em; 
        color: #666;
        margin: 0;
    }
    
    /* Overriding Streamlit's default components to fit card style */
    div[data-testid="stMetricValue"] {
        font-size: 2.2em !important;
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. MOCK DATA GENERATION ---

@st.cache_data
def generate_mock_data(n_rows=10000):
    """Generates complex mock data for the dashboard based on the required dimensions."""
    np.random.seed(42) # for reproducibility

    # Target KPIs from screenshots:
    # Total Patients: 9060
    # Total Cost: ~1.91 Billion
    # Avg Cost per Patient: ~210.88K
    
    total_patients = 9060
    avg_cost_per_encounter = 18000 # Average cost per encounter/claim
    
    # Dimensions
    conditions = ['Diabetes', 'Dialysis', 'Other']
    condition_weights = [0.33, 0.33, 0.34] 
    age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    age_weights = [0.05, 0.05, 0.1, 0.15, 0.20, 0.20, 0.15, 0.1]
    genders = ['M', 'F']
    races = ['asian', 'black', 'hawaiian', 'native', 'other', 'white']
    payers = [f'Payer_{i}' for i in range(1, 11)]
    cities = ['Adelanto', 'Agoura Hills', 'Alameda', 'Alhambra', 'Also Viejo', 'American Canyon', 'Anaheim', 'Antelope', 'Los Angeles', 'San Diego', 'San Francisco', 'Sacramento']
    
    # Create patient-level data first
    patients = pd.DataFrame({
        'PatientID': range(1, total_patients + 1),
        'AgeGroup': np.random.choice(age_groups, size=total_patients, p=age_weights),
        'Gender': np.random.choice(genders, size=total_patients),
        'Race': np.random.choice(races, size=total_patients),
        'ConditionType': np.random.choice(conditions, size=total_patients, p=condition_weights),
        'City': np.random.choice(cities, size=total_patients),
        'Payer': np.random.choice(payers, size=total_patients),
        'StartYear': np.random.randint(1950, 2025, size=total_patients) 
    })

    # Create encounter/claim level data
    patient_claims = np.random.randint(1, 6, total_patients)
    data_list = []
    current_date = datetime.now().year
    
    for i, row in patients.iterrows():
        n_claims = patient_claims[i]
        for _ in range(n_claims):
            claim_cost = np.random.normal(loc=avg_cost_per_encounter, scale=avg_cost_per_encounter / 3)
            if row['ConditionType'] in ['Dialysis', 'Diabetes']:
                claim_cost *= np.random.uniform(1.5, 3.5)
            
            claim_cost = max(100, claim_cost)
            claim_year = np.random.randint(row['StartYear'], current_date + 1)
            
            data_list.append({
                'PatientID': row['PatientID'],
                'AgeGroup': row['AgeGroup'],
                'Gender': row['Gender'],
                'Race': row['Race'],
                'ConditionType': row['ConditionType'],
                'City': row['City'],
                'Payer': row['Payer'],
                'ClaimCost': claim_cost,
                'ClaimYear': claim_year
            })

    df = pd.DataFrame(data_list)
    df = df.merge(patients[['PatientID', 'StartYear']], on='PatientID', how='left')
    
    total_cost_calculated = df['ClaimCost'].sum()
    st.session_state['initial_total_cost'] = total_cost_calculated
    
    return df

df_full = generate_mock_data()


# --- 2. FILTER & STATE MANAGEMENT ---

# Set initial page state
if 'page' not in st.session_state:
    st.session_state['page'] = 'Overview' 
    
# --- Sidebar (Filters & Navigation) ---
with st.sidebar:
    st.title("Healthcare Analytics GUI")
    st.markdown("---")
    
    # Navigation
    page = st.selectbox(
        "Select Dashboard Page",
        ['Overview', 'Demographics', 'Financial & Predictive', 'Payer Analysis', 'High-Risk Segmentation'],
        key='page_select'
    )
    st.session_state['page'] = page
    st.markdown("---")
    st.subheader("Global Filters")

    # Filters 
    min_year = int(df_full['ClaimYear'].min())
    max_year = int(df_full['ClaimYear'].max())
    year_range = st.slider(
        'Year Range',
        min_value=min_year,
        max_value=max_year,
        value=(2000, max_year) 
    )
    
    all_conditions = df_full['ConditionType'].unique().tolist()
    selected_conditions = st.multiselect(
        'Condition Type',
        options=all_conditions,
        default=all_conditions
    )
    
    all_age_groups = sorted(df_full['AgeGroup'].unique().tolist())
    selected_age_groups = st.multiselect(
        'Age Group',
        options=all_age_groups,
        default=all_age_groups
    )
    
    all_races = df_full['Race'].unique().tolist()
    selected_races = st.multiselect(
        'Race/Ethnicity',
        options=all_races,
        default=all_races
    )

    # Apply filters
    df_filtered = df_full[
        (df_full['ClaimYear'] >= year_range[0]) & 
        (df_full['ClaimYear'] <= year_range[1]) &
        (df_full['ConditionType'].isin(selected_conditions)) &
        (df_full['AgeGroup'].isin(selected_age_groups)) &
        (df_full['Race'].isin(selected_races))
    ]

    st.markdown("---")
    st.caption(f"Filtered Data: **{len(df_filtered):,}** Claims")


# --- Helper Functions ---

def get_kpis(df):
    """Calculates the main Key Performance Indicators."""
    total_cost = df['ClaimCost'].sum()
    total_patients = df['PatientID'].nunique()
    
    avg_cost_per_patient = total_cost / total_patients if total_patients > 0 else 0
    # Mocking the value to match the screenshot style
    payer_coverage_percent = 2.88
    
    return total_patients, avg_cost_per_patient, total_cost, payer_coverage_percent

def format_value(value, is_currency=True):
    """Formats large numbers for KPI cards."""
    if value >= 1e9:
        return f"{value / 1e9:.2f}bn"
    elif value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}K"
    else:
        return f"{value:,.0f}"

def render_kpi_cards(total_patients, avg_cost_per_patient, total_cost, payer_coverage_percent):
    """Renders the four main KPI cards at the top using custom styling."""
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stCustomCard" style="text-align: center;">
                <p class="kpi-label">Total Patients</p>
                <p class="kpi-value" style="color: {SECONDARY_COLOR};">{total_patients:,.0f}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stCustomCard" style="text-align: center;">
                <p class="kpi-label">Avg Cost per Patient</p>
                <p class="kpi-value" style="color: {SECONDARY_COLOR};">${format_value(avg_cost_per_patient)}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="stCustomCard" style="text-align: center;">
                <p class="kpi-label">Total Cost</p>
                <p class="kpi-value" style="color: {SECONDARY_COLOR};">${format_value(total_cost)}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="stCustomCard" style="text-align: center;">
                <p class="kpi-label">Payer Coverage %</p>
                <p class="kpi-value" style="color: {SECONDARY_COLOR};">{payer_coverage_percent:.2f}%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    st.markdown("---")


# --- 3. PAGE FUNCTIONS ---

def overview_page(df):
    """Renders the main Overview page content."""
    
    st.title("Overview: Key Performance Indicators & Trends")
    
    if df.empty:
        st.warning("No data matches the selected filters. Please adjust the filters in the sidebar.")
        return

    total_patients, avg_cost_per_patient, total_cost, payer_coverage_percent = get_kpis(df)
    render_kpi_cards(total_patients, avg_cost_per_patient, total_cost, payer_coverage_percent)

    # Charts
    col1, col2 = st.columns([7, 3])

    with col1:
        st.subheader("Total Cost Over Years")
        df_yearly = df.groupby('ClaimYear')['ClaimCost'].sum().reset_index()
        
        fig_time = px.bar(
            df_yearly, 
            x='ClaimYear', 
            y='ClaimCost', 
            title='Total Claim Cost Trend',
            labels={'ClaimYear': 'Claim Year', 'ClaimCost': 'Total Cost (USD)'},
            color_discrete_sequence=[ACCENT_COLOR]
        )
        fig_time.update_traces(marker_line_width=0, opacity=0.8)
        fig_time.update_layout(
            xaxis_tickformat='d', 
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        st.subheader("Geographic Distribution of Claim Cost")
        st.info("Top 10 Cities by Patient Count.")
        df_city = df.groupby('City').agg(
            {'PatientID': 'nunique', 'ClaimCost': 'sum'}
        ).rename(columns={'PatientID': 'Total Patients', 'ClaimCost': 'Total Cost'}).reset_index()
        
        df_city['Total Cost'] = df_city['Total Cost'].apply(lambda x: f"${format_value(x)}")
        
        st.dataframe(
            df_city.sort_values('Total Patients', ascending=False).head(10),
            column_order=('City', 'Total Patients', 'Total Cost'),
            hide_index=True,
            use_container_width=True
        )


def demographics_page(df):
    """Renders the Demographics page content."""
    
    st.title("Demographics and Condition Insights")

    if df.empty:
        st.warning("No data matches the selected filters. Please adjust the filters in the sidebar.")
        return
    
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("Total Claim Cost by Age Group and Gender")
        df_age_gender = df.groupby(['AgeGroup', 'Gender'])['ClaimCost'].sum().reset_index()
        
        age_group_order = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        df_age_gender['AgeGroup'] = pd.Categorical(df_age_gender['AgeGroup'], categories=age_group_order, ordered=True)
        df_age_gender = df_age_gender.sort_values('AgeGroup')
        
        fig_bar = px.bar(
            df_age_gender, 
            x='ClaimCost', 
            y='AgeGroup', 
            color='Gender',
            orientation='h',
            title='Cost Breakdown by Age & Gender',
            labels={'ClaimCost': 'Total Cost (USD)', 'AgeGroup': 'Age Group'},
            color_discrete_map={'M': SECONDARY_COLOR, 'F': ACCENT_COLOR} 
        )
        fig_bar.update_layout(
            yaxis={'categoryorder':'array', 'categoryarray':age_group_order[::-1]},
            xaxis_title="Total Cost",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Patient Distribution by Condition Type")
        df_condition = df.groupby('ConditionType')['PatientID'].nunique().reset_index().rename(columns={'PatientID': 'Total Patients'})
        
        fig_pie = px.pie(
            df_condition, 
            values='Total Patients', 
            names='ConditionType', 
            title='Condition Distribution',
            color_discrete_sequence=[SECONDARY_COLOR, ACCENT_COLOR, 'lightgray']
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("High-Cost Patient Segments by City")
        st.markdown("Cities with the highest cumulative claim costs.")
        df_city_summary = df.groupby('City').agg(
            {'PatientID': 'nunique', 'ClaimCost': 'sum'}
        ).rename(columns={'PatientID': 'Total Patients', 'ClaimCost': 'Total Cost'}).reset_index()
        
        st.dataframe(
            df_city_summary.sort_values('Total Cost', ascending=False).head(5).style.format({'Total Cost': lambda x: f"${x:,.0f}"}),
            column_order=('City', 'Total Patients', 'Total Cost'),
            hide_index=True,
            use_container_width=True
        )


def financial_predictive_page(df):
    """Renders the Financial & Predictive page content, including the What-If simulation."""
    
    st.title("Financial & Predictive Analysis")

    if df.empty:
        st.warning("No data matches the selected filters. Please adjust the filters in the sidebar.")
        return

    # --- What-If Parameter ---
    st.subheader("Budget Forecasting Simulation")
    cost_growth_percent = st.slider(
        'Cost Growth % (What-If Parameter)', 
        min_value=0, 
        max_value=20, 
        value=5, 
        step=1,
        help="Simulate the financial impact of rising chronic care treatment costs by applying this percentage growth rate to the current total cost."
    )
    
    # Calculate initial and projected costs
    total_cost_current = df['ClaimCost'].sum()
    adjusted_cost = total_cost_current * (1 + cost_growth_percent / 100)
    
    st.markdown("---")
    
    # --- KPIs and Gauges ---
    col1, col2, col3 = st.columns([3, 4, 3])
    
    with col1:
        st.subheader("Total Cost by Payer (Top 10)")
        df_payer = df.groupby('Payer')['ClaimCost'].sum().reset_index().sort_values('ClaimCost', ascending=False).head(10)
        
        fig_payer = px.bar(
            df_payer, 
            x='ClaimCost', 
            y='Payer', 
            orientation='h',
            title='Cost Distribution by Top Payer',
            labels={'ClaimCost': 'Total Cost', 'Payer': 'Payer ID'},
            color_discrete_sequence=[SECONDARY_COLOR]
        )
        fig_payer.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_tickformat='$'
        )
        st.plotly_chart(fig_payer, use_container_width=True)

    with col2:
        st.subheader("Average Claim vs Target")
        target_cost = 421760 # Mock target value from screenshot (421.76K)
        current_avg = df['ClaimCost'].mean()
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_avg,
            number = {'prefix': "$", 'valueformat': ".0f"},
            title = {'text': "Average Claim Cost"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, target_cost], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': ACCENT_COLOR},
                'steps': [
                    {'range': [0, target_cost * 0.75], 'color': "lightgray"},
                    {'range': [target_cost * 0.75, target_cost], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target_cost
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=50, b=0, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col3:
        st.subheader("Forecasted Budget")
        st.markdown(
            f"""
            <div class="stCustomCard" style="text-align: center; border: 2px solid #DC3545; background-color: #FFF0F0;">
                <p class="kpi-label">Projected Adjusted Cost (Next Period)</p>
                <p style="font-size: 32px; font-weight: bold; color: #DC3545; margin-top: 5px; margin-bottom: 5px;">${format_value(adjusted_cost)}</p>
                <p class="kpi-label" style="font-size: 14px; color: #DC3545; margin-top: 10px;">Simulated Growth: **+{cost_growth_percent}%**</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # --- Monthly Trend Chart ---
    st.subheader("Claim Cost Trend with Projection")
    
    df_trend = df.groupby('ClaimYear')['ClaimCost'].sum().reset_index()
    df_trend = df_trend[df_trend['ClaimYear'] >= 2010] 
    
    projected_year = df_trend['ClaimYear'].max() + 1
    projected_cost = total_cost_current * (1 + cost_growth_percent / 100)
    
    df_projected = pd.DataFrame({
        'ClaimYear': [projected_year],
        'ClaimCost': [projected_cost],
        'Type': ['Projected']
    })
    
    df_trend['Type'] = 'Historical'
    df_trend_combined = pd.concat([df_trend, df_projected], ignore_index=True)
    
    fig_trend = px.line(
        df_trend_combined, 
        x='ClaimYear', 
        y='ClaimCost', 
        color='Type',
        title=f'Historical Trend and {projected_year} Projection',
        markers=True,
        labels={'ClaimYear': 'Year', 'ClaimCost': 'Total Cost (USD)'},
        color_discrete_map={'Historical': SECONDARY_COLOR, 'Projected': '#DC3545'} 
    )
    
    fig_trend.update_traces(
        marker=dict(size=12, symbol='circle', line=dict(width=1, color='DarkSlateGrey')), 
        selector=dict(mode='markers', name='Projected')
    )
    fig_trend.update_layout(hovermode="x unified")
    
    st.plotly_chart(fig_trend, use_container_width=True)


def payer_analysis_page(df):
    """Renders the dedicated Payer Analysis page."""
    
    st.title("Payer Analysis: Breakdown of Coverage and Cost Drivers")
    
    if df.empty:
        st.warning("No data matches the selected filters. Please adjust the filters in the sidebar.")
        return

    col1, col2 = st.columns([5, 5])
    
    with col1:
        st.subheader("Cost Distribution by Payer and Condition")
        df_payer_cond = df.groupby(['Payer', 'ConditionType'])['ClaimCost'].sum().reset_index()
        
        fig_cond = px.bar(
            df_payer_cond, 
            x='Payer', 
            y='ClaimCost', 
            color='ConditionType',
            title='Total Cost by Payer, Segmented by Chronic Condition',
            labels={'ClaimCost': 'Total Cost (USD)', 'ConditionType': 'Condition'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_cond.update_layout(xaxis_tickangle=-45, legend_title_text='Condition')
        st.plotly_chart(fig_cond, use_container_width=True)

    with col2:
        st.subheader("Average Claim Cost by Payer")
        df_payer_avg = df.groupby('Payer')['ClaimCost'].mean().reset_index().sort_values('ClaimCost', ascending=False)
        
        fig_avg = px.bar(
            df_payer_avg, 
            x='Payer', 
            y='ClaimCost', 
            color_discrete_sequence=[ACCENT_COLOR],
            title='Average Claim Value per Payer',
            labels={'ClaimCost': 'Average Claim Cost (USD)'}
        )
        fig_avg.update_layout(xaxis_tickangle=-45, yaxis_tickformat='$')
        st.plotly_chart(fig_avg, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Payer Detail Table")
    
    df_payer_detail = df.groupby('Payer').agg(
        Total_Patients=('PatientID', 'nunique'),
        Total_Claims=('PatientID', 'size'),
        Total_Cost=('ClaimCost', 'sum'),
        Avg_Claim_Cost=('ClaimCost', 'mean')
    ).reset_index()

    st.dataframe(
        df_payer_detail.sort_values('Total_Cost', ascending=False).style.format({
            'Total_Cost': '${:,.0f}',
            'Avg_Claim_Cost': '${:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )

def risk_segmentation_page(df):
    """Renders the High-Risk Patient Segmentation page."""
    
    st.title("High-Risk Segmentation: Identifying Top-Cost Patients")

    if df.empty:
        st.warning("No data matches the selected filters. Please adjust the filters in the sidebar.")
        return

    # --- Top N Selector ---
    df_patient_summary = df.groupby('PatientID').agg(
        Total_Claim_Cost=('ClaimCost', 'sum'),
        Claim_Count=('ClaimCost', 'size'),
        AgeGroup=('AgeGroup', 'first'),
        ConditionType=('ConditionType', 'first'),
        Payer=('Payer', 'first')
    ).reset_index()
    
    df_patient_summary = df_patient_summary.sort_values('Total_Claim_Cost', ascending=False)
    
    max_top_n = min(250, df_patient_summary['PatientID'].nunique())
    
    top_n = st.slider(
        'Select Top N High-Cost Patients for Drilldown',
        min_value=10, 
        max_value=max_top_n, 
        value=50, 
        step=10,
        help=f"Analyze the top {max_top_n} patients who account for the highest cumulative claim costs."
    )
    
    df_top_n = df_patient_summary.head(top_n).copy()
    
    total_top_n_cost = df_top_n['Total_Claim_Cost'].sum()
    percent_of_total_cost = (total_top_n_cost / df_full['ClaimCost'].sum()) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"Total Cost for Top {top_n} Patients", 
            value=f"${format_value(total_top_n_cost)}",
            delta=f"{percent_of_total_cost:.1f}% of Grand Total Cost"
        )
    with col2:
        st.metric(
            label="Average Cost per High-Risk Patient",
            value=f"${format_value(df_top_n['Total_Claim_Cost'].mean())}"
        )

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(f"Condition Distribution of Top {top_n} Patients")
        df_cond_top = df_top_n.groupby('ConditionType').size().reset_index(name='Count')
        
        fig_cond_top = px.bar(
            df_cond_top,
            x='ConditionType',
            y='Count',
            color='ConditionType',
            title='High-Risk Patients by Condition',
            color_discrete_sequence=[SECONDARY_COLOR, ACCENT_COLOR, 'lightgray']
        )
        st.plotly_chart(fig_cond_top, use_container_width=True)

    with col4:
        st.subheader(f"Age Group Distribution of Top {top_n} Patients")
        df_age_top = df_top_n.groupby('AgeGroup').size().reset_index(name='Count')
        
        fig_age_top = px.pie(
            df_age_top,
            values='Count',
            names='AgeGroup',
            title='High-Risk Patients by Age Group'
        )
        st.plotly_chart(fig_age_top, use_container_width=True)
        
    st.markdown("---")
    st.subheader(f"Detail Table: Claims for Top {top_n} Patients")
    
    df_detail_table = df_top_n[['PatientID', 'AgeGroup', 'ConditionType', 'Payer', 'Total_Claim_Cost', 'Claim_Count']].copy()
    
    st.dataframe(
        df_detail_table.style.format({'Total_Claim_Cost': '${:,.0f}'}),
        use_container_width=True,
        hide_index=True
    )


# --- 4. MAIN APP LOGIC ---

if st.session_state['page'] == 'Overview':
    overview_page(df_filtered)
elif st.session_state['page'] == 'Demographics':
    demographics_page(df_filtered)
elif st.session_state['page'] == 'Financial & Predictive':
    financial_predictive_page(df_filtered)
elif st.session_state['page'] == 'Payer Analysis':
    payer_analysis_page(df_filtered)
elif st.session_state['page'] == 'High-Risk Segmentation':
    risk_segmentation_page(df_filtered)
