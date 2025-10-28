import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="Healthcare Analytics GUI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the theme colors
ACCENT_COLOR = '#007bff' # Bright Blue
SECONDARY_COLOR = '#1f77b4' # Darker Blue
FRAUD_COLOR = '#DC3545' # Red for warnings/fraud
BACKGROUND_COLOR = '#FFFFFF'

# --- Custom CSS for Professional Look (Unchanged) ---
st.markdown("""
<style>
    /* General Styling and Font (Trying to emulate Segoe UI/clean corporate look) */
    html, body, [class*="stApp"] {
        font-family: 'Segoe UI', Inter, sans-serif;
        color: #333333;
        background-color: #f0f2f6;
    }
    
    /* Header Styling */
    h1 {
        font-weight: 700;
        color: #1f77b4;
        border-bottom: 2px solid #007bff;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    
    h2, h3 {
        color: #333333;
        font-weight: 600;
    }

    /* Custom Card Style for KPIs/Text Boxes */
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
    
    /* Overriding Streamlit's default components */
    div[data-testid="stMetricValue"] {
        font-size: 2.2em !important;
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. DATA LOADING AND VALIDATION (Multi-File) ---

# ----------------------------------------------------------------------
# === USER ACTION REQUIRED: REPLACE THESE PATHS WITH YOUR ACTUAL CSV FILE PATHS ===
# NOTE: Using the six key files identified: claims, encounters, patients, conditions, providers, payers
CSV_FILE_PATHS = {
    'claims': '/Users/glokesh/Documents/MRPCSV/claims.csv', 
    'encounters': '/Users/glokesh/Documents/MRPCSV/encounters.csv', 
    'patients': '/Users/glokesh/Documents/MRPCSV/patients.csv',
    'conditions': '/Users/glokesh/Documents/MRPCSV/conditions.csv',
    'providers': '/Users/glokesh/Documents/MRPCSV/providers.csv',
    'payers': '/Users/glokesh/Documents/MRPCSV/payers.csv',
    
}  
# ----------------------------------------------------------------------

REQUIRED_COLUMNS = [
    'PatientID', 'ClaimCost', 'AgeGroup', 'ConditionType', 
    'Payer', 'ClaimYear', 'Gender',
    'ClaimFilingDate', 'ProcessingTimeDays', 'FraudFlag', 'ProviderID', 'SettlementDurationDays'
]

@st.cache_data(show_spinner="Loading and merging 6 datasets...")
def load_data_from_path(file_paths):
    """Loads, merges, and validates the multiple CSV files using a Claims-centric model."""
    
    if any('data/' in p for p in file_paths.values()):
        return None, "Please update all placeholder paths in the `CSV_FILE_PATHS` dictionary with your actual file locations."

    data_frames = {}
    try:
        for name, path in file_paths.items():
            # Load with low_memory=False to handle potential type issues in large Synthea files
            data_frames[name] = pd.read_csv(path, low_memory=False).rename(columns=lambda x: x.strip())
            
        # 2. Merge Logic: Start with CLAIMS (Fact)
        df_full = data_frames['claims'].copy()
        
        # --- 2a. Merge Claims with Encounters (to get Cost, Date, and Encounter details) ---
        # Assuming Claims.APPOINTMENTID links to Encounters.Id
        # We only select necessary columns from Encounters and drop the redundant 'Id' before merging.
        df_encounters = data_frames['encounters'][
            ['Id', 'START', 'TOTAL_CLAIM_COST', 'PROVIDER', 'PAYER', 'REASONDESCRIPTION']
        ].rename(columns={'Id': 'Encounter_Id'}) # Rename 'Id' to avoid conflict
        
        df_full = df_full.merge(
            df_encounters, 
            left_on='APPOINTMENTID', 
            right_on='Encounter_Id', 
            how='left'
        )
        
        # --- 2b. Standardize Columns and Consolidate IDs ---
        # NOTE: Columns from Encounters merge without suffixes now, e.g., 'TOTAL_CLAIM_COST'
        df_full = df_full.rename(columns={
            'PATIENTID': 'PatientID',
            'TOTAL_CLAIM_COST': 'ClaimCost', # Cost from Encounter
            'SERVICEDATE': 'ClaimFilingDate_raw',     # Date from Claims
            'PRIMARYPATIENTINSURANCEID': 'PayerID_claim', # ID from Claims
            'STATUSP': 'ClaimStatus',                 # Primary status for daily tracking
            'PROVIDER': 'ProviderID_enc',             # ID from Encounters
            'PAYER': 'PayerID_enc',                   # ID from Encounters
            'START': 'Encounter_Start_Date',
            'PROVIDERID': 'ProviderID_claim'
        })
        
        # Consolidate Provider ID (Prioritize Claims Provider ID if available)
        df_full['ProviderID'] = df_full['ProviderID_claim'].fillna(df_full['ProviderID_enc'])
        
        # Consolidate Payer ID
        df_full['PayerID'] = df_full['PayerID_claim'].fillna(df_full['PayerID_enc'])
        
        # Consolidate Date (Prioritize Claims Service Date, then Encounter Start Date)
        df_full['ClaimFilingDate'] = pd.to_datetime(df_full['ClaimFilingDate_raw'].fillna(df_full['Encounter_Start_Date']), errors='coerce', utc=True)
        # Convert to localizable datetime object
        df_full['ClaimFilingDate'] = df_full['ClaimFilingDate'].dt.tz_localize(None) 
        

        # --- 2c. Merge Dimensions (Patients, Payers, Conditions) ---
        
        # Patients (for Age, Gender, City)
        # Assuming patient ID is 'Id' in patients.csv
        df_full = df_full.merge(
            data_frames['patients'][['Id', 'BIRTHDATE', 'GENDER', 'RACE', 'CITY']], 
            left_on='PatientID', 
            right_on='Id', 
            how='left'
        ).rename(columns={'GENDER': 'Gender', 'RACE': 'Race', 'CITY': 'City'})
        df_full.drop(columns=['Id'], inplace=True, errors='ignore') # Drop patient Id column after merge
        
        # Payers (for Payer Name)
        df_full = df_full.merge(
            data_frames['payers'][['Id', 'NAME']], 
            left_on='PayerID', 
            right_on='Id', 
            how='left'
        ).rename(columns={'NAME': 'Payer'})
        df_full.drop(columns=['Id'], inplace=True, errors='ignore') # Drop payer Id column after merge
        
        # Conditions (for ConditionType - using the first condition for the patient)
        if 'DESCRIPTION' in data_frames['conditions'].columns:
            # Group conditions by patient and take the first one found as the primary condition
            df_conditions_agg = data_frames['conditions'].groupby('PATIENT')['DESCRIPTION'].first().reset_index()
            df_full = df_full.merge(
                df_conditions_agg,
                left_on='PatientID',
                right_on='PATIENT',
                how='left'
            ).rename(columns={'DESCRIPTION': 'ConditionType'})
        else:
            df_full['ConditionType'] = 'Unspecified'


        # --- 2d. Final Cleaning, Calculations, and Mock Operational Fields ---
        
        df_full['ClaimCost'] = pd.to_numeric(df_full['ClaimCost'], errors='coerce')
        df_full['ClaimYear'] = df_full['ClaimFilingDate'].dt.year

        # Calculate approximate AgeGroup from BIRTHDATE
        if 'BIRTHDATE' in df_full.columns:
            now = pd.Timestamp('now')
            df_full['BIRTHDATE'] = pd.to_datetime(df_full['BIRTHDATE'], errors='coerce')
            df_full['Age'] = (now - df_full['BIRTHDATE']).dt.days // 365
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 200]
            labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
            df_full['AgeGroup'] = pd.cut(df_full['Age'], bins=bins, labels=labels, right=False).astype(str).replace('nan', 'Unknown')
        else:
            df_full['AgeGroup'] = 'Unknown'

        # Mock operational fields if not present (necessary for Daily/Weekly pages)
        if 'ProcessingTimeDays' not in df_full.columns:
            df_full['ProcessingTimeDays'] = np.random.randint(5, 60, len(df_full))
        if 'SettlementDurationDays' not in df_full.columns:
            df_full['SettlementDurationDays'] = df_full['ProcessingTimeDays'] + np.random.randint(0, 30, len(df_full))
        if 'FraudFlag' not in df_full.columns:
            df_full['FraudFlag'] = np.random.choice([0, 1], len(df_full), p=[0.98, 0.02])
        
        
        # 3. Final Validation
        df_full.dropna(subset=['ClaimCost', 'ClaimFilingDate', 'PatientID'], inplace=True)
        
        # Check if all final required columns are present
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_full.columns]
        if missing_cols:
             return None, f"Data Error: Missing critical columns after merge/mapping: {', '.join(missing_cols)}. Please check your file contents and column names against the assumptions in the code."

        if df_full.empty:
            return None, "Data Error: After cleaning, the combined dataset is empty. Check your data quality and join integrity."

        st.session_state['initial_total_cost'] = df_full['ClaimCost'].sum()
        
        return df_full, None

    except FileNotFoundError as e:
        return None, f"File Not Found Error: One or more files could not be found. Check the path: {e}"
    except Exception as e:
        # Catch any remaining errors for debugging
        return None, f"An unexpected error occurred during data loading and merging: {e}"

# Load data directly using the path
df_full, error_message = load_data_from_path(CSV_FILE_PATHS)

# --- 2. FILTER & STATE MANAGEMENT ---

# Set initial page state
if 'page' not in st.session_state:
    st.session_state['page'] = 'Daily Operations' 
    
# --- Handle No Data State ---
if df_full is None:
    st.info("### ⚠️ Data Loading Error")
    st.error(error_message)
    st.markdown(f"""
    ---
    **ACTION REQUIRED:** Please update the file paths in the `CSV_FILE_PATHS` dictionary in the code.
    
    **Required Columns (across all merged files):**
    - `PatientID`, `ClaimCost`, `AgeGroup`, `ConditionType`, `Payer`, `ClaimYear`, `Gender`, 
    - **Operational Fields:** `ClaimFilingDate`, `ProcessingTimeDays`, `FraudFlag` (0/1), `ProviderID`, `SettlementDurationDays`
    """)
    st.stop()


# --- Sidebar (Filters & Navigation) ---
with st.sidebar:
    st.title("Healthcare Analytics GUI")
    st.caption("Data Model: 6 files merged on key identifiers.")
    st.markdown("---")
    
    # Navigation
    page = st.selectbox(
        "Select Dashboard Page",
        [
            'Daily Operations', 
            'Weekly Performance', 
            'Monthly Finance', 
            'Demographics & Conditions', 
            'Payer & Risk Segmentation'
        ],
        key='page_select'
    )
    st.session_state['page'] = page
    st.markdown("---")
    st.subheader("Global Filters")

    # Filters 
    min_date = df_full['ClaimFilingDate'].min().date()
    max_date = df_full['ClaimFilingDate'].max().date()
    
    date_range = st.date_input(
        'Date Range',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date) 
    )
    
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
    else:
        start_date = pd.to_datetime(min_date)
        end_date = pd.to_datetime(max_date) + timedelta(days=1)


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
    
    # Apply filters
    df_filtered = df_full[
        (df_full['ClaimFilingDate'] >= start_date) & 
        (df_full['ClaimFilingDate'] < end_date) &
        (df_full['ConditionType'].isin(selected_conditions)) &
        (df_full['AgeGroup'].isin(selected_age_groups))
    ].copy() 

    st.markdown("---")
    st.caption(f"Claims in view: **{len(df_filtered):,}**")


# --- Helper Functions ---

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

def render_kpi_cards(kpis):
    """Renders dynamic KPI cards from a dictionary."""
    cols = st.columns(len(kpis))
    
    for i, (label, value, value_format, color) in enumerate(kpis):
        with cols[i]:
            display_value = value_format.format(value)
            st.markdown(
                f"""
                <div class="stCustomCard" style="text-align: center; border-left: 5px solid {color};">
                    <p class="kpi-label">{label}</p>
                    <p class="kpi-value" style="color: {color};">{display_value}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    st.markdown("---")

# --- 3. PAGE FUNCTIONS ---

def daily_operations_page(df):
    """Daily View: Tracks ongoing claims, pending, processing time, and fraud."""
    st.title("Daily Operations: Claims Tracking & Fraud Detection")

    if df.empty:
        st.warning("No data matches the selected filters.")
        return
        
    # Using ClaimStatus derived from the Claims.STATUSP field
    df['ClaimStatus_Display'] = df['ClaimStatus'].apply(lambda x: str(x).replace('_', ' ').title().strip())
    
    # Calculate KPIs for today (or the last day in the filter range)
    last_day = df['ClaimFilingDate'].max().date()
    df_today = df[df['ClaimFilingDate'].dt.date == last_day]
    
    # 1. KPIs
    avg_processing_time = df_today['ProcessingTimeDays'].mean() if not df_today.empty else 0
    fraud_claims = len(df_today[df_today['FraudFlag'] == 1])
    claims_filed_today = len(df_today)
    pending_claims = len(df_today[df_today['ClaimStatus_Display'].str.contains('Pending|Open|Review', na=False)])

    kpis = [
        ("Claims Filed (Today)", claims_filed_today, "{:,.0f}", SECONDARY_COLOR),
        ("Avg Processing Time (Days)", avg_processing_time, "{:.1f}", ACCENT_COLOR),
        ("Fraud Flagged (Today)", fraud_claims, "{:,.0f}", FRAUD_COLOR),
        ("Pending Claims (Filter)", pending_claims, "{:,.0f}", SECONDARY_COLOR),
    ]
    render_kpi_cards(kpis)
    
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("Claims Status Breakdown (Filter Period)")
        df_status = df['ClaimStatus_Display'].value_counts().reset_index()
        df_status.columns = ['Status', 'Count']
        
        # Color mapping based on status names
        status_map = {
            s: FRAUD_COLOR if 'Fraud' in s else ACCENT_COLOR if 'Open' in s or 'Pending' in s or 'Review' in s else SECONDARY_COLOR 
            for s in df_status['Status']
        }
        
        fig_status = px.bar(
            df_status, x='Status', y='Count', 
            color='Status',
            color_discrete_map=status_map,
            title='Current Claims Status Distribution'
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col2:
        st.subheader("Top Cost Conditions")
        df_cond = df.groupby('ConditionType')['ClaimCost'].sum().nlargest(5).reset_index()
        
        fig_cond = px.pie(
            df_cond, 
            values='ClaimCost', 
            names='ConditionType', 
            title='Top 5 Conditions by Total Cost'
        )
        st.plotly_chart(fig_cond, use_container_width=True)


def weekly_performance_page(df):
    """Weekly View: Performance tracking, cost trends, settlement duration, and provider ranking."""
    st.title("Weekly Performance: Trend & Provider Analysis")

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Aggregate data by week
    df_weekly = df.set_index('ClaimFilingDate').resample('W')['ClaimCost'].agg(
        Total_Cost='sum', 
        Avg_Cost='mean'
    ).reset_index()
    
    # Calculate Weekly KPIs
    avg_settlement_duration = df['SettlementDurationDays'].mean()
    total_weekly_cost = df_weekly['Total_Cost'].mean() if not df_weekly.empty else 0
    
    kpis = [
        ("Avg Cost (Per Week)", total_weekly_cost, f"${format_value(total_weekly_cost)}", SECONDARY_COLOR),
        ("Avg Settlement Duration (Days)", avg_settlement_duration, "{:.1f}", ACCENT_COLOR),
    ]
    render_kpi_cards(kpis)

    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.subheader("Weekly Claim Cost Trend")
        
        fig_trend = px.line(
            df_weekly, 
            x='ClaimFilingDate', y='Total_Cost', 
            title='Total Claim Cost Trend (Weekly)',
            labels={'Total_Cost': 'Total Cost (USD)', 'ClaimFilingDate': 'Week Ending'},
            markers=True,
            color_discrete_sequence=[ACCENT_COLOR]
        )
        fig_trend.update_xaxes(dtick="W", tickformat="%b %d") # Format for weekly ticks
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.subheader("Top 5 Providers by Claim Volume")
        df_providers = df['ProviderID'].value_counts().nlargest(5).reset_index()
        df_providers.columns = ['Provider', 'Claims']
        
        fig_providers = px.bar(
            df_providers, x='Claims', y='Provider', 
            orientation='h',
            title='Provider Ranking by Claim Count',
            color_discrete_sequence=[SECONDARY_COLOR]
        )
        fig_providers.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_providers, use_container_width=True)


def monthly_finance_page(df):
    """Monthly View: Financial overview, budget tracking, and readmission rates."""
    st.title("Monthly Finance: Strategy & Budget Overview")

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    df['Month'] = df['ClaimFilingDate'].dt.to_period('M').astype(str)
    df_monthly_actual = df.groupby('Month')['ClaimCost'].sum().reset_index().rename(columns={'ClaimCost': 'Actual_Spend'})
    
    # --- MOCK BUDGET LOGIC (No finance_targets.csv) ---
    avg_monthly_cost = df_monthly_actual['Actual_Spend'].mean()
    df_monthly = df_monthly_actual.copy()
    df_monthly['Budget'] = avg_monthly_cost * 1.1 
    
    # Per Member Per Month (PMPM) Cost Calculation
    df_monthly['TotalPatients'] = df.groupby('Month')['PatientID'].nunique().values
    df_monthly['CostPerMember'] = df_monthly['Actual_Spend'] / df_monthly['TotalPatients']
    
    # Mock Readmission Rate (This data needs to be tracked at the patient level over time for accuracy)
    avg_readmission = 0.08 

    # 1. KPIs
    total_cost_current = df['ClaimCost'].sum()
    
    kpis = [
        ("Total Claim Cost (Filter)", total_cost_current, f"${format_value(total_cost_current)}", SECONDARY_COLOR),
        ("Avg Readmission Rate (Mock)", avg_readmission * 100, "{:.2f}%", ACCENT_COLOR),
        ("Avg Cost Per Member (PMPM)", df_monthly['CostPerMember'].mean(), f"${format_value(df_monthly['CostPerMember'].mean())}", SECONDARY_COLOR),
    ]
    render_kpi_cards(kpis)

    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("Budget vs. Actual Spend (Monthly)")
        st.info("Budget is modeled as 110% of the average monthly historical spend due to missing target data.")
        
        fig_budget = go.Figure()
        fig_budget.add_trace(go.Bar(
            x=df_monthly['Month'], y=df_monthly['Actual_Spend'], name='Actual Spend', marker_color=ACCENT_COLOR
        ))
        fig_budget.add_trace(go.Scatter(
            x=df_monthly['Month'], y=df_monthly['Budget'], name='Budget', mode='lines+markers', line=dict(color=FRAUD_COLOR, width=3)
        ))
        fig_budget.update_layout(title='Budget vs. Actual Claim Cost')
        st.plotly_chart(fig_budget, use_container_width=True)
        
    with col2:
        st.subheader("Cost Forecasting (What-If)")
        
        cost_growth_percent = st.slider(
            'Cost Growth % (Next Period)', 
            min_value=0, 
            max_value=20, 
            value=5, 
            step=1,
            key='monthly_forecast_slider'
        )
        
        adjusted_cost = total_cost_current * (1 + cost_growth_percent / 100)
        
        st.markdown(
            f"""
            <div class="stCustomCard" style="text-align: center; border: 2px solid #DC3545; background-color: #FFF0F0;">
                <p class="kpi-label">Projected Annual Budget Need</p>
                <p style="font-size: 30px; font-weight: bold; color: #DC3545; margin-top: 5px; margin-bottom: 5px;">${format_value(adjusted_cost)}</p>
                <p class="kpi-label" style="font-size: 14px; color: #DC3545;">Simulated Growth: **+{cost_growth_percent}%**</p>
            </div>
            """, 
            unsafe_allow_html=True
        )


def demographics_conditions_page(df):
    """Consolidated Demographics and Condition Insights Page."""
    
    st.title("Demographics & Conditions: Patient Profile and Cost Drivers")

    if df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # KPIs for this page
    total_patients = df['PatientID'].nunique()
    avg_cost_per_patient = df['ClaimCost'].sum() / total_patients if total_patients > 0 else 0
    
    kpis = [
        ("Total Patients", total_patients, "{:,.0f}", SECONDARY_COLOR),
        ("Avg Cost Per Patient", avg_cost_per_patient, f"${format_value(avg_cost_per_patient)}", ACCENT_COLOR),
    ]
    render_kpi_cards(kpis)
    
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("Total Claim Cost by Age Group and Gender")
        df_age_gender = df.groupby(['AgeGroup', 'Gender'])['ClaimCost'].sum().reset_index()
        age_group_order = sorted(df['AgeGroup'].unique().tolist())
        
        if age_group_order:
            df_age_gender['AgeGroup'] = pd.Categorical(df_age_gender['AgeGroup'], categories=age_group_order, ordered=True)
            df_age_gender = df_age_gender.sort_values('AgeGroup')
        
        fig_bar = px.bar(
            df_age_gender, x='ClaimCost', y='AgeGroup', color='Gender', orientation='h',
            title='Cost Breakdown by Age & Gender',
            labels={'ClaimCost': 'Total Cost (USD)', 'AgeGroup': 'Age Group'},
            color_discrete_map={'M': SECONDARY_COLOR, 'F': ACCENT_COLOR} 
        )
        if age_group_order:
            fig_bar.update_layout(yaxis={'categoryorder':'array', 'categoryarray':age_group_order[::-1]})
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
        st.plotly_chart(fig_pie, use_container_width=True)


def payer_risk_segmentation_page(df):
    """Consolidated Payer Analysis and High-Risk Segmentation Page."""
    
    st.title("Payer & Risk Segmentation: Financial Deep Dive (Quarterly Report Focus)")
    st.info("This page aligns with the Quarterly need for long-term cost and risk analysis.")

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    # --- Payer Analysis Section ---
    st.subheader("Payer Analysis Summary")
    
    total_cost = df['ClaimCost'].sum()
    
    # Dynamic Payer Coverage %: (Percentage of claims above the median cost)
    total_claims = len(df)
    threshold = df['ClaimCost'].quantile(0.5) if total_claims > 0 else 0
    covered_claims = len(df[df['ClaimCost'] > threshold])
    payer_coverage_percent = (covered_claims / total_claims) * 100 if total_claims > 0 else 0
    
    kpis = [
        ("Total Claims", total_claims, "{:,.0f}", SECONDARY_COLOR),
        ("Total Cost", total_cost, f"${format_value(total_cost)}", ACCENT_COLOR),
        ("Coverage Metric %", payer_coverage_percent, "{:.2f}%", SECONDARY_COLOR),
    ]
    render_kpi_cards(kpis)

    col1, col2 = st.columns([5, 5])
    
    with col1:
        st.subheader("Cost Distribution by Payer and Condition")
        df_payer_cond = df.groupby(['Payer', 'ConditionType'])['ClaimCost'].sum().reset_index()
        
        fig_cond = px.bar(
            df_payer_cond, 
            x='Payer', y='ClaimCost', color='ConditionType',
            title='Total Cost by Payer, Segmented by Chronic Condition',
            labels={'ClaimCost': 'Total Cost (USD)'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_cond, use_container_width=True)

    with col2:
        st.subheader("Payer Detail Table (Cost)")
        df_payer_detail = df.groupby('Payer').agg(
            Total_Patients=('PatientID', 'nunique'),
            Total_Cost=('ClaimCost', 'sum'),
            Avg_Claim_Cost=('ClaimCost', 'mean')
        ).reset_index()

        st.dataframe(
            df_payer_detail.sort_values('Total_Cost', ascending=False).style.format({
                'Total_Cost': '${:,.0f}',
                'Avg_Claim_Cost': '${:,.0f}'
            }).head(10),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    
    # --- Risk Segmentation Section ---
    st.subheader("High-Risk Patient Segmentation")

    df_patient_summary = df.groupby('PatientID').agg(
        Total_Claim_Cost=('ClaimCost', 'sum'),
        AgeGroup=('AgeGroup', 'first'),
        ConditionType=('ConditionType', 'first')
    ).reset_index()
    
    df_patient_summary = df_patient_summary.sort_values('Total_Claim_Cost', ascending=False)
    
    max_top_n = min(100, df_patient_summary['PatientID'].nunique())
    
    if max_top_n < 10:
        top_n = max_top_n
    else:
        top_n = st.slider(
            'Select Top N High-Cost Patients',
            min_value=10, max_value=max_top_n, value=min(50, max_top_n), step=10,
            key='risk_top_n'
        )
    
    df_top_n = df_patient_summary.head(top_n).copy()
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            label=f"Total Cost for Top {top_n} Patients", 
            value=f"${format_value(df_top_n['Total_Claim_Cost'].sum())}",
        )
    with col4:
        st.metric(
            label="Avg Cost per High-Risk Patient",
            value=f"${format_value(df_top_n['Total_Claim_Cost'].mean())}"
        )

    st.markdown("---")
    st.dataframe(
        df_top_n.style.format({'Total_Claim_Cost': '${:,.0f}'}),
        use_container_width=True,
        hide_index=True
    )


# --- 4. MAIN APP LOGIC ---

if st.session_state['page'] == 'Daily Operations':
    daily_operations_page(df_filtered)
elif st.session_state['page'] == 'Weekly Performance':
    weekly_performance_page(df_filtered)
elif st.session_state['page'] == 'Monthly Finance':
    monthly_finance_page(df_filtered)
elif st.session_state['page'] == 'Demographics & Conditions':
    demographics_conditions_page(df_filtered)
elif st.session_state['page'] == 'Payer & Risk Segmentation':
    payer_risk_segmentation_page(df_filtered)
