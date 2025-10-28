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
ACCENT_COLOR = '#007bff' # A bright blue for highlights
SECONDARY_COLOR = '#1f77b4' # Plotly default blue, good for charts
BACKGROUND_COLOR = '#FFFFFF'

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
    
    # Calculate costs per patient to hit the target
    avg_cost_per_encounter = 18000 # Average cost per encounter/claim
    
    # Dimensions
    conditions = ['Diabetes', 'Dialysis', 'Other']
    condition_weights = [0.33, 0.33, 0.34] # Based on the pie chart (33.33% each)
    
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
        'StartYear': np.random.randint(1950, 2025, size=total_patients) # Year of first diagnosis/encounter
    })

    # Create encounter/claim level data
    # Each patient has 1 to 5 claims
    patient_claims = np.random.randint(1, 6, total_patients)
    data_list = []
    
    current_date = datetime.now().year
    
    for i, row in patients.iterrows():
        n_claims = patient_claims[i]
        for _ in range(n_claims):
            claim_cost = np.random.normal(loc=avg_cost_per_encounter, scale=avg_cost_per_encounter / 3)
            # Make Dialysis/Diabetes claims more expensive
            if row['ConditionType'] in ['Dialysis', 'Diabetes']:
                claim_cost *= np.random.uniform(1.5, 3.5)
            
            # Ensure costs are positive
            claim_cost = max(100, claim_cost)
            
            # Generate a realistic claim year
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
    
    # Recalculate Total Cost and check
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
    st.title("Healthcare Analytics")
    
    # Navigation
    page = st.selectbox(
        "Select Dashboard Page",
        ['Overview', 'Demographics', 'Financial & Predictive'],
        key='page_select'
    )
    st.session_state['page'] = page
    st.markdown("---")
    st.subheader("Global Filters")

    # Filters (based on the Overview/Demographics page structure)
    
    # Year Slider Filter
    min_year = int(df_full['ClaimYear'].min())
    max_year = int(df_full['ClaimYear'].max())
    year_range = st.slider(
        'Year Range',
        min_value=min_year,
        max_value=max_year,
        value=(2000, max_year) # Default to a more recent range for visualization clarity
    )
    
    # Condition Type Filter
    all_conditions = df_full['ConditionType'].unique().tolist()
    selected_conditions = st.multiselect(
        'Condition Type',
        options=all_conditions,
        default=all_conditions
    )
    
    # Age Group Filter
    all_age_groups = sorted(df_full['AgeGroup'].unique().tolist())
    selected_age_groups = st.multiselect(
        'Age Group',
        options=all_age_groups,
        default=all_age_groups
    )
    
    # Race Filter (optional based on space)
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

    # Display data stats after filtering
    st.markdown(f"**Data Loaded:** {len(df_filtered):,} Claims")


# --- Helper Functions ---

def get_kpis(df):
    """Calculates the main Key Performance Indicators."""
    total_cost = df['ClaimCost'].sum()
    total_patients = df['PatientID'].nunique()
    
    if total_patients > 0:
        avg_cost_per_patient = total_cost / total_patients
    else:
        avg_cost_per_patient = 0
        
    # Mock Payer Coverage % calculation (2.88K in screenshot)
    # Let's assume Payer Coverage % is a ratio of claims fully covered.
    # We will simply mock the value for display to match the screenshot style.
    payer_coverage_percent = 0.0288 * 100 # Displaying 2.88%
    
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
    """Renders the four main KPI cards at the top."""
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; background-color: {BACKGROUND_COLOR}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                <p style="font-size: 24px; font-weight: bold; color: {SECONDARY_COLOR}; margin-bottom: 5px;">{total_patients:,.0f}</p>
                <p style="font-size: 14px; color: #666; margin: 0;">Total Patients</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; background-color: {BACKGROUND_COLOR}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                <p style="font-size: 24px; font-weight: bold; color: {SECONDARY_COLOR}; margin-bottom: 5px;">${format_value(avg_cost_per_patient)}</p>
                <p style="font-size: 14px; color: #666; margin: 0;">Avg Cost per Patient</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; background-color: {BACKGROUND_COLOR}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                <p style="font-size: 24px; font-weight: bold; color: {SECONDARY_COLOR}; margin-bottom: 5px;">${format_value(total_cost)}</p>
                <p style="font-size: 14px; color: #666; margin: 0;">Total Cost</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; background-color: {BACKGROUND_COLOR}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                <p style="font-size: 24px; font-weight: bold; color: {SECONDARY_COLOR}; margin-bottom: 5px;">{payer_coverage_percent:.2f}%</p>
                <p style="font-size: 14px; color: #666; margin: 0;">Payer Coverage %</p>
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
        st.warning("No data matches the selected filters.")
        return

    total_patients, avg_cost_per_patient, total_cost, payer_coverage_percent = get_kpis(df)
    render_kpi_cards(total_patients, avg_cost_per_patient, total_cost, payer_coverage_percent)

    # Charts
    col1, col2 = st.columns([7, 3])

    with col1:
        st.subheader("Total Cost Over Years")
        # Aggregate by year
        df_yearly = df.groupby('ClaimYear')['ClaimCost'].sum().reset_index()
        
        # Line/Bar Chart for Total Cost Over Years
        fig_time = px.bar(
            df_yearly, 
            x='ClaimYear', 
            y='ClaimCost', 
            title='Total Cost Over Years (Filtered)',
            labels={'ClaimYear': 'Start Year', 'ClaimCost': 'Total Cost (USD)'},
            color_discrete_sequence=[ACCENT_COLOR]
        )
        fig_time.update_traces(marker_line_width=0, opacity=0.8)
        fig_time.update_layout(xaxis_tickformat='d') # Ensure year is displayed as integer
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        st.subheader("Geographic Distribution of Claim Cost (Mock Map)")
        st.warning(
            "A real map requires geographic coordinates (lat/lon). "
            "This is a text-based representation matching the screenshot intent."
        )
        # Aggregate by City
        df_city = df.groupby('City').agg(
            {'PatientID': 'nunique', 'ClaimCost': 'sum'}
        ).rename(columns={'PatientID': 'Total Patients', 'ClaimCost': 'Total Cost'}).reset_index()
        
        df_city['Total Cost'] = df_city['Total Cost'].apply(lambda x: f"${format_value(x)}")
        
        st.dataframe(
            df_city.sort_values('Total Patients', ascending=False),
            column_order=('City', 'Total Patients', 'Total Cost'),
            hide_index=True
        )


def demographics_page(df):
    """Renders the Demographics page content."""
    
    st.title("Demographics and Condition Insights")

    if df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.subheader("Total Claim Cost by Age Group and Gender")
        df_age_gender = df.groupby(['AgeGroup', 'Gender'])['ClaimCost'].sum().reset_index()
        
        # Use a custom order for AgeGroup
        age_group_order = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        df_age_gender['AgeGroup'] = pd.Categorical(df_age_gender['AgeGroup'], categories=age_group_order, ordered=True)
        df_age_gender = df_age_gender.sort_values('AgeGroup')
        
        # Bar Chart
        fig_bar = px.bar(
            df_age_gender, 
            x='ClaimCost', 
            y='AgeGroup', 
            color='Gender',
            orientation='h',
            title='Total Claim Cost by Age Group and Gender',
            labels={'ClaimCost': 'Total Cost (USD)', 'AgeGroup': 'Age Group'},
            color_discrete_map={'M': SECONDARY_COLOR, 'F': ACCENT_COLOR} # Blue for Male, Teal/Darker for Female
        )
        fig_bar.update_layout(yaxis={'categoryorder':'array', 'categoryarray':age_group_order[::-1]}) # Reverse order for visualization
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Distribution of Conditions")
        df_condition = df.groupby('ConditionType')['PatientID'].nunique().reset_index().rename(columns={'PatientID': 'Total Patients'})
        
        # Pie Chart
        fig_pie = px.pie(
            df_condition, 
            values='Total Patients', 
            names='ConditionType', 
            title='Patient Distribution by Condition Type',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("Patient and Cost Distribution Summary")
        # City-level summary table (similar to the one in the screenshot)
        df_city_summary = df.groupby('City').agg(
            {'PatientID': 'nunique', 'ClaimCost': 'sum'}
        ).rename(columns={'PatientID': 'Total Patients', 'ClaimCost': 'Total Cost'}).reset_index()
        
        df_city_summary['Total Cost'] = df_city_summary['Total Cost'].round(2)
        
        st.dataframe(
            df_city_summary.sort_values('Total Patients', ascending=False).head(10),
            column_order=('City', 'Total Patients', 'Total Cost'),
            hide_index=True,
            use_container_width=True
        )


def financial_predictive_page(df):
    """Renders the Financial & Predictive page content, including the What-If simulation."""
    
    st.title("Financial & Predictive Analysis")

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    # --- What-If Parameter ---
    st.subheader("Predictive Cost Simulation")
    cost_growth_percent = st.slider(
        'Cost Growth % (What-If Parameter)', 
        min_value=0, 
        max_value=20, 
        value=5, 
        step=1,
        help="Simulate the financial impact of rising treatment costs by applying this percentage growth to the Adjusted Cost."
    )
    
    # Calculate initial and projected costs
    total_cost_current = df['ClaimCost'].sum()
    adjusted_cost = total_cost_current * (1 + cost_growth_percent / 100)
    
    st.markdown("---")
    
    # --- KPIs and Gauges ---
    col1, col2, col3 = st.columns([3, 4, 3])
    
    with col1:
        st.subheader("Total Cost by Payer")
        df_payer = df.groupby('Payer')['ClaimCost'].sum().reset_index().sort_values('ClaimCost', ascending=False).head(10)
        
        # Bar chart for Payer Cost
        fig_payer = px.bar(
            df_payer, 
            x='ClaimCost', 
            y='Payer', 
            orientation='h',
            title='Top 10 Payers by Total Cost',
            labels={'ClaimCost': 'Total Cost (USD)', 'Payer': 'Payer ID'},
            color_discrete_sequence=[SECONDARY_COLOR]
        )
        fig_payer.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_payer, use_container_width=True)

    with col2:
        st.subheader("Average Claim vs Target")
        target_cost = 421760 # Mock target value from screenshot (421.76K)
        current_avg = df['ClaimCost'].mean()
        
        # Gauge Chart (Radial chart equivalent)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_avg,
            number = {'prefix': "$", 'valueformat': ".2f"},
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
        st.subheader("Forecasted Cost")
        st.markdown(
            f"""
            <div style="text-align: center; border: 1px solid #e0e0e0; border-radius: 8px; padding: 30px; margin-top: 30px; background-color: {BACKGROUND_COLOR}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                <p style="font-size: 32px; font-weight: bold; color: #DC3545; margin-bottom: 5px;">${format_value(adjusted_cost)}</p>
                <p style="font-size: 16px; color: #666; margin: 0;">Adjusted Cost (Next Period)</p>
                <p style="font-size: 14px; color: #999; margin-top: 10px;">Cost Growth: +{cost_growth_percent}%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # --- Monthly Trend Chart ---
    st.subheader("Monthly Claim Cost Trend")
    
    # Aggregate data by year (mocking monthly for simplicity and using year since we don't have month data)
    df_trend = df.groupby('ClaimYear')['ClaimCost'].sum().reset_index()
    df_trend = df_trend[df_trend['ClaimYear'] >= 2010] # Filter to recent years for better trend visualization
    
    # Apply the cost growth simulation to the projected year (max year + 1)
    projected_year = df_trend['ClaimYear'].max() + 1
    
    # Create the projected data point
    projected_cost = total_cost_current * (1 + cost_growth_percent / 100)
    
    df_projected = pd.DataFrame({
        'ClaimYear': [projected_year],
        'ClaimCost': [projected_cost],
        'Type': ['Projected']
    })
    
    df_trend['Type'] = 'Historical'
    df_trend_combined = pd.concat([df_trend, df_projected], ignore_index=True)
    
    # Scatter/Line Chart
    fig_trend = px.line(
        df_trend_combined, 
        x='ClaimYear', 
        y='ClaimCost', 
        color='Type',
        title='Monthly Claim Cost Trend (Simulated)',
        markers=True,
        labels={'ClaimYear': 'Year', 'ClaimCost': 'Total Cost (USD)'},
        color_discrete_map={'Historical': SECONDARY_COLOR, 'Projected': '#DC3545'} # Red for projected point
    )
    
    # Enhance the scatter point for the projection
    fig_trend.update_traces(
        marker=dict(size=12), 
        selector=dict(mode='markers', name='Projected')
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)


# --- 4. MAIN APP LOGIC ---

if st.session_state['page'] == 'Overview':
    overview_page(df_filtered)
elif st.session_state['page'] == 'Demographics':
    demographics_page(df_filtered)
elif st.session_state['page'] == 'Financial & Predictive':
    financial_predictive_page(df_filtered)
