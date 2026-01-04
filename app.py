"""
Nuclear Energy Projection Dashboard
=====================================

Interactive Streamlit dashboard for exploring nuclear energy projections,
IEA scenario comparisons, and avoided emissions analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Page config
st.set_page_config(
    page_title="Nuclear Energy Projection Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better formatting
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }

    /* Remove extra padding from containers */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    /* Fix chart sizing and spacing */
    .stPlotlyChart {
        height: auto !important;
        margin-bottom: 1.5rem;
    }

    /* Better spacing for metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 500;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }

    /* Section headers */
    h2 {
        color: #1f77b4;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #2c3e50;
        font-weight: 600;
        padding-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Improve dataframe display */
    .stDataFrame {
        font-size: 0.9rem;
    }

    /* Better column spacing */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }

    /* Improve radio button layout */
    .stRadio > div {
        flex-direction: row;
        gap: 1rem;
    }

    /* Better info/warning boxes */
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# Data Loading Functions
# ========================================

@st.cache_data
def load_all_data():
    """Load all processed data files"""
    data_dir = Path("data/processed")

    data = {
        'tracker': pd.read_csv(data_dir / "nuclear_tracker_cleaned.csv"),
        'projections': pd.read_csv(data_dir / "nuclear_projections_2050.csv"),
        'global_projections': pd.read_csv(data_dir / "global_projections_2050.csv"),
        'scenario_comparison': pd.read_csv(data_dir / "scenario_comparison.csv"),
        'global_gap': pd.read_csv(data_dir / "global_gap_analysis.csv"),
        'regional_gaps': pd.read_csv(data_dir / "regional_gaps_2050.csv"),
        'emissions': pd.read_csv(data_dir / "avoided_emissions.csv"),
        'global_emissions': pd.read_csv(data_dir / "global_avoided_emissions.csv"),
        'pipeline_contribution': pd.read_csv(data_dir / "pipeline_contribution_2050.csv")
    }

    return data


# Load data
with st.spinner("Loading data..."):
    data = load_all_data()

# ========================================
# Sidebar - Navigation & Filters
# ========================================

st.sidebar.markdown("## Nuclear Energy Dashboard")
st.sidebar.markdown("---")

# Navigation
analysis_type = st.sidebar.selectbox(
    "Select Analysis",
    ["🌍 Global Overview",
     "🏠 Regional Analysis",
     "📊 Projections 2025-2050",
     "🎯 IEA Scenario Comparison",
     "🌱 Avoided Emissions",
     "🔧 Pipeline Analysis",
     "📈 Statistical Analysis"]
)

st.sidebar.markdown("---")

# Filters
st.sidebar.markdown("### Filters")

# Year range filter
min_year = int(data['projections']['year'].min())
max_year = int(data['projections']['year'].max())
year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Region filter
all_regions = ['All'] + sorted(data['projections']['region'].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    all_regions,
    default=['All']
)

# Scenario filter (if enhanced model data available)
scenario_options = ['Base', 'Conservative', 'Aggressive']
selected_scenario = st.sidebar.selectbox(
    "Projection Scenario",
    scenario_options
)

# Define scenario multipliers for projections
SCENARIO_MULTIPLIERS = {
    'Base': 1.0,
    'Conservative': 0.85,
    'Aggressive': 1.20
}
scenario_multiplier = SCENARIO_MULTIPLIERS[selected_scenario]

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard visualizes nuclear energy projections through 2050,
    comparing current pipeline against IEA Net Zero scenarios.

    **Data Sources**:
    - Global Nuclear Power Tracker (Sept 2025)
    - IEA Net Zero by 2050 Scenario

    **Project**: Climate Data Science Research
    """
)

# ========================================
# Main Content Area
# ========================================

# Header
st.markdown('<div class="main-header"> Nuclear Energy Projection Dashboard</div>',
            unsafe_allow_html=True)

# ========================================
# Page 1: Global Overview
# ========================================

if analysis_type == "🌍 Global Overview":
    st.markdown("## Global Overview")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_plants = len(data['tracker'])
        st.metric("Total Plants", f"{total_plants:,}")

    with col2:
        total_capacity = data['tracker']['Capacity (MW)'].sum() / 1000
        st.metric("Total Capacity", f"{total_capacity:,.0f} GW")

    with col3:
        projection_2050 = data['global_projections'][data['global_projections']['year'] == 2050]['generation_twh'].values[0] * scenario_multiplier
        st.metric("2050 Projection", f"{projection_2050:,.0f} TWh")

    with col4:
        total_avoided = data['global_emissions'][data['global_emissions']['year'] == 2050]['avoided_emissions_mtco2'].values[0] * scenario_multiplier
        st.metric("2050 Avoided Emissions", f"{total_avoided:,.0f} MtCO2")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Plant Status Distribution")

        status_counts = data['tracker']['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        fig = px.pie(
            status_counts.head(5),
            names='Status',
            values='Count',
            title="Plant Status Breakdown (Top 5)",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Regional Distribution")

        region_counts = data['tracker']['Region'].value_counts().head(10).reset_index()
        region_counts.columns = ['Region', 'Count']

        fig = px.bar(
            region_counts,
            x='Count',
            y='Region',
            orientation='h',
            title="Top 10 Regions by Plant Count",
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Global projections timeline
    st.markdown("### Global Nuclear Generation Projection (2025-2050)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['global_projections']['year'],
        y=data['global_projections']['generation_twh'],
        mode='lines+markers',
        name='Projected Generation',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Generation (TWh/year)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

# ========================================
# Page 2: Regional Analysis
# ========================================

elif analysis_type == "🏠 Regional Analysis":
    st.markdown("## Regional Analysis")

    # Filter by selected regions
    if 'All' not in selected_regions and len(selected_regions) > 0:
        regional_data = data['projections'][data['projections']['region'].isin(selected_regions)].copy()
    else:
        regional_data = data['projections'].copy()

    # Apply scenario multiplier
    regional_data['generation_twh'] = regional_data['generation_twh'] * scenario_multiplier
    regional_data['capacity_gw'] = regional_data['capacity_gw'] * scenario_multiplier

    # Regional projections by year
    st.markdown("### Regional Generation Projections")

    fig = px.line(
        regional_data[(regional_data['year'] >= year_range[0]) & (regional_data['year'] <= year_range[1])],
        x='year',
        y='generation_twh',
        color='region',
        title=f"Regional Nuclear Generation ({year_range[0]}-{year_range[1]})",
        markers=True
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Generation (TWh/year)",
        hovermode='x unified',
        template='plotly_white',
        legend_title="Region",
        height=500,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Regional capacity distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Regional Capacity (2050)")

        capacity_2050 = regional_data[regional_data['year'] == 2050].sort_values('capacity_gw', ascending=True)

        fig = px.bar(
            capacity_2050,
            x='capacity_gw',
            y='region',
            orientation='h',
            title="Regional Capacity in 2050",
            color='capacity_gw',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title="Capacity (GW)",
            yaxis_title="Region",
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Regional Generation Share (2050)")

        gen_2050 = regional_data[regional_data['year'] == 2050]

        fig = px.pie(
            gen_2050,
            names='region',
            values='generation_twh',
            title="Share of Global Generation in 2050"
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# Page 3: Projections 2025-2050
# ========================================

elif analysis_type == "📊 Projections 2025-2050":
    st.markdown("## Nuclear Generation Projections (2025-2050)")

    # Visualization type selector
    viz_type = st.radio(
        "Select Visualization Type",
        ["Line Chart", "Area Chart", "Bar Chart (by Year)", "Heatmap"],
        horizontal=True
    )

    # Filter data
    if 'All' not in selected_regions and len(selected_regions) > 0:
        proj_data = data['projections'][data['projections']['region'].isin(selected_regions)].copy()
    else:
        proj_data = data['projections'].copy()

    proj_data = proj_data[(proj_data['year'] >= year_range[0]) & (proj_data['year'] <= year_range[1])]

    # Apply scenario multiplier
    proj_data['generation_twh'] = proj_data['generation_twh'] * scenario_multiplier

    if viz_type == "Line Chart":
        fig = px.line(
            proj_data,
            x='year',
            y='generation_twh',
            color='region',
            markers=True,
            title="Nuclear Generation Projections by Region"
        )

    elif viz_type == "Area Chart":
        fig = px.area(
            proj_data,
            x='year',
            y='generation_twh',
            color='region',
            title="Nuclear Generation Projections (Stacked)"
        )

    elif viz_type == "Bar Chart (by Year)":
        # Group by year for selected years
        milestone_years = st.multiselect(
            "Select Years",
            sorted(proj_data['year'].unique()),
            default=[2030, 2040, 2050]
        )

        bar_data = proj_data[proj_data['year'].isin(milestone_years)]

        fig = px.bar(
            bar_data,
            x='year',
            y='generation_twh',
            color='region',
            title="Nuclear Generation by Year (Grouped)",
            barmode='group'
        )

    else:  # Heatmap
        pivot_data = proj_data.pivot(index='region', columns='year', values='generation_twh')

        fig = px.imshow(
            pivot_data,
            labels=dict(x="Year", y="Region", color="Generation (TWh)"),
            title="Nuclear Generation Heatmap",
            aspect="auto",
            color_continuous_scale="YlOrRd"
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Generation (TWh/year)",
        hovermode='x unified',
        template='plotly_white',
        height=550,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Growth analysis
    st.markdown("### Growth Analysis")

    global_growth = data['global_projections'].copy()
    global_growth['yoy_growth'] = global_growth['generation_twh'].pct_change() * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=global_growth['year'],
        y=global_growth['yoy_growth'],
        name='YoY Growth Rate',
        marker_color='lightblue'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Year-over-Year Growth Rate",
        xaxis_title="Year",
        yaxis_title="Growth Rate (%)",
        template='plotly_white',
        height=450,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

# ========================================
# Page 4: IEA Scenario Comparison
# ========================================

elif analysis_type == "🎯 IEA Scenario Comparison":
    st.markdown("## IEA Net Zero Scenario Comparison")

    # Apply scenario multiplier to gap data
    global_gap_data = data['global_gap'].copy()
    global_gap_data['generation_twh'] = global_gap_data['generation_twh'] * scenario_multiplier
    global_gap_data['gap_twh'] = global_gap_data['target_generation_twh'] - global_gap_data['generation_twh']

    regional_gaps_data = data['regional_gaps'].copy()
    regional_gaps_data['generation_twh'] = regional_gaps_data['generation_twh'] * scenario_multiplier
    regional_gaps_data['gap_twh'] = regional_gaps_data['target_generation_twh'] - regional_gaps_data['generation_twh']
    regional_gaps_data['gap_percentage'] = (regional_gaps_data['gap_twh'] / regional_gaps_data['target_generation_twh'] * 100).fillna(0)

    # Global gap analysis
    st.markdown("### Global Generation Gap Analysis")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=global_gap_data['year'],
        y=global_gap_data['generation_twh'],
        mode='lines+markers',
        name='Projected Generation',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=global_gap_data['year'],
        y=global_gap_data['target_generation_twh'],
        mode='lines+markers',
        name='IEA NZE Target',
        line=dict(color='green', width=2, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=global_gap_data['year'],
        y=global_gap_data['gap_twh'],
        mode='lines',
        name='Gap',
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='red', width=1)
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Generation (TWh/year)",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=550,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Regional gaps (2050)
    st.markdown("### Regional Gaps in 2050")

    col1, col2 = st.columns(2)

    with col1:
        # Gap bar chart
        regional_gaps_sorted = regional_gaps_data.sort_values('gap_twh')

        fig = go.Figure()

        colors = ['red' if gap > 0 else 'green' for gap in regional_gaps_sorted['gap_twh']]

        fig.add_trace(go.Bar(
            x=regional_gaps_sorted['gap_twh'],
            y=regional_gaps_sorted['region'],
            orientation='h',
            marker_color=colors,
            text=regional_gaps_sorted['gap_twh'].round(0),
            textposition='outside'
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="black")

        fig.update_layout(
            title="Regional Generation Gap (2050)",
            xaxis_title="Gap (TWh) - Negative = Surplus, Positive = Shortfall",
            yaxis_title="Region",
            template='plotly_white',
            showlegend=False,
            height=450,
            margin=dict(l=20, r=20, t=60, b=60)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Alignment status
        alignment_counts = regional_gaps_data['alignment'].value_counts().reset_index()
        alignment_counts.columns = ['Alignment', 'Count']

        fig = px.pie(
            alignment_counts,
            names='Alignment',
            values='Count',
            title="Regional Alignment Status (2050)",
            color='Alignment',
            color_discrete_map={
                'On track': 'green',
                'Shortfall': 'orange',
                'Major shortfall': 'red',
                'Exceeds target': 'blue'
            }
        )

        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Gap summary table
    st.markdown("### Gap Summary Table")

    gap_table = regional_gaps_data[['region', 'generation_twh', 'target_generation_twh',
                                        'gap_twh', 'gap_percentage', 'alignment']].copy()
    gap_table.columns = ['Region', 'Projected (TWh)', 'IEA Target (TWh)',
                         'Gap (TWh)', 'Gap (%)', 'Status']

    # Format numbers
    gap_table['Projected (TWh)'] = gap_table['Projected (TWh)'].round(1)
    gap_table['IEA Target (TWh)'] = gap_table['IEA Target (TWh)'].round(1)
    gap_table['Gap (TWh)'] = gap_table['Gap (TWh)'].round(1)
    gap_table['Gap (%)'] = gap_table['Gap (%)'].round(1)

    st.dataframe(gap_table, use_container_width=True, hide_index=True)

# ========================================
# Page 5: Avoided Emissions
# ========================================

elif analysis_type == "🌱 Avoided Emissions":
    st.markdown("## Avoided CO2 Emissions Analysis")

    # Apply scenario multiplier to emissions data
    global_emissions_data = data['global_emissions'].copy()
    global_emissions_data['avoided_emissions_mtco2'] = global_emissions_data['avoided_emissions_mtco2'] * scenario_multiplier
    global_emissions_data['cumulative_avoided_mtco2'] = global_emissions_data['cumulative_avoided_mtco2'] * scenario_multiplier
    global_emissions_data['cumulative_avoided_gtco2'] = global_emissions_data['cumulative_avoided_gtco2'] * scenario_multiplier

    emissions_data = data['emissions'].copy()
    emissions_data['avoided_emissions_mtco2'] = emissions_data['avoided_emissions_mtco2'] * scenario_multiplier

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        emissions_2050 = global_emissions_data[global_emissions_data['year'] == 2050]['avoided_emissions_mtco2'].values[0]
        st.metric("Annual Avoided (2050)", f"{emissions_2050:,.0f} MtCO2")

    with col2:
        cumulative = global_emissions_data['cumulative_avoided_mtco2'].iloc[-1] / 1000
        st.metric("Cumulative (2025-2050)", f"{cumulative:,.1f} GtCO2")

    with col3:
        car_equivalent = cumulative * 1000 / 7.5 / 1_000_000  # Convert to million cars
        st.metric("Equivalent (25 years)", f"{car_equivalent:,.0f}M cars")

    # Global emissions timeline
    st.markdown("### Global Avoided Emissions Over Time")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Annual Avoided Emissions", "Cumulative Avoided Emissions"),
        vertical_spacing=0.15
    )

    # Annual emissions
    fig.add_trace(
        go.Scatter(
            x=global_emissions_data['year'],
            y=global_emissions_data['avoided_emissions_mtco2'],
            mode='lines+markers',
            name='Annual',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,128,0,0.2)'
        ),
        row=1, col=1
    )

    # Cumulative emissions
    fig.add_trace(
        go.Scatter(
            x=global_emissions_data['year'],
            y=global_emissions_data['cumulative_avoided_gtco2'],
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='darkgreen', width=2)
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="MtCO2/year", row=1, col=1)
    fig.update_yaxes(title_text="GtCO2 (cumulative)", row=2, col=1)

    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Regional emissions (2050)
    st.markdown("### Regional Avoided Emissions (2050)")

    regional_emissions = emissions_data[emissions_data['year'] == 2050].sort_values('avoided_emissions_mtco2', ascending=True)

    fig = px.bar(
        regional_emissions,
        x='avoided_emissions_mtco2',
        y='region',
        orientation='h',
        title="Avoided Emissions by Region in 2050",
        color='avoided_emissions_mtco2',
        color_continuous_scale='Greens'
    )

    fig.update_layout(
        xaxis_title="Avoided Emissions (MtCO2/year)",
        yaxis_title="Region",
        showlegend=False,
        height=500,
        margin=dict(l=20, r=20, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

# ========================================
# Page 6: Pipeline Analysis
# ========================================

elif analysis_type == "🔧 Pipeline Analysis":
    st.markdown("## Nuclear Pipeline Analysis")

    st.markdown("### Pipeline Contribution to 2050 Generation")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Operating Plants',
        x=data['pipeline_contribution']['region'],
        y=data['pipeline_contribution']['operating_generation_twh'],
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        name='Pipeline Projects',
        x=data['pipeline_contribution']['region'],
        y=data['pipeline_contribution']['pipeline_generation_twh'],
        marker_color='orange'
    ))

    fig.update_layout(
        barmode='stack',
        title="2050 Generation: Operating vs Pipeline",
        xaxis_title="Region",
        yaxis_title="Generation (TWh/year)",
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=500,
        margin=dict(l=60, r=40, t=60, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Pipeline percentage
    st.markdown("### Pipeline Contribution Percentage")

    fig = px.bar(
        data['pipeline_contribution'].sort_values('pipeline_percentage', ascending=True),
        x='pipeline_percentage',
        y='region',
        orientation='h',
        title="Pipeline Projects as % of 2050 Generation",
        color='pipeline_percentage',
        color_continuous_scale='RdYlGn_r',
        text='pipeline_percentage'
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

    fig.update_layout(
        xaxis_title="Pipeline Contribution (%)",
        yaxis_title="Region",
        showlegend=False,
        height=500,
        margin=dict(l=20, r=60, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Units breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Operating vs Pipeline Units")

        units_data = pd.DataFrame({
            'Category': ['Operating', 'Pipeline'],
            'Units': [
                data['pipeline_contribution']['operating_units'].sum(),
                data['pipeline_contribution']['pipeline_units'].sum()
            ]
        })

        fig = px.pie(
            units_data,
            names='Category',
            values='Units',
            title="Plant Units: Operating vs Pipeline",
            color='Category',
            color_discrete_map={'Operating': 'blue', 'Pipeline': 'orange'}
        )

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Regional Pipeline Units")

        fig = px.bar(
            data['pipeline_contribution'].sort_values('pipeline_units', ascending=True),
            x='pipeline_units',
            y='region',
            orientation='h',
            title="Pipeline Units by Region",
            color='pipeline_units',
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=40),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

# ========================================
# Page 7: Statistical Analysis
# ========================================

elif analysis_type == "📈 Statistical Analysis":
    st.markdown("## Statistical Analysis")

    st.info("""
    This page provides statistical insights into the nuclear projection data.
    For detailed statistical tests and analysis, refer to the Statistical Analysis Notebook.
    """)

    # Capacity distribution
    st.markdown("### Plant Capacity Distribution")

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data['tracker']['Capacity (MW)'].dropna(),
        nbinsx=50,
        name='Capacity Distribution',
        marker_color='steelblue'
    ))

    fig.update_layout(
        title="Distribution of Plant Capacities",
        xaxis_title="Capacity (MW)",
        yaxis_title="Frequency",
        template='plotly_white',
        height=450,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Box plot by status
    st.markdown("### Capacity by Plant Status")

    # Filter for main status types
    status_filter = data['tracker']['Status'].str.lower().str.contains('operating|construction|announced', na=False)
    status_data = data['tracker'][status_filter]

    fig = px.box(
        status_data,
        x='Status',
        y='Capacity (MW)',
        title="Capacity Distribution by Plant Status",
        color='Status'
    )

    fig.update_layout(
        xaxis_title="Plant Status",
        yaxis_title="Capacity (MW)",
        showlegend=False,
        height=450,
        margin=dict(l=60, r=40, t=60, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    st.markdown("### Capacity vs Generation Correlation")

    # Create scatter plot
    features_data = data['tracker'][['Capacity (MW)']].copy()
    features_data['annual_generation_twh'] = (
        features_data['Capacity (MW)'] * 8760 * 0.9 / 1_000_000
    )
    features_data = features_data.dropna()

    fig = px.scatter(
        features_data,
        x='Capacity (MW)',
        y='annual_generation_twh',
        title="Capacity vs Annual Generation",
        trendline="ols",
        labels={
            'Capacity (MW)': 'Capacity (MW)',
            'annual_generation_twh': 'Annual Generation (TWh)'
        }
    )

    fig.update_layout(
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    correlation = features_data['Capacity (MW)'].corr(features_data['annual_generation_twh'])
    st.metric("Pearson Correlation", f"{correlation:.4f}")

    # Summary statistics
    st.markdown("### Summary Statistics")

    summary_stats = data['tracker']['Capacity (MW)'].describe().to_frame()
    summary_stats.columns = ['Value']
    summary_stats['Value'] = summary_stats['Value'].round(2)

    st.dataframe(summary_stats, use_container_width=True)

# ========================================
# Footer
# ========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Nuclear Energy Projection Dashboard | Data Science Research Project</p>
    <p>Data sources: Global Nuclear Power Tracker (Sept 2025), IEA NZE Scenario (2021)</p>
</div>
""", unsafe_allow_html=True)
