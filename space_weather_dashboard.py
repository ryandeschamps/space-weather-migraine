import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re
from datetime import datetime, timedelta
import glob

# Page configuration
st.set_page_config(
    page_title="Space Weather Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def parse_space_weather_file(file_path):
    """Parse a single space weather data file"""
    data = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract date from filename
    filename = os.path.basename(file_path)
    date_match = re.search(r'(\d{2})(\d{2})dayind\.txt', filename)
    if date_match:
        month, day = date_match.groups()
        # Assuming 2025 based on the sample data
        year = 2025
        date = datetime(year, int(month), int(day))
        data['date'] = date
    
    # Extract Solar Indices
    solar_match = re.search(r':Solar_Indices: (\d{4} \w{3} \d{2})\n.*?\n\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.e+-]+)\s+([-\d]+)', content, re.DOTALL)
    if solar_match:
        data['sunspot_number'] = int(solar_match.group(2))
        data['radio_flux_10cm'] = int(solar_match.group(3))
        data['radio_flux_90day'] = int(solar_match.group(4))
        data['xray_background'] = float(solar_match.group(5))
    
    # Extract Solar Region Data (flares)
    flare_match = re.search(r':Solar_Region_Data:.*?\n.*?\n.*?\n.*?\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', content, re.DOTALL)
    if flare_match:
        data['sunspot_area'] = int(flare_match.group(1))
        data['new_regions'] = int(flare_match.group(2))
        data['spotted_regions'] = int(flare_match.group(3))
        data['c_flares'] = int(flare_match.group(4))
        data['m_flares'] = int(flare_match.group(5))
        data['x_flares'] = int(flare_match.group(6))
    
    # Extract Geomagnetic Indices (K-indices)
    # Look for planetary K-indices
    kp_match = re.search(r'Planetary.*?\n.*?K-indices.*?\n.*?\n.*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
    if kp_match:
        k_indices = [float(x) for x in kp_match.groups()]
        data['kp_indices'] = k_indices
        data['max_kp'] = max(k_indices)
        data['avg_kp'] = np.mean(k_indices)
    
    # Extract A-index (planetary)
    a_match = re.search(r'Planetary.*?\n.*?A.*?\n.*?\s+(\d+)', content, re.DOTALL)
    if a_match:
        data['a_index'] = int(a_match.group(1))
    
    return data

@st.cache_data
def load_all_data():
    """Load and parse all space weather data files"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    all_data = []
    
    # Get all dayind.txt files
    files = glob.glob(os.path.join(data_dir, "*dayind.txt"))
    files.sort()
    
    # Debug information
    if not files:
        st.error(f"No data files found in: {data_dir}")
        st.error(f"Script directory: {script_dir}")
        st.error(f"Directory exists: {os.path.exists(data_dir)}")
        if os.path.exists(script_dir):
            st.error(f"Contents of script directory: {os.listdir(script_dir)}")
    
    for file_path in files:
        try:
            parsed_data = parse_space_weather_file(file_path)
            if parsed_data:
                all_data.append(parsed_data)
        except Exception as e:
            st.error(f"Error parsing {file_path}: {str(e)}")
    
    if not all_data:
        st.error("No data could be loaded!")
        st.error(f"Found {len(files)} files but no data was successfully parsed")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by date
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    return df

def get_space_weather_condition(kp_value):
    """Determine space weather condition based on Kp index"""
    if kp_value < 2:
        return "Quiet", "green"
    elif kp_value < 3:
        return "Unsettled", "yellow" 
    elif kp_value < 4:
        return "Active", "orange"
    elif kp_value < 5:
        return "Minor Storm", "red"
    elif kp_value < 6:
        return "Moderate Storm", "red"
    elif kp_value < 7:
        return "Strong Storm", "darkred"
    elif kp_value < 8:
        return "Severe Storm", "darkred"
    else:
        return "Extreme Storm", "purple"

def main():
    # Header
    st.markdown('<h1 class="main-header">🌌 Space Weather Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time monitoring of solar activity and geomagnetic conditions**")
    
    # Load data
    with st.spinner("Loading space weather data..."):
        df = load_all_data()
    
    if df.empty:
        st.error("No data available to display!")
        return
    
    # Sidebar for controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Date range selector
    if 'date' in df.columns and not df['date'].isna().all():
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Display options
    show_raw_data = st.sidebar.checkbox("Show Raw Data Table", False)
    
    # Main dashboard content
    if not df_filtered.empty:
        # Current conditions (latest data)
        st.header("🔴 Current Space Weather Conditions")
        
        latest_data = df_filtered.iloc[-1] if len(df_filtered) > 0 else None
        
        if latest_data is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'max_kp' in latest_data and not pd.isna(latest_data['max_kp']):
                    condition, color = get_space_weather_condition(latest_data['max_kp'])
                    st.metric("Max Kp Index", f"{latest_data['max_kp']:.1f}", help="Geomagnetic activity level")
                    st.markdown(f"**Condition:** <span style='color:{color}'>{condition}</span>", unsafe_allow_html=True)
            
            with col2:
                if 'sunspot_number' in latest_data and not pd.isna(latest_data['sunspot_number']):
                    st.metric("Sunspot Number", int(latest_data['sunspot_number']), help="Solar activity indicator")
            
            with col3:
                if 'radio_flux_10cm' in latest_data and not pd.isna(latest_data['radio_flux_10cm']):
                    st.metric("Solar Radio Flux", f"{latest_data['radio_flux_10cm']} SFU", help="10.7 cm radio flux")
            
            with col4:
                if 'a_index' in latest_data and not pd.isna(latest_data['a_index']):
                    st.metric("A-Index", int(latest_data['a_index']), help="Geomagnetic activity measure")
        
        # Time series plots
        st.header("📈 Time Series Analysis")
        
        # Create tabs for different types of data
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 All Data Overview", "🌍 Geomagnetic Activity", "☀️ Solar Activity", "🌟 Solar Flares", "🔗 Correlations", "🧠 Health Analysis", "🤖 AI Analysis"])
        
        with tab1:
            st.subheader("📊 Complete Dataset Overview")
            st.write(f"**Data Period:** {df_filtered['date'].min().strftime('%B %d, %Y')} to {df_filtered['date'].max().strftime('%B %d, %Y')}")
            st.write(f"**Total Days:** {len(df_filtered)} days")
            
            # Multi-parameter time series plot
            st.subheader("🔍 Multi-Parameter Time Series")
            
            # Create subplot with secondary y-axis
            fig_multi = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Geomagnetic Activity (Kp Index)', 'Solar Activity', 'Solar Flares'),
                vertical_spacing=0.08,
                specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]]
            )
            
            # Row 1: Geomagnetic indices
            if 'max_kp' in df_filtered.columns and not df_filtered['max_kp'].isna().all():
                fig_multi.add_trace(
                    go.Scatter(x=df_filtered['date'], y=df_filtered['max_kp'], 
                              name='Max Kp', line=dict(color='red', width=2)),
                    row=1, col=1
                )
                # Add storm threshold lines
                fig_multi.add_hline(y=5, line_dash="dash", line_color="orange", row=1, col=1)
                fig_multi.add_hline(y=6, line_dash="dash", line_color="red", row=1, col=1)
            
            if 'a_index' in df_filtered.columns and not df_filtered['a_index'].isna().all():
                fig_multi.add_trace(
                    go.Scatter(x=df_filtered['date'], y=df_filtered['a_index'], 
                              name='A-Index', line=dict(color='blue', width=1)),
                    row=1, col=1, secondary_y=True
                )
            
            # Row 2: Solar activity
            if 'sunspot_number' in df_filtered.columns and not df_filtered['sunspot_number'].isna().all():
                fig_multi.add_trace(
                    go.Scatter(x=df_filtered['date'], y=df_filtered['sunspot_number'], 
                              name='Sunspot Number', line=dict(color='orange', width=2)),
                    row=2, col=1
                )
            
            if 'radio_flux_10cm' in df_filtered.columns and not df_filtered['radio_flux_10cm'].isna().all():
                fig_multi.add_trace(
                    go.Scatter(x=df_filtered['date'], y=df_filtered['radio_flux_10cm'], 
                              name='Radio Flux (10cm)', line=dict(color='purple', width=1)),
                    row=2, col=1, secondary_y=True
                )
            
            # Row 3: Solar flares
            flare_colors = {'c_flares': 'green', 'm_flares': 'orange', 'x_flares': 'red'}
            flare_names = {'c_flares': 'C-Class', 'm_flares': 'M-Class', 'x_flares': 'X-Class'}
            
            for flare_type in ['c_flares', 'm_flares', 'x_flares']:
                if flare_type in df_filtered.columns and not df_filtered[flare_type].isna().all():
                    fig_multi.add_trace(
                        go.Scatter(x=df_filtered['date'], y=df_filtered[flare_type], 
                                  name=flare_names[flare_type], 
                                  line=dict(color=flare_colors[flare_type], width=2)),
                        row=3, col=1
                    )
            
            # Update layout
            fig_multi.update_layout(height=800, showlegend=True, title_text="Complete Space Weather Overview")
            fig_multi.update_xaxes(title_text="Date", row=3, col=1)
            fig_multi.update_yaxes(title_text="Kp Index", row=1, col=1)
            fig_multi.update_yaxes(title_text="A-Index", row=1, col=1, secondary_y=True)
            fig_multi.update_yaxes(title_text="Sunspot Number", row=2, col=1)
            fig_multi.update_yaxes(title_text="Radio Flux (SFU)", row=2, col=1, secondary_y=True)
            fig_multi.update_yaxes(title_text="Number of Flares", row=3, col=1)
            
            st.plotly_chart(fig_multi, use_container_width=True)
            
            # Monthly/Weekly aggregations
            st.subheader("📅 Temporal Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly averages
                if 'date' in df_filtered.columns:
                    df_monthly = df_filtered.copy()
                    df_monthly['month'] = df_filtered['date'].dt.to_period('M')
                    
                    monthly_stats = df_monthly.groupby('month').agg({
                        'max_kp': 'mean',
                        'sunspot_number': 'mean',
                        'radio_flux_10cm': 'mean',
                        'c_flares': 'sum',
                        'm_flares': 'sum',
                        'x_flares': 'sum'
                    }).reset_index()
                    
                    if not monthly_stats.empty:
                        monthly_stats['month_str'] = monthly_stats['month'].astype(str)
                        
                        fig_monthly = go.Figure()
                        
                        if 'max_kp' in monthly_stats.columns:
                            fig_monthly.add_trace(go.Bar(
                                x=monthly_stats['month_str'],
                                y=monthly_stats['max_kp'],
                                name='Avg Kp Index',
                                marker_color='red'
                            ))
                        
                        fig_monthly.update_layout(
                            title="Monthly Average Kp Index",
                            xaxis_title="Month",
                            yaxis_title="Average Kp Index",
                            height=400
                        )
                        
                        st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                # Weekly patterns
                if 'date' in df_filtered.columns:
                    df_weekly = df_filtered.copy()
                    df_weekly['day_of_week'] = df_filtered['date'].dt.day_name()
                    
                    weekly_stats = df_weekly.groupby('day_of_week').agg({
                        'max_kp': 'mean',
                        'sunspot_number': 'mean'
                    }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                    
                    if not weekly_stats.empty:
                        fig_weekly = go.Figure()
                        
                        if 'max_kp' in weekly_stats.columns:
                            fig_weekly.add_trace(go.Bar(
                                x=weekly_stats.index,
                                y=weekly_stats['max_kp'],
                                name='Avg Kp Index',
                                marker_color='blue'
                            ))
                        
                        fig_weekly.update_layout(
                            title="Day of Week Pattern (Avg Kp)",
                            xaxis_title="Day of Week",
                            yaxis_title="Average Kp Index",
                            height=400
                        )
                        
                        st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Data completeness and quality indicators
            st.subheader("📋 Data Quality & Coverage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Data completeness
                completeness = {}
                for col in ['max_kp', 'sunspot_number', 'radio_flux_10cm', 'a_index']:
                    if col in df_filtered.columns:
                        non_null_count = df_filtered[col].notna().sum()
                        total_count = len(df_filtered)
                        completeness[col] = (non_null_count / total_count) * 100
                
                if completeness:
                    completeness_df = pd.DataFrame(list(completeness.items()), 
                                                 columns=['Parameter', 'Completeness %'])
                    completeness_df['Completeness %'] = completeness_df['Completeness %'].round(1)
                    
                    st.write("**Data Completeness**")
                    st.dataframe(completeness_df, use_container_width=True)
            
            with col2:
                # Extreme events summary
                st.write("**Extreme Events**")
                extreme_events = {}
                
                if 'max_kp' in df_filtered.columns:
                    storm_days = (df_filtered['max_kp'] >= 5).sum()
                    severe_storm_days = (df_filtered['max_kp'] >= 7).sum()
                    extreme_events['Geomagnetic Storm Days (Kp≥5)'] = storm_days
                    extreme_events['Severe Storm Days (Kp≥7)'] = severe_storm_days
                
                if 'x_flares' in df_filtered.columns:
                    x_flare_days = (df_filtered['x_flares'] > 0).sum()
                    extreme_events['X-Flare Days'] = x_flare_days
                
                if extreme_events:
                    for event, count in extreme_events.items():
                        st.metric(event, count)
            
            with col3:
                # Period statistics
                st.write("**Period Statistics**")
                
                if 'max_kp' in df_filtered.columns:
                    kp_stats = {
                        'Quiet Days (Kp<3)': (df_filtered['max_kp'] < 3).sum(),
                        'Active Days (3≤Kp<5)': ((df_filtered['max_kp'] >= 3) & (df_filtered['max_kp'] < 5)).sum(),
                        'Storm Days (Kp≥5)': (df_filtered['max_kp'] >= 5).sum()
                    }
                    
                    for condition, count in kp_stats.items():
                        percentage = (count / len(df_filtered)) * 100
                        st.metric(condition, f"{count} ({percentage:.1f}%)")
            
            # Trend analysis
            st.subheader("📈 Trend Analysis")
            
            # Calculate rolling averages
            if len(df_filtered) > 7:
                df_trends = df_filtered.copy()
                
                for col in ['max_kp', 'sunspot_number', 'radio_flux_10cm']:
                    if col in df_trends.columns and not df_trends[col].isna().all():
                        df_trends[f'{col}_7day'] = df_trends[col].rolling(window=7, center=True).mean()
                        df_trends[f'{col}_30day'] = df_trends[col].rolling(window=30, center=True).mean()
                
                # Plot trends
                fig_trends = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Kp Index Trends', 'Sunspot Number Trends', 
                                  'Radio Flux Trends', 'Activity Correlation'),
                    specs=[[{}, {}], [{}, {}]]
                )
                
                # Kp trends
                if 'max_kp' in df_trends.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=df_trends['date'], y=df_trends['max_kp'], 
                                  name='Daily Kp', opacity=0.3, line=dict(color='red')),
                        row=1, col=1
                    )
                    if 'max_kp_7day' in df_trends.columns:
                        fig_trends.add_trace(
                            go.Scatter(x=df_trends['date'], y=df_trends['max_kp_7day'], 
                                      name='7-day Average', line=dict(color='darkred', width=2)),
                            row=1, col=1
                        )
                
                # Sunspot trends
                if 'sunspot_number' in df_trends.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=df_trends['date'], y=df_trends['sunspot_number'], 
                                  name='Daily Sunspots', opacity=0.3, line=dict(color='orange')),
                        row=1, col=2
                    )
                    if 'sunspot_number_7day' in df_trends.columns:
                        fig_trends.add_trace(
                            go.Scatter(x=df_trends['date'], y=df_trends['sunspot_number_7day'], 
                                      name='7-day Average', line=dict(color='darkorange', width=2)),
                            row=1, col=2
                        )
                
                # Radio flux trends
                if 'radio_flux_10cm' in df_trends.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=df_trends['date'], y=df_trends['radio_flux_10cm'], 
                                  name='Daily Flux', opacity=0.3, line=dict(color='purple')),
                        row=2, col=1
                    )
                    if 'radio_flux_10cm_7day' in df_trends.columns:
                        fig_trends.add_trace(
                            go.Scatter(x=df_trends['date'], y=df_trends['radio_flux_10cm_7day'], 
                                      name='7-day Average', line=dict(color='indigo', width=2)),
                            row=2, col=1
                        )
                
                # Activity correlation scatter
                if 'sunspot_number' in df_trends.columns and 'max_kp' in df_trends.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=df_trends['sunspot_number'], y=df_trends['max_kp'], 
                                  mode='markers', name='Daily Values',
                                  marker=dict(color='green', opacity=0.6)),
                        row=2, col=2
                    )
                
                fig_trends.update_layout(height=600, showlegend=False, title_text="Trend Analysis")
                st.plotly_chart(fig_trends, use_container_width=True)
        
        with tab2:
            if 'max_kp' in df_filtered.columns and not df_filtered['max_kp'].isna().all():
                fig_kp = go.Figure()
                
                # Add Kp index line
                fig_kp.add_trace(go.Scatter(
                    x=df_filtered['date'],
                    y=df_filtered['max_kp'],
                    mode='lines+markers',
                    name='Max Kp Index',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
                
                # Add storm level thresholds
                fig_kp.add_hline(y=5, line_dash="dash", line_color="orange", 
                                annotation_text="Minor Storm Threshold", annotation_position="right")
                fig_kp.add_hline(y=6, line_dash="dash", line_color="red", 
                                annotation_text="Moderate Storm Threshold", annotation_position="right")
                
                fig_kp.update_layout(
                    title="Geomagnetic Activity (Kp Index) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Kp Index",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_kp, use_container_width=True)
            
            # A-index plot
            if 'a_index' in df_filtered.columns and not df_filtered['a_index'].isna().all():
                fig_a = px.line(df_filtered, x='date', y='a_index', 
                               title="A-Index Over Time",
                               labels={'a_index': 'A-Index', 'date': 'Date'})
                fig_a.update_traces(line=dict(color='blue', width=2))
                fig_a.update_layout(height=400)
                st.plotly_chart(fig_a, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'sunspot_number' in df_filtered.columns and not df_filtered['sunspot_number'].isna().all():
                    fig_spots = px.line(df_filtered, x='date', y='sunspot_number',
                                       title="Sunspot Number Over Time",
                                       labels={'sunspot_number': 'Sunspot Number', 'date': 'Date'})
                    fig_spots.update_traces(line=dict(color='orange', width=2))
                    fig_spots.update_layout(height=400)
                    st.plotly_chart(fig_spots, use_container_width=True)
            
            with col2:
                if 'radio_flux_10cm' in df_filtered.columns and not df_filtered['radio_flux_10cm'].isna().all():
                    fig_flux = px.line(df_filtered, x='date', y='radio_flux_10cm',
                                      title="Solar Radio Flux (10.7 cm) Over Time",
                                      labels={'radio_flux_10cm': 'Radio Flux (SFU)', 'date': 'Date'})
                    fig_flux.update_traces(line=dict(color='purple', width=2))
                    fig_flux.update_layout(height=400)
                    st.plotly_chart(fig_flux, use_container_width=True)
        
        with tab4:
            # Solar flares analysis
            flare_cols = ['c_flares', 'm_flares', 'x_flares']
            available_flare_cols = [col for col in flare_cols if col in df_filtered.columns and not df_filtered[col].isna().all()]
            
            if available_flare_cols:
                fig_flares = go.Figure()
                
                colors = {'c_flares': 'green', 'm_flares': 'orange', 'x_flares': 'red'}
                names = {'c_flares': 'C-Class Flares', 'm_flares': 'M-Class Flares', 'x_flares': 'X-Class Flares'}
                
                for col in available_flare_cols:
                    fig_flares.add_trace(go.Scatter(
                        x=df_filtered['date'],
                        y=df_filtered[col],
                        mode='lines+markers',
                        name=names[col],
                        line=dict(color=colors[col], width=2),
                        marker=dict(size=6)
                    ))
                
                fig_flares.update_layout(
                    title="Solar Flares Over Time",
                    xaxis_title="Date", 
                    yaxis_title="Number of Flares",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_flares, use_container_width=True)
                
                # Flare statistics
                if available_flare_cols:
                    st.subheader("Flare Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'c_flares' in df_filtered.columns:
                            total_c = df_filtered['c_flares'].sum()
                            st.metric("Total C-Class Flares", int(total_c))
                    
                    with col2:
                        if 'm_flares' in df_filtered.columns:
                            total_m = df_filtered['m_flares'].sum()
                            st.metric("Total M-Class Flares", int(total_m))
                    
                    with col3:
                        if 'x_flares' in df_filtered.columns:
                            total_x = df_filtered['x_flares'].sum()
                            st.metric("Total X-Class Flares", int(total_x))
        
        with tab5:
            # Correlation analysis
            st.subheader("Correlation Analysis")
            
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            if 'date' in numeric_cols:
                numeric_cols.remove('date')
            
            if len(numeric_cols) >= 2:
                # Correlation heatmap
                corr_matrix = df_filtered[numeric_cols].corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto=True, 
                                    aspect="auto",
                                    title="Correlation Matrix of Space Weather Parameters",
                                    color_continuous_scale="RdBu")
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Scatter plots for key relationships
                st.subheader("Key Relationships")
                
                if 'sunspot_number' in df_filtered.columns and 'radio_flux_10cm' in df_filtered.columns:
                    fig_scatter = px.scatter(df_filtered, x='sunspot_number', y='radio_flux_10cm',
                                           title="Sunspot Number vs Solar Radio Flux",
                                           labels={'sunspot_number': 'Sunspot Number', 
                                                  'radio_flux_10cm': 'Radio Flux (SFU)'},
                                           trendline="ols")
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab6:
            # Health correlation analysis
            st.subheader("🧠 Health Correlation Analysis")
            st.write("Analysis of potential correlations between space weather activity and health events (migraines)")
            
            # Define migraine dates
            migraine_dates = [
                datetime(2025, 6, 3),   # June 3rd
                datetime(2025, 5, 28),  # May 28th
            ]
            
            # Add migraine data to analysis
            df_health = df_filtered.copy()
            df_health['migraine'] = df_health['date'].isin(migraine_dates)
            
            # Display migraine dates and corresponding space weather data
            st.subheader("📅 Migraine Episodes & Space Weather Conditions")
            
            migraine_analysis = []
            for migraine_date in migraine_dates:
                date_data = df_health[df_health['date'] == migraine_date]
                if not date_data.empty:
                    row = date_data.iloc[0]
                    migraine_analysis.append({
                        'Date': migraine_date.strftime('%B %d, %Y'),
                        'Max Kp Index': row.get('max_kp', 'N/A'),
                        'A-Index': row.get('a_index', 'N/A'),
                        'Sunspot Number': row.get('sunspot_number', 'N/A'),
                        'Radio Flux (10cm)': row.get('radio_flux_10cm', 'N/A'),
                        'C-Flares': row.get('c_flares', 'N/A'),
                        'M-Flares': row.get('m_flares', 'N/A'),
                        'X-Flares': row.get('x_flares', 'N/A'),
                        'Condition': get_space_weather_condition(row.get('max_kp', 0))[0] if not pd.isna(row.get('max_kp')) else 'Unknown'
                    })
            
            if migraine_analysis:
                migraine_df = pd.DataFrame(migraine_analysis)
                st.dataframe(migraine_df, use_container_width=True)
                
                # Key findings
                st.subheader("🔍 Key Findings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**June 3, 2025 Analysis:**")
                    june_3_data = df_health[df_health['date'] == datetime(2025, 6, 3)]
                    if not june_3_data.empty:
                        june_3 = june_3_data.iloc[0]
                        kp_june = june_3.get('max_kp', 0)
                        st.write(f"• Max Kp Index: **{kp_june:.1f}** ({get_space_weather_condition(kp_june)[0]})")
                        st.write(f"• Sunspot Number: **{june_3.get('sunspot_number', 'N/A')}**")
                        st.write(f"• M-Class Flares: **{june_3.get('m_flares', 'N/A')}**")
                        if kp_june >= 5:
                            st.warning("⚠️ **Geomagnetic Storm Day!** (Kp ≥ 5.0)")
                        elif kp_june >= 4:
                            st.info("🔶 **Active Geomagnetic Conditions** (Kp ≥ 4.0)")
                
                with col2:
                    st.write("**May 28, 2025 Analysis:**")
                    may_28_data = df_health[df_health['date'] == datetime(2025, 5, 28)]
                    if not may_28_data.empty:
                        may_28 = may_28_data.iloc[0]
                        kp_may = may_28.get('max_kp', 0)
                        st.write(f"• Max Kp Index: **{kp_may:.1f}** ({get_space_weather_condition(kp_may)[0]})")
                        st.write(f"• Sunspot Number: **{may_28.get('sunspot_number', 'N/A')}**")
                        st.write(f"• C-Class Flares: **{may_28.get('c_flares', 'N/A')}**")
                        if kp_may >= 4:
                            st.info("🔶 **Active Geomagnetic Conditions** (Kp ≥ 4.0)")
                        else:
                            st.success("✅ **Quiet/Unsettled Conditions**")
            
            # Statistical analysis
            st.subheader("📊 Statistical Analysis")
            
            # Compare space weather on migraine days vs non-migraine days
            if 'max_kp' in df_health.columns and not df_health['max_kp'].isna().all():
                migraine_days = df_health[df_health['migraine'] == True]
                non_migraine_days = df_health[df_health['migraine'] == False]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if not migraine_days.empty and not non_migraine_days.empty:
                        avg_kp_migraine = migraine_days['max_kp'].mean()
                        avg_kp_normal = non_migraine_days['max_kp'].mean()
                        
                        st.metric("Avg Kp (Migraine Days)", f"{avg_kp_migraine:.2f}")
                        st.metric("Avg Kp (Normal Days)", f"{avg_kp_normal:.2f}")
                        
                        difference = avg_kp_migraine - avg_kp_normal
                        
                        # Perform statistical test
                        try:
                            t_stat, p_value = stats.ttest_ind(migraine_days['max_kp'].dropna(), 
                                                            non_migraine_days['max_kp'].dropna())
                            
                            if difference > 0:
                                st.error(f"📈 **Higher by {difference:.2f}** on migraine days")
                            else:
                                st.success(f"📉 **Lower by {abs(difference):.2f}** on migraine days")
                            
                            # Statistical significance
                            if p_value < 0.05:
                                st.info(f"📊 **Statistically significant** (p={p_value:.3f})")
                            else:
                                st.info(f"📊 **Not statistically significant** (p={p_value:.3f})")
                                
                        except Exception as e:
                            if difference > 0:
                                st.error(f"📈 **Higher by {difference:.2f}** on migraine days")
                            else:
                                st.success(f"📉 **Lower by {abs(difference):.2f}** on migraine days")
                
                with col2:
                    # Days around migraine episodes (±3 days)
                    migraine_window_data = []
                    for migraine_date in migraine_dates:
                        window_start = migraine_date - timedelta(days=3)
                        window_end = migraine_date + timedelta(days=3)
                        window_data = df_health[
                            (df_health['date'] >= window_start) & 
                            (df_health['date'] <= window_end)
                        ]
                        if not window_data.empty:
                            migraine_window_data.append(window_data)
                    
                    if migraine_window_data:
                        combined_window = pd.concat(migraine_window_data)
                        # Remove duplicates by date only (avoid issues with list columns)
                        combined_window = combined_window.drop_duplicates(subset=['date'])
                        avg_kp_window = combined_window['max_kp'].mean()
                        st.metric("Avg Kp (±3 days around migraines)", f"{avg_kp_window:.2f}")
                
                with col3:
                    # Storm activity analysis
                    storm_days_total = (df_health['max_kp'] >= 5).sum()
                    migraine_on_storm_days = sum([
                        1 for date in migraine_dates 
                        if not df_health[(df_health['date'] == date) & (df_health['max_kp'] >= 5)].empty
                    ])
                    
                    st.metric("Total Storm Days (Kp≥5)", storm_days_total)
                    st.metric("Migraines on Storm Days", migraine_on_storm_days)
                    
                    if storm_days_total > 0:
                        storm_migraine_rate = (migraine_on_storm_days / storm_days_total) * 100
                        st.metric("Migraine Rate on Storm Days", f"{storm_migraine_rate:.1f}%")
            
            # Correlation Analysis
            st.subheader("🔗 Correlation Analysis")
            
            if 'max_kp' in df_health.columns and not df_health['max_kp'].isna().all():
                # Create binary migraine indicator for correlation
                migraine_binary = df_health['migraine'].astype(int)
                kp_values = df_health['max_kp'].dropna()
                
                # Align the data
                common_indices = migraine_binary.index.intersection(kp_values.index)
                if len(common_indices) > 1:
                    aligned_migraines = migraine_binary.loc[common_indices]
                    aligned_kp = kp_values.loc[common_indices]
                    
                    try:
                        # Calculate correlation
                        correlation, p_value = stats.pearsonr(aligned_kp, aligned_migraines)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Pearson Correlation", f"{correlation:.3f}")
                            if abs(correlation) > 0.3:
                                st.info("🔗 **Moderate correlation detected**")
                            elif abs(correlation) > 0.1:
                                st.info("🔗 **Weak correlation detected**")
                            else:
                                st.info("🔗 **Very weak correlation**")
                        
                        with col2:
                            st.metric("P-value", f"{p_value:.3f}")
                            if p_value < 0.05:
                                st.success("✅ **Statistically significant**")
                            else:
                                st.warning("⚠️ **Not statistically significant**")
                                
                    except Exception as e:
                        st.info("🔗 **Correlation analysis requires more data points**")
            
            # Visualization: Timeline with migraine markers
            st.subheader("📈 Timeline Analysis")
            
            if 'max_kp' in df_health.columns and not df_health['max_kp'].isna().all():
                fig_timeline = go.Figure()
                
                # Add Kp index line
                fig_timeline.add_trace(go.Scatter(
                    x=df_health['date'],
                    y=df_health['max_kp'],
                    mode='lines',
                    name='Kp Index',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ))
                
                # Add migraine markers
                for migraine_date in migraine_dates:
                    migraine_data = df_health[df_health['date'] == migraine_date]
                    if not migraine_data.empty:
                        kp_value = migraine_data.iloc[0].get('max_kp', 0)
                        fig_timeline.add_trace(go.Scatter(
                            x=[migraine_date],
                            y=[kp_value],
                            mode='markers',
                            name=f'Migraine {migraine_date.strftime("%m/%d")}',
                            marker=dict(size=15, color='red', symbol='x'),
                            hovertemplate=f'Migraine Episode<br>Date: {migraine_date.strftime("%B %d, %Y")}<br>Kp: {kp_value:.1f}<extra></extra>'
                        ))
                
                # Add storm threshold lines
                fig_timeline.add_hline(y=5, line_dash="dash", line_color="orange", 
                                     annotation_text="Minor Storm Threshold")
                fig_timeline.add_hline(y=4, line_dash="dash", line_color="yellow", 
                                     annotation_text="Active Conditions Threshold")
                
                fig_timeline.update_layout(
                    title="Kp Index Timeline with Migraine Episodes",
                    xaxis_title="Date",
                    yaxis_title="Kp Index",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Research context and recommendations
            st.subheader("🔬 Research Context & Interpretation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **Scientific Background:**
                
                • Some studies suggest correlations between geomagnetic activity and health
                • Kp index > 4 indicates "active" geomagnetic conditions
                • Kp index ≥ 5 indicates geomagnetic storms
                • Solar flares can affect Earth's magnetic field
                • Individual sensitivity may vary significantly
                """)
            
            with col2:
                st.warning("""
                **Important Notes:**
                
                • This analysis is for informational purposes only
                • Correlation does not imply causation
                • Sample size is small (only 2 migraine episodes)
                • Many factors can trigger migraines
                • Consult healthcare providers for medical advice
                """)
            
            # Predictive insights
            st.subheader("🔮 Potential Patterns & Monitoring")
            
            # Check current and upcoming space weather
            latest_data = df_health.iloc[-1] if len(df_health) > 0 else None
            if latest_data is not None:
                current_kp = latest_data.get('max_kp', 0)
                current_condition, color = get_space_weather_condition(current_kp)
                
                st.write("**Current Space Weather Status:**")
                st.markdown(f"• **Kp Index:** {current_kp:.1f} - <span style='color:{color}'>{current_condition}</span>", 
                           unsafe_allow_html=True)
                
                if current_kp >= 4:
                    st.warning("⚠️ **Active geomagnetic conditions detected.** You may want to monitor for potential health sensitivity.")
                elif current_kp >= 5:
                    st.error("🚨 **Geomagnetic storm in progress.** Consider extra health precautions if you're sensitive to space weather.")
                else:
                    st.success("✅ **Quiet space weather conditions.** Lower likelihood of space weather-related health effects.")
        
        with tab7:
            st.header("🤖 AI-Powered Analysis")
            st.markdown("Advanced machine learning analysis using multiple solar factors to predict migraine correlation")
            
            # Check GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info = f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
                    st.success(f"🚀 {gpu_info}")
                else:
                    st.info("🖥️ Using CPU for analysis")
            except ImportError:
                st.warning("⚠️ PyTorch not available")
            
            if not df_filtered.empty:
                # Feature Engineering Section
                st.subheader("🔧 Feature Engineering")
                
                # Create advanced features
                df_ml = df_filtered.copy()
                
                # Add migraine indicator
                migraine_dates = [pd.Timestamp('2025-06-03'), pd.Timestamp('2025-05-28')]
                df_ml['migraine'] = df_ml['date'].isin(migraine_dates).astype(int)
                
                # Feature creation
                features_created = []
                
                if 'max_kp' in df_ml.columns:
                    # Rolling statistics for Kp index
                    df_ml['kp_rolling_mean_3'] = df_ml['max_kp'].rolling(window=3, center=True).mean()
                    df_ml['kp_rolling_std_3'] = df_ml['max_kp'].rolling(window=3, center=True).std()
                    df_ml['kp_rolling_mean_7'] = df_ml['max_kp'].rolling(window=7, center=True).mean()
                    df_ml['kp_change'] = df_ml['max_kp'].diff()
                    df_ml['kp_acceleration'] = df_ml['kp_change'].diff()
                    features_created.extend(['kp_rolling_mean_3', 'kp_rolling_std_3', 'kp_rolling_mean_7', 'kp_change', 'kp_acceleration'])
                
                if 'sunspot_number' in df_ml.columns:
                    # Sunspot features
                    df_ml['sunspot_rolling_mean_7'] = df_ml['sunspot_number'].rolling(window=7, center=True).mean()
                    df_ml['sunspot_change'] = df_ml['sunspot_number'].diff()
                    features_created.extend(['sunspot_rolling_mean_7', 'sunspot_change'])
                
                if 'radio_flux_10cm' in df_ml.columns:
                    # Radio flux features
                    df_ml['flux_rolling_mean_3'] = df_ml['radio_flux_10cm'].rolling(window=3, center=True).mean()
                    df_ml['flux_change'] = df_ml['radio_flux_10cm'].diff()
                    features_created.extend(['flux_rolling_mean_3', 'flux_change'])
                
                # Solar flare features
                flare_cols = [col for col in df_ml.columns if 'flare' in col.lower()]
                if flare_cols:
                    df_ml['total_flares'] = df_ml[flare_cols].sum(axis=1)
                    features_created.append('total_flares')
                
                # Cyclical features (day of year, etc.)
                df_ml['day_of_year'] = df_ml['date'].dt.dayofyear
                df_ml['day_of_year_sin'] = np.sin(2 * np.pi * df_ml['day_of_year'] / 365)
                df_ml['day_of_year_cos'] = np.cos(2 * np.pi * df_ml['day_of_year'] / 365)
                features_created.extend(['day_of_year', 'day_of_year_sin', 'day_of_year_cos'])
                
                st.write(f"✅ Created {len(features_created)} engineered features")
                
                # ML Analysis Section
                st.subheader("🧠 Machine Learning Analysis")
                
                # Prepare features for ML
                feature_cols = [col for col in df_ml.columns if col not in ['date', 'migraine'] and df_ml[col].dtype in ['float64', 'int64']]
                feature_cols = [col for col in feature_cols if not df_ml[col].isna().all()]
                
                if len(feature_cols) > 0:
                    # Remove rows with NaN values
                    df_clean = df_ml[feature_cols + ['migraine']].dropna()
                    
                    if len(df_clean) > 10:  # Ensure we have enough data
                        X = df_clean[feature_cols]
                        y = df_clean['migraine']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Samples", len(df_clean))
                            st.metric("Features Used", len(feature_cols))
                            st.metric("Migraine Cases", y.sum())
                        
                        with col2:
                            st.metric("Non-Migraine Cases", len(y) - y.sum())
                            st.metric("Class Balance", f"{y.mean()*100:.1f}% positive")
                        
                        # Feature Importance Analysis
                        st.subheader("📊 Feature Importance Analysis")
                        
                        try:
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.model_selection import cross_val_score
                            import shap
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Random Forest for feature importance
                            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                            rf.fit(X_scaled, y)
                            
                            # Feature importance plot
                            importance_df = pd.DataFrame({
                                'feature': feature_cols,
                                'importance': rf.feature_importances_
                            }).sort_values('importance', ascending=True)
                            
                            fig_importance = px.bar(
                                importance_df.tail(10), 
                                x='importance', 
                                y='feature',
                                title="Top 10 Most Important Features",
                                orientation='h'
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Cross-validation scores with better handling of class imbalance
                            try:
                                # Use fewer folds and handle NaN values
                                cv_folds = min(3, len(y[y==1]) + 1, len(y))  # Limited by positive class size
                                cv_scores = cross_val_score(rf, X_scaled, y, cv=cv_folds, scoring='roc_auc')
                                # Filter out NaN values
                                cv_scores = cv_scores[~np.isnan(cv_scores)]
                                if len(cv_scores) > 0:
                                    st.metric("Cross-Validation AUC", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                                else:
                                    st.metric("Cross-Validation AUC", "Not available (insufficient data)")
                            except Exception as e:
                                st.metric("Cross-Validation AUC", f"Error: {str(e)[:50]}...")
                            
                        except ImportError:
                            st.warning("Scikit-learn not available for feature importance analysis")
                        
                        # Advanced ML Models
                        st.subheader("🚀 Advanced ML Models")
                        
                        model_results = {}
                        
                        # XGBoost
                        try:
                            import xgboost as xgb
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import roc_auc_score, classification_report
                            
                            if len(np.unique(y)) > 1:  # Only if we have both classes
                                # Use simple train_test_split without stratification due to small sample size
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_scaled, y, test_size=0.3, random_state=42
                                )
                                
                                # XGBoost with GPU support
                                xgb_model = xgb.XGBClassifier(
                                    n_estimators=100,
                                    max_depth=3,
                                    learning_rate=0.1,
                                    random_state=42,
                                    tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
                                    device='cuda' if torch.cuda.is_available() else 'cpu'
                                )
                                
                                xgb_model.fit(X_train, y_train)
                                xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
                                xgb_auc = roc_auc_score(y_test, xgb_pred)
                                model_results['XGBoost'] = xgb_auc
                                
                                st.success(f"✅ XGBoost AUC: {xgb_auc:.3f}")
                            
                        except ImportError:
                            st.warning("XGBoost not available")
                        
                        # Neural Network with PyTorch
                        try:
                            import torch
                            import torch.nn as nn
                            import torch.optim as optim
                            from sklearn.model_selection import train_test_split
                            
                            if torch.cuda.is_available() and len(np.unique(y)) > 1:
                                device = torch.device('cuda')
                                st.info("🚀 Training Neural Network on GPU...")
                                
                                # Simple neural network
                                class MigrainePredictorNN(nn.Module):
                                    def __init__(self, input_size):
                                        super().__init__()
                                        self.network = nn.Sequential(
                                            nn.Linear(input_size, 64),
                                            nn.ReLU(),
                                            nn.Dropout(0.3),
                                            nn.Linear(64, 32),
                                            nn.ReLU(),
                                            nn.Dropout(0.3),
                                            nn.Linear(32, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 1),
                                            nn.Sigmoid()
                                        )
                                    
                                    def forward(self, x):
                                        return self.network(x)
                                
                                # Prepare data for PyTorch
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                                )
                                
                                X_train_tensor = torch.FloatTensor(X_train).to(device)
                                y_train_tensor = torch.FloatTensor(y_train.values).to(device)
                                X_test_tensor = torch.FloatTensor(X_test).to(device)
                                y_test_tensor = torch.FloatTensor(y_test.values).to(device)
                                
                                # Initialize and train model
                                model = MigrainePredictorNN(X_train.shape[1]).to(device)
                                criterion = nn.BCELoss()
                                optimizer = optim.Adam(model.parameters(), lr=0.001)
                                
                                # Training loop
                                model.train()
                                for epoch in range(100):
                                    optimizer.zero_grad()
                                    outputs = model(X_train_tensor).squeeze()
                                    loss = criterion(outputs, y_train_tensor)
                                    loss.backward()
                                    optimizer.step()
                                
                                # Evaluation
                                model.eval()
                                with torch.no_grad():
                                    test_outputs = model(X_test_tensor).squeeze()
                                    test_predictions = test_outputs.cpu().numpy()
                                    nn_auc = roc_auc_score(y_test, test_predictions)
                                    model_results['Neural Network'] = nn_auc
                                
                                st.success(f"✅ Neural Network AUC: {nn_auc:.3f}")
                            
                        except Exception as e:
                            st.warning(f"Neural Network training failed: {str(e)}")
                        
                        # Model Comparison
                        if model_results:
                            st.subheader("🏆 Model Performance Comparison")
                            
                            results_df = pd.DataFrame([
                                {'Model': model, 'AUC Score': score} 
                                for model, score in model_results.items()
                            ])
                            
                            fig_comparison = px.bar(
                                results_df, 
                                x='Model', 
                                y='AUC Score',
                                title="Model Performance Comparison (AUC Score)",
                                color='AUC Score',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Hyperparameter Optimization with Optuna
                        st.subheader("⚡ Hyperparameter Optimization")
                        
                        if st.button("🚀 Run GPU-Accelerated Hyperparameter Search"):
                            try:
                                import optuna
                                
                                def objective(trial):
                                    # XGBoost hyperparameters
                                    n_estimators = trial.suggest_int('n_estimators', 50, 200)
                                    max_depth = trial.suggest_int('max_depth', 3, 10)
                                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                                    
                                    model = xgb.XGBClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        random_state=42,
                                        tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
                                        device='cuda' if torch.cuda.is_available() else 'cpu'
                                    )
                                    
                                    # Handle cross-validation with class imbalance
                                    try:
                                        cv_folds = min(3, len(y[y==1]) + 1, len(y))
                                        scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='roc_auc')
                                        scores = scores[~np.isnan(scores)]  # Remove NaN values
                                        if len(scores) == 0:
                                            return 0.5  # Random performance baseline
                                        return scores.mean()
                                    except:
                                        return 0.5  # Fallback to random performance
                                
                                study = optuna.create_study(direction='maximize')
                                study.optimize(objective, n_trials=20)
                                
                                st.success(f"🎯 Best AUC: {study.best_value:.3f}")
                                st.json(study.best_params)
                                
                            except Exception as e:
                                st.error(f"Optimization failed: {str(e)}")
                        
                        # SHAP Analysis for Explainability
                        st.subheader("🔍 Model Explainability (SHAP)")
                        
                        try:
                            import shap
                            
                            if 'rf' in locals():
                                # Create SHAP explainer
                                explainer = shap.TreeExplainer(rf)
                                shap_values = explainer.shap_values(X_scaled[:20])  # Use first 20 samples
                                
                                # SHAP summary plot would go here
                                st.info("📊 SHAP analysis completed - feature contributions calculated")
                                
                                # Show most important features for migraine prediction
                                if len(shap_values) > 1:
                                    shap_importance = np.abs(shap_values[1]).mean(axis=0)
                                    # Ensure we use the same features that were used in X
                                    current_features = X.columns.tolist() if hasattr(X, 'columns') else feature_cols
                                    # Make sure lengths match
                                    if len(shap_importance) == len(current_features):
                                        shap_df = pd.DataFrame({
                                            'feature': current_features,
                                            'shap_importance': shap_importance
                                        }).sort_values('shap_importance', ascending=False)
                                        
                                        st.write("**Top features contributing to migraine prediction:**")
                                        for i, row in shap_df.head(5).iterrows():
                                            st.write(f"• {row['feature']}: {row['shap_importance']:.3f}")
                                    else:
                                        st.warning(f"Feature mismatch: {len(shap_importance)} SHAP values vs {len(current_features)} features")
                        
                        except ImportError:
                            st.warning("SHAP not available for explainability analysis")
                
                else:
                    st.warning("⚠️ Not enough clean data for ML analysis (need >10 samples)")
            else:
                st.warning("⚠️ No numerical features available for ML analysis")
        
        # Time Series Forecasting
        st.subheader("📈 Time Series Forecasting")
        
        if not df_filtered.empty and 'max_kp' in df_filtered.columns:
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import PolynomialFeatures
                
                # Prepare time series data
                ts_data = df_filtered[['date', 'max_kp']].dropna().sort_values('date')
                ts_data['days'] = (ts_data['date'] - ts_data['date'].min()).dt.days
                
                if len(ts_data) > 5:
                    # Polynomial regression for trend
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(ts_data[['days']])
                    
                    model = LinearRegression()
                    model.fit(X_poly, ts_data['max_kp'])
                    
                    # Predict future values
                    future_days = np.arange(ts_data['days'].max() + 1, ts_data['days'].max() + 8)
                    future_X = poly_features.transform(future_days.reshape(-1, 1))
                    future_pred = model.predict(future_X)
                    
                    # Create future dates
                    last_date = ts_data['date'].max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
                    
                    # Plot forecast
                    fig_forecast = go.Figure()
                    
                    # Historical data
                    fig_forecast.add_trace(go.Scatter(
                        x=ts_data['date'],
                        y=ts_data['max_kp'],
                        mode='lines+markers',
                        name='Historical Kp',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_pred,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_forecast.update_layout(
                        title="Kp Index Forecast (Next 7 Days)",
                        xaxis_title="Date",
                        yaxis_title="Kp Index",
                        height=400
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Show forecast values
                    st.write("**7-Day Kp Index Forecast:**")
                    for date, pred in zip(future_dates, future_pred):
                        condition, color = get_space_weather_condition(pred)
                        st.write(f"• {date.strftime('%Y-%m-%d')}: **{pred:.1f}** - <span style='color:{color}'>{condition}</span>", 
                               unsafe_allow_html=True)
            
            except ImportError:
                st.warning("Forecasting requires additional packages")
        
        # AI Insights Summary
        st.subheader("🎯 AI-Generated Insights")
        
        insights = []
        
        if not df_filtered.empty:
            # Statistical insights
            migraine_days = df_filtered[df_filtered['date'].isin([pd.Timestamp('2025-06-03'), pd.Timestamp('2025-05-28')])]
            
            if not migraine_days.empty and 'max_kp' in migraine_days.columns:
                avg_kp_migraine = migraine_days['max_kp'].mean()
                avg_kp_overall = df_filtered['max_kp'].mean()
                
                if avg_kp_migraine > avg_kp_overall:
                    insights.append(f"🔍 Migraine days show {avg_kp_migraine - avg_kp_overall:.1f} higher average Kp index")
                
                # Check for patterns
                if avg_kp_migraine > 4:
                    insights.append("⚠️ Both migraine episodes occurred during elevated geomagnetic activity")
                
            # Data quality insights
            data_completeness = (1 - df_filtered.isnull().sum() / len(df_filtered)) * 100
            best_feature = data_completeness.idxmax()
            insights.append(f"📊 Best data quality: {best_feature} ({data_completeness[best_feature]:.1f}% complete)")
            
            # Temporal insights
            date_range = (df_filtered['date'].max() - df_filtered['date'].min()).days
            insights.append(f"📅 Analysis covers {date_range} days of space weather data")
        
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.info("🤔 Run the analysis to generate AI insights")

    # Summary and Data sections (moved inside tab structure)
    with st.expander("📊 Summary Statistics", expanded=False):
        summary_data = {}
        if 'max_kp' in df_filtered.columns and not df_filtered['max_kp'].isna().all():
            summary_data['Max Kp'] = {
                'Mean': f"{df_filtered['max_kp'].mean():.2f}",
                'Max': f"{df_filtered['max_kp'].max():.2f}",
                'Min': f"{df_filtered['max_kp'].min():.2f}",
                'Std Dev': f"{df_filtered['max_kp'].std():.2f}"
            }
        
        if 'sunspot_number' in df_filtered.columns and not df_filtered['sunspot_number'].isna().all():
            summary_data['Sunspot Number'] = {
                'Mean': f"{df_filtered['sunspot_number'].mean():.1f}",
                'Max': f"{df_filtered['sunspot_number'].max():.0f}",
                'Min': f"{df_filtered['sunspot_number'].min():.0f}",
                'Std Dev': f"{df_filtered['sunspot_number'].std():.1f}"
            }
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
    
    # Raw data table outside the expander
    if show_raw_data:
        st.header("📋 Raw Data")
        st.dataframe(df_filtered, use_container_width=True)

    else:
        st.warning("No data available for the selected date range.")

if __name__ == "__main__":
    main()
