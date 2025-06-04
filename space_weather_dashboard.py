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
            st.header("🤖 Advanced AI Analysis")
            st.markdown("**Multi-factor machine learning analysis to identify complex patterns between solar activity and health events**")
            
            # AI Analysis requires the imports at the top
            try:
                import sklearn
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import classification_report, confusion_matrix
                import xgboost as xgb
                import lightgbm as lgb
                
                # Advanced AI analysis implementation
                ai_analysis_available = True
            except ImportError as e:
                st.warning(f"⚠️ AI libraries not available in deployment environment: {str(e)}")
                st.info("💡 AI analysis is available when running locally with GPU acceleration")
                ai_analysis_available = False
            
            if ai_analysis_available:
                # Feature engineering for AI models
                st.subheader("🔧 Feature Engineering")
                
                # Create comprehensive feature set
                ai_features = df_filtered.copy()
                
                # Add time-based features
                ai_features['day_of_year'] = ai_features['date'].dt.dayofyear
                ai_features['month'] = ai_features['date'].dt.month
                ai_features['week_of_year'] = ai_features['date'].dt.isocalendar().week
                
                # Rolling statistics (capture temporal patterns)
                window_sizes = [3, 7, 14]
                for window in window_sizes:
                    if 'max_kp' in ai_features.columns:
                        ai_features[f'kp_rolling_mean_{window}d'] = ai_features['max_kp'].rolling(window=window).mean()
                        ai_features[f'kp_rolling_std_{window}d'] = ai_features['max_kp'].rolling(window=window).std()
                        ai_features[f'kp_rolling_max_{window}d'] = ai_features['max_kp'].rolling(window=window).max()
                    
                    if 'sunspot_number' in ai_features.columns:
                        ai_features[f'sunspot_rolling_mean_{window}d'] = ai_features['sunspot_number'].rolling(window=window).mean()
                        ai_features[f'sunspot_rolling_std_{window}d'] = ai_features['sunspot_number'].rolling(window=window).std()
                
                # Interaction features
                if 'max_kp' in ai_features.columns and 'sunspot_number' in ai_features.columns:
                    ai_features['kp_sunspot_interaction'] = ai_features['max_kp'] * ai_features['sunspot_number']
                    ai_features['kp_sunspot_ratio'] = ai_features['max_kp'] / (ai_features['sunspot_number'] + 1)
                
                # Storm intensity features
                if 'max_kp' in ai_features.columns:
                    ai_features['storm_intensity'] = pd.cut(ai_features['max_kp'], 
                                                          bins=[0, 2, 4, 5, 6, 7, 10], 
                                                          labels=[0, 1, 2, 3, 4, 5]).astype(float)
                
                # Display feature summary
                feature_cols = [col for col in ai_features.columns if col not in ['date', 'migraine']]
                st.write(f"**Generated {len(feature_cols)} features** including temporal patterns, rolling statistics, and interaction terms")
                
                # Advanced ML Analysis
                st.subheader("🧠 Multi-Algorithm Analysis")
                
                # Prepare data for ML
                if 'migraine' in ai_features.columns:
                    # Remove rows with NaN values
                    ml_data = ai_features.dropna()
                    
                    if len(ml_data) > 10:  # Need sufficient data
                        X = ml_data[feature_cols].select_dtypes(include=[np.number])
                        y = ml_data['migraine'].astype(int)
                        
                        # Handle case where we have very few positive samples
                        positive_samples = y.sum()
                        total_samples = len(y)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", total_samples)
                        with col2:
                            st.metric("Migraine Episodes", positive_samples)
                        with col3:
                            st.metric("Class Balance", f"{positive_samples/total_samples:.1%}")
                        
                        if positive_samples > 0 and total_samples > positive_samples:
                            st.subheader("🎯 Predictive Models")
                            
                            # Multiple algorithms comparison
                            algorithms = {
                                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
                            }
                            
                            # Cross-validation scores
                            cv_results = {}
                            feature_importance = {}
                            
                            for name, model in algorithms.items():
                                try:
                                    # Cross-validation
                                    cv_scores = cross_val_score(model, X, y, cv=min(5, total_samples//2), scoring='roc_auc')
                                    cv_results[name] = {
                                        'Mean CV Score': f"{cv_scores.mean():.3f}",
                                        'Std CV Score': f"{cv_scores.std():.3f}"
                                    }
                                    
                                    # Fit model for feature importance
                                    model.fit(X, y)
                                    if hasattr(model, 'feature_importances_'):
                                        importance = pd.DataFrame({
                                            'feature': X.columns,
                                            'importance': model.feature_importances_
                                        }).sort_values('importance', ascending=False)
                                        feature_importance[name] = importance.head(10)
                                        
                                except Exception as e:
                                    cv_results[name] = {'Error': str(e)}
                            
                            # Display results
                            if cv_results:
                                st.write("**Cross-Validation Performance:**")
                                results_df = pd.DataFrame(cv_results).T
                                st.dataframe(results_df, use_container_width=True)
                            
                            # Feature importance analysis
                            if feature_importance:
                                st.subheader("🔍 Feature Importance Analysis")
                                
                                # Select best performing model
                                best_model = 'Random Forest'  # Default
                                if cv_results:
                                    try:
                                        best_model = max(cv_results.keys(), 
                                                       key=lambda x: float(cv_results[x].get('Mean CV Score', '0')))
                                    except:
                                        pass
                                
                                if best_model in feature_importance:
                                    st.write(f"**Top Features from {best_model}:**")
                                    
                                    # Create importance plot
                                    importance_data = feature_importance[best_model]
                                    fig_importance = px.bar(
                                        importance_data.head(15),
                                        x='importance',
                                        y='feature',
                                        orientation='h',
                                        title=f"Top 15 Feature Importances - {best_model}",
                                        labels={'importance': 'Feature Importance', 'feature': 'Features'}
                                    )
                                    fig_importance.update_layout(height=600)
                                    st.plotly_chart(fig_importance, use_container_width=True)
                                    
                                    # Insights
                                    top_features = importance_data.head(5)['feature'].tolist()
                                    st.info(f"🎯 **Key Insights**: The most predictive features are: {', '.join(top_features[:3])}")
                        
                        # Pattern Discovery
                        st.subheader("🔍 Pattern Discovery")
                        
                        # Clustering analysis for pattern discovery
                        from sklearn.cluster import KMeans
                        from sklearn.decomposition import PCA
                        
                        # PCA for dimensionality reduction
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X.fillna(X.mean()))
                        
                        pca = PCA(n_components=min(5, X_scaled.shape[1]))
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Clustering
                        n_clusters = min(4, len(ml_data)//2)
                        if n_clusters >= 2:
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(X_pca)
                            
                            # Visualize clusters
                            cluster_df = pd.DataFrame({
                                'PC1': X_pca[:, 0],
                                'PC2': X_pca[:, 1],
                                'Cluster': clusters,
                                'Migraine': y,
                                'Date': ml_data['date'].values
                            })
                            
                            fig_cluster = px.scatter(
                                cluster_df,
                                x='PC1',
                                y='PC2',
                                color='Cluster',
                                symbol='Migraine',
                                title="Space Weather Patterns (PCA + K-Means Clustering)",
                                hover_data=['Date']
                            )
                            st.plotly_chart(fig_cluster, use_container_width=True)
                            
                            # Cluster analysis
                            cluster_analysis = []
                            for cluster_id in range(n_clusters):
                                cluster_mask = clusters == cluster_id
                                cluster_migraines = y[cluster_mask].sum()
                                cluster_total = cluster_mask.sum()
                                cluster_rate = cluster_migraines / cluster_total if cluster_total > 0 else 0
                                
                                cluster_analysis.append({
                                    'Cluster': f"Cluster {cluster_id}",
                                    'Days': cluster_total,
                                    'Migraines': cluster_migraines,
                                    'Migraine Rate': f"{cluster_rate:.1%}"
                                })
                            
                            cluster_df_summary = pd.DataFrame(cluster_analysis)
                            st.write("**Cluster Analysis:**")
                            st.dataframe(cluster_df_summary, use_container_width=True)
                        
                    else:
                        st.warning("⚠️ Insufficient data for machine learning analysis. Need more historical data.")
                else:
                    st.info("💡 Migraine data not available for AI analysis. Please ensure health data is properly loaded.")
            
            else:
                # Simplified analysis for deployment
                st.subheader("📊 Basic Pattern Analysis")
                st.info("💡 **Local Development Required**: Advanced AI analysis with GPU acceleration is available when running locally with the full ML stack.")
                
                # Simple correlation analysis
                if 'migraine' in df_filtered.columns and 'max_kp' in df_filtered.columns:
                    migraine_days = df_filtered[df_filtered['migraine'] == True]
                    if len(migraine_days) > 0:
                        st.write("**Simple Pattern Detection:**")
                        avg_kp_migraine = migraine_days['max_kp'].mean()
                        avg_kp_normal = df_filtered[df_filtered['migraine'] == False]['max_kp'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Avg Kp (Migraine Days)", f"{avg_kp_migraine:.2f}")
                        with col2:
                            st.metric("Avg Kp (Normal Days)", f"{avg_kp_normal:.2f}")
                        
                        if avg_kp_migraine > avg_kp_normal:
                            st.info(f"🔍 **Pattern Found**: Space weather activity is {((avg_kp_migraine/avg_kp_normal-1)*100):.1f}% higher on migraine days")
        
        # Summary statistics
        st.header("📊 Summary Statistics")
        
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
        
        # Raw data table
        if show_raw_data:
            st.header("📋 Raw Data")
            st.dataframe(df_filtered, use_container_width=True)
    
    else:
        st.warning("No data available for the selected date range.")

if __name__ == "__main__":
    main()
