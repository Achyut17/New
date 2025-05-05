import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Page configuration
st.set_page_config(
    page_title="MrBeast Channel Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for YouTube-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #606060;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 10px 0;
        text-align: left;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #606060;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #000000;
        margin-bottom: 0;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 20px 0 15px 0;
    }
    .tab-header {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-weight: 500;
    }
    .selectbox-label {
        font-size: 0.9rem;
        margin-bottom: 8px;
        color: #606060;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #2d7ff9;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlock"] > div:has(div[data-baseweb="tab-list"])) {
        background-color: #f5f5f5;
        padding: 0;
        border-radius: 8px;
    }
    div[data-testid="stSidebarUserContent"] {
        background-color: #f9f9f9;
    }
    section[data-testid="stSidebar"] {
        background-color: #f9f9f9;
        width: 250px;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 5rem;
    }
    .sidebar-header {
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .sidebar-text {
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .chart-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .nav-tab {
        display: inline-block;
        padding: 10px 20px;
        margin-right: 10px;
        cursor: pointer;
    }
    .nav-tab-active {
        border-bottom: 2px solid #2d7ff9;
        color: #000000;
        font-weight: 500;
    }
    .nav-tab-inactive {
        color: #606060;
    }
    .engagement-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #000000;
    }
    .video-card {
        display: flex;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .video-thumbnail {
        width: 160px;
        height: 90px;
        background-color: #e0e0e0;
        margin-right: 15px;
        border-radius: 4px;
    }
    .video-info {
        flex: 1;
    }
    .video-title {
        font-weight: 500;
        margin-bottom: 5px;
    }
    .video-metadata {
        font-size: 0.8rem;
        color: #606060;
        margin-bottom: 10px;
    }
    .video-stats {
        display: flex;
        gap: 15px;
    }
    .video-stat-item {
        flex: 1;
    }
    .video-stat-label {
        font-size: 0.7rem;
        color: #606060;
        text-transform: uppercase;
    }
    .video-stat-value {
        font-weight: 500;
    }
    .stSlider {
        padding-bottom: 1rem;
    }
    .stSlider > div > div > div {
        background-color: #2d7ff9;
    }
    .stTextInput > div > div > input {
        background-color: #f5f5f5;
    }
    .key-insights {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .insight-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    .insight-emoji {
        margin-right: 10px;
        font-size: 1.2rem;
    }
    .insight-text {
        flex: 1;
    }
    .insight-highlight {
        font-weight: 600;
    }
    .insight-metric {
        color: #2d7ff9;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load and process the data
@st.cache_data
def load_data():
    data = pd.read_csv("attached_assets/MrBeast_Full_Channel_Analytics_With_Titles.csv")
    
    # Convert date to datetime format
    data['Publish Date'] = pd.to_datetime(data['Publish Date'], format='%d-%m-%Y')
    
    # Add month and year columns for filtering
    data['Month'] = data['Publish Date'].dt.month_name()
    data['Month Num'] = data['Publish Date'].dt.month
    data['Year'] = data['Publish Date'].dt.year
    data['MonthYear'] = data['Publish Date'].dt.strftime('%b %Y')
    data['Day'] = data['Publish Date'].dt.day
    
    # Calculate watch time (estimated using views * duration)
    data['Total Watch Time (hours)'] = (data['Views'] * data['Duration (Seconds)']) / 3600
    
    # Calculate engagement rate ((likes + comments) / views * 100)
    data['Engagement Rate (%)'] = ((data['Likes'] + data['Comments']) / data['Views']) * 100
    
    # Calculate views per second (efficiency)
    data['Views per Second'] = data['Views'] / data['Duration (Seconds)']
    
    # Format duration as minutes:seconds
    data['Duration (Formatted)'] = data['Duration (Seconds)'].apply(
        lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
    )
    
    # Sort by publish date
    data = data.sort_values('Publish Date')
    
    return data

def format_large_number(num):
    """Format large numbers in millions (M) or billions (B)"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num:.0f}"

def forecast_views(time_series_data, forecast_periods=30):
    """Create a simple forecast for views using ARIMA model"""
    try:
        # If insufficient data, use simple exponential smoothing
        if len(time_series_data) < 10:
            model = ExponentialSmoothing(time_series_data, trend='add', seasonal=None)
            fit_model = model.fit()
            forecast = fit_model.forecast(forecast_periods)
        else:
            # Use ARIMA for better data
            model = ARIMA(time_series_data, order=(1, 1, 1))
            fit_model = model.fit()
            forecast = fit_model.forecast(steps=forecast_periods)
            
        return forecast
    except Exception as e:
        # Fallback to linear regression if issues with ARIMA
        x = np.arange(len(time_series_data))
        y = time_series_data.values
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        
        forecast_x = np.arange(len(time_series_data), len(time_series_data) + forecast_periods)
        forecast_x = sm.add_constant(forecast_x)
        forecast = model.predict(forecast_x)
        
        return forecast

# Load the data
data = load_data()

# Get the latest 30 days data for initial view
last_date = data['Publish Date'].max()
start_date = last_date - pd.Timedelta(days=30)
default_data = data[data['Publish Date'] >= start_date]

# Setup sidebar for filters
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Filter Options</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-text'>Filter Type</div>", unsafe_allow_html=True)
    filter_type = st.radio("Filter Type", ["Preset Periods", "Custom Date Range"], label_visibility="collapsed")
    
    if filter_type == "Preset Periods":
        st.markdown("<div class='sidebar-text'>Time Period</div>", unsafe_allow_html=True)
        period_options = ["Last 30 days", "Last 90 days", "Last 12 months", "Lifetime"]
        selected_period = st.selectbox("Time Period", period_options, label_visibility="collapsed", key="period_select")
        
        # Year and month filter are only shown when in preset periods
        st.markdown("<div class='sidebar-text'>Year</div>", unsafe_allow_html=True)
        years = sorted(data['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Year", years, label_visibility="collapsed", key="year_select")

        # Month filter with "All" option
        st.markdown("<div class='sidebar-text'>Month</div>", unsafe_allow_html=True)
        months = ["All"] + sorted(data[data['Year'] == selected_year]['Month'].unique().tolist(), 
                                key=lambda x: datetime.strptime(x, '%B').month)
        selected_month = st.selectbox("Month", months, label_visibility="collapsed", key="month_select")
    else:
        # Custom date range picker
        st.markdown("<div class='sidebar-text'>Date Range</div>", unsafe_allow_html=True)
        min_date = data['Publish Date'].min().date()
        max_date = data['Publish Date'].max().date()
        
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="start_date")
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="end_date")
        
        if start_date > end_date:
            st.error("Start date must be before end date")
            # Swap the dates to ensure valid range
            start_date, end_date = end_date, start_date

# Filter the data based on selections
if filter_type == "Preset Periods":
    if selected_period == "Last 30 days":
        filtered_data = data[data['Publish Date'] >= (data['Publish Date'].max() - pd.Timedelta(days=30))]
    elif selected_period == "Last 90 days":
        filtered_data = data[data['Publish Date'] >= (data['Publish Date'].max() - pd.Timedelta(days=90))]
    elif selected_period == "Last 12 months":
        filtered_data = data[data['Publish Date'] >= (data['Publish Date'].max() - pd.Timedelta(days=365))]
    else:  # Lifetime
        filtered_data = data.copy()
        
    # Further filter by year and month if selected
    if selected_period == "Lifetime":
        filtered_data = data[data['Year'] == selected_year]
        if selected_month != "All":
            filtered_data = filtered_data[filtered_data['Month'] == selected_month]
else:
    # Filter by custom date range
    filtered_data = data[(data['Publish Date'].dt.date >= start_date) & 
                        (data['Publish Date'].dt.date <= end_date)]

# Calculate key metrics for the filtered period
total_videos = len(filtered_data)
total_views = filtered_data['Views'].sum()
total_likes = filtered_data['Likes'].sum()
total_comments = filtered_data['Comments'].sum()
engagement_rate = round(((total_likes + total_comments) / total_views) * 100, 2) if total_views > 0 else 0

# Calculate key insights
if len(filtered_data) > 0:
    # Most popular video by views
    most_popular_video = filtered_data.loc[filtered_data['Views'].idxmax()]
    
    # Most engaging video by like ratio
    most_engaging_video = filtered_data.loc[filtered_data['Like Ratio (%)'].idxmax()]
    
    # Optimal video length based on views per second
    optimal_video = filtered_data.loc[filtered_data['Views per Second'].idxmax()]
    optimal_duration = optimal_video['Duration (Formatted)']
else:
    most_popular_video = None
    most_engaging_video = None
    optimal_duration = None

# Main content area
# Header with MrBeast logo and title
col1, col2 = st.columns([1, 6])
with col1:
    # MrBeast logo (use a placeholder image if the remote image doesn't load)
    st.image("attached_assets/image_1746410705102.png", width=80)
with col2:
    st.markdown("<div class='main-header'>MrBeast Channel Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>390,000,000 subscribers • 867 videos • 81,472,786,518 views</div>", unsafe_allow_html=True)

# Navigation tabs for Channel vs Video Performance
channel_tab, video_tab = st.tabs(["Channel Performance", "Video Performance"])

with channel_tab:
    # Channel Performance Overview
    st.markdown("<div class='section-header'>Channel Performance Overview</div>", unsafe_allow_html=True)

    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Videos</div><div class='metric-value'>{}</div></div>".format(total_videos), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Views</div><div class='metric-value'>{}</div></div>".format(format_large_number(total_views)), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Likes</div><div class='metric-value'>{}</div></div>".format(format_large_number(total_likes)), unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Comments</div><div class='metric-value'>{}</div></div>".format(format_large_number(total_comments)), unsafe_allow_html=True)

    # Engagement rate
    st.markdown("<div class='metric-card'><div class='metric-label'>Engagement Rate</div><div class='engagement-value'>{}%</div></div>".format(engagement_rate), unsafe_allow_html=True)

    # Performance Trends section
    st.markdown("<div class='section-header'>Performance Trends</div>", unsafe_allow_html=True)
    st.markdown("<div class='selectbox-label'>Select Metric to Visualize</div>", unsafe_allow_html=True)

    # Create a dropdown for metric selection
    metric_options = ["Views", "Likes", "Comments", "Like Ratio (%)", "Comment Ratio (%)", "Engagement Rate (%)", "Duration (Seconds)"]
    selected_viz_metric = st.selectbox(
        "Select metric to visualize",
        metric_options,
        index=0,
        label_visibility="collapsed",
        key="main_metric"
    )

    # Create metric for showing video performance over time
    st.markdown(f"<div class='chart-container'><h3>{selected_viz_metric} per Video Over Time</h3>", unsafe_allow_html=True)

    # Prepare data for time series visualization
    if len(filtered_data) > 0:
        # Create scatter plot for video performance over time
        fig_time = px.scatter(
            filtered_data,
            x='Publish Date',
            y=selected_viz_metric,
            size='Views',
            hover_name='Video Title',
            color_discrete_sequence=['#2d7ff9'],
            labels={selected_viz_metric: selected_viz_metric, 'Publish Date': 'Publication Date'},
            template='plotly_white'
        )
        
        # Add a trend line
        fig_time.add_trace(
            go.Scatter(
                x=filtered_data['Publish Date'],
                y=filtered_data[selected_viz_metric],
                mode='lines',
                line=dict(color='#2d7ff9', width=2),
                showlegend=False
            )
        )
        
        fig_time.update_layout(
            xaxis_title='Publication Date',
            yaxis_title=selected_viz_metric,
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='closest',
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Add trend forecast
        if selected_viz_metric == "Views" and st.checkbox("Show Trend Forecast", value=True):
            st.markdown("<div class='chart-container'><h3>Forecast for Future Views</h3>", unsafe_allow_html=True)
            
            # Forecast method selection
            st.markdown("<div style='margin-bottom: 15px;'>Forecast Method</div>", unsafe_allow_html=True)
            forecast_method = st.radio(
                "Forecast Method",
                ["Linear Regression", "ARIMA Model"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            if len(filtered_data) >= 5:  # Need at least some data points for forecasting
                # Prepare time series data - aggregate by month for cleaner forecasting
                monthly_data = filtered_data.copy()
                monthly_data['Month'] = monthly_data['Publish Date'].dt.strftime('%Y-%m')
                monthly_avg = monthly_data.groupby('Month').agg({
                    selected_viz_metric: 'mean',
                    'Publish Date': 'min'  # Use first date of month
                }).reset_index()
                
                ts_data = monthly_avg.set_index('Publish Date')[selected_viz_metric]
                
                # Set forecast periods
                forecast_months = 5
                
                # Get forecast
                if forecast_method == "Linear Regression":
                    # Simple linear regression
                    x = np.arange(len(ts_data))
                    y = ts_data.values
                    x = sm.add_constant(x)
                    model = sm.OLS(y, x).fit()
                    
                    forecast_x = np.arange(len(ts_data), len(ts_data) + forecast_months)
                    forecast_x = sm.add_constant(forecast_x)
                    forecast = model.predict(forecast_x)
                else:
                    # ARIMA model
                    try:
                        model = ARIMA(ts_data, order=(1, 1, 1))
                        fit_model = model.fit()
                        forecast = fit_model.forecast(steps=forecast_months)
                    except:
                        # Fallback to exponential smoothing
                        model = ExponentialSmoothing(ts_data, trend='add', seasonal=None)
                        fit_model = model.fit()
                        forecast = fit_model.forecast(forecast_months)
                
                # Create future dates (monthly)
                last_date = ts_data.index.max()
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_months)]
                
                # Create forecast visualization
                fig_forecast = go.Figure()
                
                # Add historical data - as a line
                fig_forecast.add_trace(
                    go.Scatter(
                        x=ts_data.index,
                        y=ts_data.values,
                        name="Historical",
                        line=dict(color="#2d7ff9", width=2),
                    )
                )
                
                # Add forecast
                fig_forecast.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=forecast,
                        name="Forecast",
                        line=dict(color="#ff0000", width=2),
                    )
                )
                
                # Calculate y-axis range
                y_min = min(min(ts_data.values), min(forecast)) * 0.9
                y_max = max(max(ts_data.values), max(forecast)) * 1.1
                
                fig_forecast.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Views",
                    yaxis_range=[y_min, y_max],
                    height=400,
                    margin=dict(l=40, r=40, t=10, b=40),
                    plot_bgcolor='white',
                    legend=dict(
                        title="Type",
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Display forecast statistics
                st.markdown("<h4>Views Forecast (Next 5 Months)</h4>", unsafe_allow_html=True)
                
                # Generate forecast months with expected values
                future_months = []
                for i, (date, value) in enumerate(zip(future_dates, forecast)):
                    month_name = date.strftime("%b %Y")
                    future_months.append({
                        "Month": i + 1,
                        "Date": month_name,
                        "Views": int(value)
                    })
                
                # Display the forecast statistics
                for i, month in enumerate(future_months):
                    st.markdown(f"Month {month['Month']} ({month['Date']}): {format_large_number(month['Views'])} views")
    else:
        st.info("No data available for the selected time period.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Video Performance Distribution
    st.markdown("<div class='section-header'>Video Performance Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='selectbox-label'>Select Metric for Distribution Analysis</div>", unsafe_allow_html=True)

    distribution_metric = st.selectbox(
        "Select metric for distribution",
        metric_options,
        index=0,
        label_visibility="collapsed",
        key="distribution_metric"
    )

    st.markdown(f"<div class='chart-container'><h3>Distribution of Video {distribution_metric}</h3>", unsafe_allow_html=True)

    if len(filtered_data) > 0:
        # Create histogram
        fig_dist = px.histogram(
            filtered_data,
            x=distribution_metric,
            nbins=10,
            color_discrete_sequence=['#2d7ff9'],
            template='plotly_white'
        )
        
        fig_dist.update_layout(
            xaxis_title=distribution_metric,
            yaxis_title='Number of Videos',
            height=350,
            margin=dict(l=40, r=40, t=10, b=40),
            bargap=0.1,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No data available for the selected time period.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Metric Correlations
    st.markdown("<div class='section-header'>Metric Correlations</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-container'><h3>Correlation Between Metrics</h3>", unsafe_allow_html=True)

    if len(filtered_data) > 2:  # Need at least a few data points for correlation
        # Create correlation matrix
        corr_metrics = ['Views', 'Likes', 'Comments', 'Duration (Seconds)']
        corr_df = filtered_data[corr_metrics].corr()
        
        # Create a heatmap
        fig_corr = px.imshow(
            corr_df,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto='.2f'
        )
        
        fig_corr.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=10, b=40),
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough data points to calculate correlations.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Key Insights
    if most_popular_video is not None and most_engaging_video is not None and optimal_duration is not None:
        st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
        
        st.markdown(f"📊 Most popular video: **{most_popular_video['Video Title']}** with **{format_large_number(most_popular_video['Views'])}** views")
        st.markdown(f"🔥 Most engaging video: **{most_engaging_video['Video Title']}** with a **{most_engaging_video['Like Ratio (%)']:.2f}%** like ratio")
        st.markdown(f"⏱️ Optimal video length: **{optimal_duration}** (based on views per second of duration)")

with video_tab:
    # Video Performance Analysis Section
    st.markdown("<div class='section-header'>Video Performance Analysis</div>", unsafe_allow_html=True)

    # Filter videos section
    st.markdown("<h3>Filter Videos</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        min_views = st.number_input("Minimum Views", min_value=0, value=0, step=1000000)

    with col2:
        sort_options = ["Publication Date (Newest)", "Views (Highest)", "Likes (Highest)", "Comments (Highest)", "Engagement Rate (Highest)"]
        sort_by = st.selectbox("Sort By", sort_options)

    # Apply additional filters
    filtered_videos = filtered_data[filtered_data['Views'] >= min_views]

    # Apply sorting
    if sort_by == "Publication Date (Newest)":
        filtered_videos = filtered_videos.sort_values('Publish Date', ascending=False)
    elif sort_by == "Views (Highest)":
        filtered_videos = filtered_videos.sort_values('Views', ascending=False)
    elif sort_by == "Likes (Highest)":
        filtered_videos = filtered_videos.sort_values('Likes', ascending=False)
    elif sort_by == "Comments (Highest)":
        filtered_videos = filtered_videos.sort_values('Comments', ascending=False)
    elif sort_by == "Engagement Rate (Highest)":
        filtered_videos = filtered_videos.sort_values('Engagement Rate (%)', ascending=False)

    # Display number of results
    st.markdown(f"<h3>Videos ({len(filtered_videos)} results)</h3>", unsafe_allow_html=True)

    # Display video cards
    for idx, video in filtered_videos.head(10).iterrows():
        st.markdown(f"""
        <div class="video-card">
            <div class="video-thumbnail"></div>
            <div class="video-info">
                <div class="video-title">{video['Video Title']}</div>
                <div class="video-metadata">Published on: {video['Publish Date'].strftime('%b %d, %Y')} • Duration: {video['Duration (Formatted)']}</div>
                <div class="video-stats">
                    <div class="video-stat-item">
                        <div class="video-stat-label">Views</div>
                        <div class="video-stat-value">{format_large_number(video['Views'])}</div>
                    </div>
                    <div class="video-stat-item">
                        <div class="video-stat-label">Likes</div>
                        <div class="video-stat-value">{format_large_number(video['Likes'])}</div>
                    </div>
                    <div class="video-stat-item">
                        <div class="video-stat-label">Comments</div>
                        <div class="video-stat-value">{format_large_number(video['Comments'])}</div>
                    </div>
                    <div class="video-stat-item">
                        <div class="video-stat-label">Like Ratio</div>
                        <div class="video-stat-value">{video['Like Ratio (%)']:.2f}%</div>
                    </div>
                    <div class="video-stat-item">
                        <div class="video-stat-label">Comment Ratio</div>
                        <div class="video-stat-value">{video['Comment Ratio (%)']:.2f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
