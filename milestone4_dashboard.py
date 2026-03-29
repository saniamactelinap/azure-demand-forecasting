# ==========================================
# MILESTONE 4: ULTIMATE AZURE CAPACITY OPTIMIZER
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# -----------------------------
# 1. PAGE CONFIG & CSS
# -----------------------------
st.set_page_config(page_title="Azure Capacity Optimizer", page_icon="☁️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117 !important; color: #FAFAFA !important; }
    [data-testid="stSidebar"] { background-color: #15181E !important; border-right: 1px solid #2D3139 !important; }
    [data-testid="stMetric"] { background-color: #1E2127 !important; border-radius: 8px !important; padding: 15px !important; border: 1px solid #2D3139 !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important; }
    [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; color: #00A4EF !important; }
    [data-testid="stMetricLabel"] { font-size: 13px !important; color: #A0AEC0 !important; font-weight: 600 !important; text-transform: uppercase !important; }
    #MainMenu, footer, header {visibility: hidden;}
    .reportview-container .main .block-container{ padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#A0AEC0'), margin=dict(t=30, b=20, l=20, r=20)
    )
    return fig

# -----------------------------
# 2. LOAD DATA & MODELS
# -----------------------------
@st.cache_data
def load_data():
    try:
        # Load main engineered dataset
        df = pd.read_csv("azure_dataset_engineered.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure categorical columns exist (Merge from raw if one-hot encoded in engineered)
        if 'azure_region' not in df.columns or 'service_type' not in df.columns:
            raw_df = pd.read_csv("azure_dataset.csv")
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
            df = pd.merge(df, raw_df[['timestamp', 'azure_region', 'service_type']], on='timestamp', how='left')

        # Calculate specific dynamic features if not present
        if 'placed_capacity' not in df.columns:
            # Seeded for consistent UI
            np.random.seed(42) 
            df['placed_capacity'] = df['demand_units'] * np.random.uniform(1.1, 1.5, len(df))
            df['wasted_capacity'] = df['placed_capacity'] - df['demand_units']
            df['utilization_pct'] = (df['demand_units'] / df['placed_capacity']) * 100
            df['underutilized_flag'] = (df['utilization_pct'] < 60).astype(int)
            df['capacity_risk_event'] = (df['utilization_pct'] > 95).astype(int)
            df['headcount_proxy'] = (df['demand_units'] / 50).astype(int) + 2

        # Forecast results
        forecast = pd.read_csv("forecast_results.csv")
        forecast['index'] = range(len(forecast))
        
        return df, forecast
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None

@st.cache_resource
def load_models():
    xgb_model, arima_model = None, None
    try: xgb_model = joblib.load("trained_xgb_model.pkl")
    except: st.warning("trained_xgb_model.pkl not found.")
    
    try: arima_model = joblib.load("trained_arima_model.pkl")
    except: st.warning("trained_arima_model.pkl not found.")
        
    return xgb_model, arima_model

df, forecast = load_data()
xgb_model, arima_model = load_models()

if df is None:
    st.stop()

# -----------------------------
# 3. GLOBAL FILTERS & SIDEBAR
# -----------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg", width=140)
    st.markdown("### Global Filters")
    
    selected_regions = st.multiselect("Regions", df['azure_region'].unique(), default=df['azure_region'].unique())
    selected_services = st.multiselect("Service Fields", df['service_type'].unique(), default=df['service_type'].unique())
    alert_threshold = st.slider("Capacity Alert Threshold (%)", min_value=70, max_value=100, value=85)
    
    st.markdown("---")
    group_toggle = st.radio("Group By (Demand Page):", ["Service Type", "Region"])

# Apply Filters
filtered_df = df[(df['azure_region'].isin(selected_regions)) & (df['service_type'].isin(selected_services))]
if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()
# -----------------------------
# DASHBOARD TITLE
# -----------------------------
st.markdown("""
<h1 style="text-align: center; color: #FAFAFA; font-size: 36px; font-weight: 700; margin-bottom: 10px;">
    Azure Demand Forecasting Model
</h1>
<p style="text-align: center; color: #A0AEC0; font-size: 16px; margin-bottom: 30px;">
    Monitor utilization, costs, risks, and forecast trends across regions and services
</p>
""", unsafe_allow_html=True)

# -----------------------------
# PROFESSIONAL NAVIGATION BAR
# -----------------------------
nav_options = ["📊 KPI Overview", "📈 Demand & Usage", "🌍 Regional Analysis", "⚙️ Model & Forecast", "🚨 Risk Alerts & Capacity"]
nav_colors = ["#2563EB", "#38A169", "#FFB900", "#9F7AEA", "#E53E3E"]  # accent color for each page

st.markdown("""
<style>
div[data-baseweb="radio"] {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 20px;
}
div[data-baseweb="radio"] label {
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
    background-color: #1F2937;
    color: #FAFAFA;
}
div[data-baseweb="radio"] label:hover {
    opacity: 0.85;
}
div[data-baseweb="radio"] input:checked + label {
    border: 2px solid #FFFFFF;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Horizontal radio buttons for page selection
page = st.radio(
    "",
    nav_options,
    index=0,
    horizontal=True
)

# Optional: Add a subtle separator line
st.markdown("<hr style='border: 1px solid #2D3139; margin: 15px 0 25px 0;'>", unsafe_allow_html=True)

# -----------------------------
# PAGE 1: KPI OVERVIEW 
# -----------------------------
if page == "📊 KPI Overview":
    # Centered headers
    st.markdown("<h1 style='text-align: center; color: #FAFAFA; letter-spacing: 1px;'> Executive KPI Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #A0AEC0; margin-bottom: 30px;'>Monitor high level financial health, resource allocation, and system stability.</p>", unsafe_allow_html=True)
    
    CHART_THEME = ['#00A4EF', '#38A169', '#FFB900', '#E53E3E', '#9F7AEA'] 
    
    # --- TOP METRICS ROW 1 (Financials & Core Stats) ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cost", f"${filtered_df['cost_usd'].sum():,.0f}")
    m2.metric("Avg Cost per Unit", f"${filtered_df['cost_per_unit'].mean():.2f}")
    m3.metric("Avg Service Availability", f"{filtered_df['service_availability_pct'].mean():.2f}%")
    auto_rate = (filtered_df['is_auto_allocated'].mean() * 100) if 'is_auto_allocated' in filtered_df.columns else 0
    m4.metric("Auto Allocation Rate", f"{auto_rate:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- TOP METRICS ROW 2 (Dynamic Risk & Performance Features) ---
    m5, m6, m7, m8 = st.columns(4)
    
    # New dynamic metric: Critical Risk Events based on alert_threshold
    critical_risk_count = len(filtered_df[filtered_df['utilization_pct'] >= alert_threshold])
    m5.metric(f"Critical Risk Events (>{alert_threshold}%)", critical_risk_count)
    
    m6.metric("SLA Breaches", f"{filtered_df['low_availability_flag'].sum():,}")
    m7.metric("Priority Score", f"{filtered_df['priority_score'].mean():.1f} / 10")
    m8.metric("Demand Growth Rate", f"{(filtered_df['demand_growth_rate'].mean() * 100):+.2f}%")
    
    st.markdown("<hr style='border: 1px solid #2D3139; margin: 30px 0;'>", unsafe_allow_html=True)
    
    # --- FULL WIDTH CHART (Monthly Cost vs Critical Risk Events) ---
    st.markdown("<h3 style='color: #FAFAFA;'> Monthly Cost Trends vs Critical Risk Events</h3>", unsafe_allow_html=True)
    
    monthly_df = filtered_df.groupby(filtered_df['timestamp'].dt.to_period("M")).agg(
        {'cost_usd':'sum', 'utilization_pct': lambda x: (x >= alert_threshold).sum()}
    ).reset_index()
    monthly_df['timestamp'] = monthly_df['timestamp'].astype(str)
    
    fig_main = go.Figure(data=[
        go.Bar(name='Total Cost ($)', x=monthly_df['timestamp'], y=monthly_df['cost_usd'], marker_color=CHART_THEME[0], opacity=0.85),
        go.Scatter(name=f'Critical Risk Events (>{alert_threshold}%)', x=monthly_df['timestamp'], y=monthly_df['utilization_pct'], yaxis='y2', mode='lines+markers', marker_color=CHART_THEME[3], line=dict(width=4), marker=dict(size=10))
    ])
    
    fig_main.update_layout(
        barmode='group', 
        height=500, 
        xaxis_title="Month", 
        yaxis=dict(title="Total Cost ($)", tickprefix="$", title_font=dict(size=14)),
        yaxis2=dict(title="Critical Risk Events (Count)", overlaying='y', side='right', showgrid=False, title_font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, font=dict(size=14)),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    st.plotly_chart(apply_dark_theme(fig_main), use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- TWO COLUMN CHARTS (Secondary Metrics) ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<h3 style='color: #FAFAFA;'> Cost by Service Type</h3>", unsafe_allow_html=True)
        fig_pie = px.pie(
            filtered_df, 
            values='cost_usd', 
            names='service_type', 
            hole=0.5, 
            color_discrete_sequence=CHART_THEME,
            labels={'service_type': 'Service Category', 'cost_usd': 'Total Cost (USD)'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=16, marker=dict(line=dict(color='#0E1117', width=3)))
        fig_pie.update_layout(showlegend=False, height=450)
        st.plotly_chart(apply_dark_theme(fig_pie), use_container_width=True)
        
    with c2:
        st.markdown("<h3 style='color: #FAFAFA;'> Top Regions by Total Cost</h3>", unsafe_allow_html=True)
        region_cost = filtered_df.groupby('azure_region')['cost_usd'].sum().reset_index().sort_values('cost_usd', ascending=True)
        fig_hbar = px.bar(
            region_cost, 
            x='cost_usd', 
            y='azure_region', 
            orientation='h', 
            color_discrete_sequence=[CHART_THEME[1]],
            labels={'cost_usd': 'Total Cost ($)', 'azure_region': 'Azure Region'},
            text_auto='.2s' 
        )
        fig_hbar.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        fig_hbar.update_layout(xaxis_title="Total Cost ($)", yaxis_title="", xaxis=dict(tickprefix="$"), height=450)
        st.plotly_chart(apply_dark_theme(fig_hbar), use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- FULL WIDTH CHART (Deep Dive) ---
    st.markdown("<h3 style='color: #FAFAFA;'> Cost Per Unit Distribution</h3>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        filtered_df, 
        x='cost_per_unit', 
        nbins=40, 
        color_discrete_sequence=[CHART_THEME[4]],
        labels={'cost_per_unit': 'Cost Per Unit ($)'}
    )
    fig_hist.update_layout(
        yaxis_title="Frequency (Record Count)", 
        xaxis_title="Cost Per Unit ($)", 
        showlegend=False,
        height=400,
        xaxis=dict(tickprefix="$", title_font=dict(size=14)),
        yaxis=dict(title_font=dict(size=14))
    )
    st.plotly_chart(apply_dark_theme(fig_hist), use_container_width=True)

# -----------------------------
# PAGE 2: DEMAND & USAGE
# -----------------------------
elif page == "📈 Demand & Usage":
    st.title("Usage & Demand Over Time")
    
    group_col = 'service_type' if group_toggle == "Service Type" else 'azure_region'
    
    # Monthly Avg Units Line Chart
    monthly_avg = filtered_df.groupby([filtered_df['timestamp'].dt.to_period("M"), group_col])['demand_units'].mean().reset_index()
    monthly_avg['timestamp'] = monthly_avg['timestamp'].astype(str)
    
    fig_line = px.line(monthly_avg, x='timestamp', y='demand_units', color=group_col, markers=True, title=f"Monthly Avg Units by {group_toggle}")
    st.plotly_chart(apply_dark_theme(fig_line), use_container_width=True)
    
    # Rolling Statistics Chart
    st.subheader("Rolling Statistics (Actual vs 30-Day Mean)")
    daily_demand = filtered_df.groupby(filtered_df['timestamp'].dt.date)['demand_units'].sum().reset_index()
    daily_demand['30_day_rolling'] = daily_demand['demand_units'].rolling(30).mean()
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=daily_demand['timestamp'], y=daily_demand['demand_units'], name="Actual Usage", line=dict(color='#A0AEC0', width=1)))
    fig_roll.add_trace(go.Scatter(x=daily_demand['timestamp'], y=daily_demand['30_day_rolling'], name="30-Day Mean", line=dict(color='#00A4EF', width=3)))
    st.plotly_chart(apply_dark_theme(fig_roll), use_container_width=True)
    
    # Weekly Seasonality Index
    st.subheader("Weekly Seasonality Index")
    filtered_df['day_name'] = filtered_df['timestamp'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    seasonality = filtered_df.groupby('day_name')['demand_units'].mean().reindex(days_order).reset_index()
    
    fig_season = px.bar(seasonality, x='day_name', y='demand_units', color_discrete_sequence=['#7FBA00'])
    st.plotly_chart(apply_dark_theme(fig_season), use_container_width=True)

# -----------------------------
# PAGE 3: REGIONAL ANALYSIS
# -----------------------------
elif page == "🌍 Regional Analysis":
    st.title("Regional Capacity & Waste Breakdown")
    
    # 1️⃣ Utilization vs Waste (Risk Colors) — decongested
    st.subheader("Utilization vs Waste (Risk Colors)")
    sample_size = min(500, len(filtered_df))  # reduced sample for clarity
    fig_bubble = px.scatter(
        filtered_df.sample(sample_size), 
        x="utilization_pct", 
        y="wasted_capacity", 
        size="cost_usd", 
        color="capacity_risk_event", 
        hover_name="azure_region",
        color_continuous_scale="Reds", 
        size_max=25  # smaller bubbles for clarity
    )
    fig_bubble.update_layout(
        xaxis_title="Utilization (%)",
        yaxis_title="Wasted Capacity",
        legend_title="Capacity Risk",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(apply_dark_theme(fig_bubble), use_container_width=True)
    
    # 2️⃣ Regional Waste & Placed Capacity
    st.subheader("Regional Waste vs Placed Capacity")
    waste_regions = filtered_df.groupby('azure_region')[['wasted_capacity', 'placed_capacity']].sum().reset_index()
    fig_waste = px.bar(
        waste_regions, 
        x='azure_region', 
        y=['wasted_capacity', 'placed_capacity'], 
        barmode='group', 
        color_discrete_sequence=['#FFB900', '#00A4EF']
    )
    fig_waste.update_layout(
        xaxis_title="Azure Region",
        yaxis_title="Capacity Units",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(apply_dark_theme(fig_waste), use_container_width=True)
    
    # 3️⃣ Utilization Heatmap (Month vs Region)
    st.subheader("Utilization Heatmap (Month vs Region)")
    filtered_df['month'] = filtered_df['timestamp'].dt.month_name()
    # order months properly
    months_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    heat_df = filtered_df.groupby(['azure_region', 'month'])['utilization_pct'].mean().reset_index()
    heat_df['month'] = pd.Categorical(heat_df['month'], categories=months_order, ordered=True)
    fig_heat = px.density_heatmap(
        heat_df, 
        x="month", 
        y="azure_region", 
        z="utilization_pct", 
        color_continuous_scale="Viridis"
    )
    fig_heat.update_layout(
        xaxis_title="Month",
        yaxis_title="Azure Region",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(apply_dark_theme(fig_heat), use_container_width=True)
    
    # 4️⃣ New Feature: Average Cost per Unit by Region
    st.subheader("Average Cost per Unit by Region")
    avg_cost = filtered_df.groupby('azure_region').apply(lambda x: (x['cost_usd'].sum() / x['demand_units'].sum())).reset_index(name='avg_cost_per_unit')
    fig_cost = px.bar(
        avg_cost, 
        x='azure_region', 
        y='avg_cost_per_unit', 
        color='avg_cost_per_unit', 
        color_continuous_scale='Blues'
    )
    fig_cost.update_layout(
        xaxis_title="Azure Region",
        yaxis_title="Avg Cost per Unit (USD)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(apply_dark_theme(fig_cost), use_container_width=True)

# -----------------------------
# PAGE 4: MODEL & FORECAST
# -----------------------------
elif page == "⚙️ Model & Forecast":
    st.title("Model Accuracy & Forecasting Performance")
    
    actuals = forecast['Actual_Demand']
    preds = forecast['Predicted_Demand']
    
    # Generate ARIMA Preds
    try:
        arima_preds = arima_model.forecast(steps=len(actuals))
    except:
        arima_preds = actuals.shift(1).fillna(actuals.mean()) # fallback
    
    # Calculate Metrics
    def calc_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ape = np.abs((y_true - y_pred) / y_true) * 100
        dir_acc = np.mean(np.sign(y_true.diff().dropna()) == np.sign(y_pred.diff().dropna())) * 100
        bias = np.mean(y_pred - y_true)
        return r2, rmse, dir_acc, bias, ape

    r2, rmse, dir_acc, bias, ape = calc_metrics(actuals, preds)
    
    # Display Metrics
    st.subheader("Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score (XGBoost)", f"{r2:.4f}")
    col2.metric("RMSE (XGBoost)", f"{rmse:.2f}")
    col3.metric("Directional Accuracy", f"{dir_acc:.1f}%")
    col4.metric("Forecast Bias", f"{bias:.2f}")
    
    cA, cB, cC = st.columns(3)
    cA.metric("Within 5% Error", f"{(ape < 5).mean()*100:.1f}%")
    cB.metric("Within 10% Error", f"{(ape < 10).mean()*100:.1f}%")
    cC.metric("Within 15% Error", f"{(ape < 15).mean()*100:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1️⃣ Forecast vs Actuals (10% Tolerance)
    st.subheader("Forecast vs Actuals (w/ 10% Tolerance Bands)")
    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(y=actuals[:150], name="Actual", line=dict(color='#A0AEC0')))
    fig_band.add_trace(go.Scatter(y=preds[:150], name="XGBoost Pred", line=dict(color='#00A4EF')))
    fig_band.add_trace(go.Scatter(y=preds[:150]*1.1, line=dict(width=0), showlegend=False))
    fig_band.add_trace(go.Scatter(y=preds[:150]*0.9, fill='tonexty', fillcolor='rgba(0, 164, 239, 0.2)', line=dict(width=0), name="10% Band"))
    st.plotly_chart(apply_dark_theme(fig_band), use_container_width=True)
    
    # 2️⃣ Parity Plot
    st.subheader("Parity Plot (Actual vs Predicted)")
    fig_parity = px.scatter(x=actuals, y=preds, opacity=0.5, color_discrete_sequence=['#00A4EF'])
    fig_parity.add_shape(type="line", x0=actuals.min(), y0=actuals.min(), x1=actuals.max(), y1=actuals.max(),
                         line=dict(color="red", dash="dash"))
    st.plotly_chart(apply_dark_theme(fig_parity), use_container_width=True)
    
    # 3️⃣ ARIMA vs XGBoost Comparison
    st.subheader("ARIMA vs XGBoost Comparison")
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(y=actuals[:100], name="Actual", line=dict(color='#A0AEC0', width=2)))
    fig_compare.add_trace(go.Scatter(y=preds[:100], name="XGBoost Pred", line=dict(color='#00A4EF', width=2)))
    fig_compare.add_trace(go.Scatter(y=arima_preds[:100], name="ARIMA Pred", line=dict(color='#FFB900', width=2, dash='dot')))
    st.plotly_chart(apply_dark_theme(fig_compare), use_container_width=True)
    
    # 4️⃣ Top 10 Feature Importances
    st.subheader("Top 10 Feature Importances (XGBoost)")
    if xgb_model is not None and hasattr(xgb_model, 'feature_importances_'):
        importance = pd.DataFrame({'Feature': xgb_model.feature_names_in_, 'Importance': xgb_model.feature_importances_})
        importance = importance.sort_values('Importance', ascending=True).tail(10)
        fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#38A169'])
        st.plotly_chart(apply_dark_theme(fig_imp), use_container_width=True)
    
    # 5️⃣ Residuals Distribution
    st.subheader("Residuals Distribution")
    fig_res = px.histogram(actuals - preds, nbins=50, color_discrete_sequence=['#FFB900'])
    st.plotly_chart(apply_dark_theme(fig_res), use_container_width=True)

# -----------------------------
# PAGE 5: RISK ALERTS & CAPACITY
# -----------------------------
elif page == "🚨 Risk Alerts & Capacity":
    st.title("Capacity Risk & Alerts")
    
    # Filter high-risk events
    high_risk_df = filtered_df[filtered_df['utilization_pct'] >= alert_threshold]
    
    # Metrics Section
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Risky Events", len(high_risk_df), "Action Required", delta_color="inverse")
    col2.metric("Risk Threshold", f"{alert_threshold}%")
    
    # Replace overcapacity incidents with dynamic Critical Risk Events
    critical_risk_count = len(filtered_df[filtered_df['utilization_pct'] >= alert_threshold])
    col3.metric(f"Critical Risk Events (>{alert_threshold}%)", critical_risk_count)
    
    # Highest Risk Region
    risk_region = filtered_df.groupby('azure_region')['utilization_pct'].max().idxmax()
    col4.metric("Highest Risk Region", risk_region)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Weekly Risk Event Timeline
    st.subheader("Weekly Capacity Risk Events")
    filtered_df['week'] = filtered_df['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_risk = filtered_df[filtered_df['utilization_pct'] >= alert_threshold].groupby('week').size().reset_index(name='risk_events')
    fig_weekly = px.bar(
        weekly_risk, x='week', y='risk_events', color_discrete_sequence=['#E53E3E'], text='risk_events'
    )
    fig_weekly.update_layout(xaxis_title="Week", yaxis_title="Risk Events", uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(apply_dark_theme(fig_weekly), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # At-Risk Records Table
    st.subheader(f"At-Risk Records (Top 50 > {alert_threshold}% Utilization)")
    st.dataframe(
        high_risk_df[['timestamp', 'azure_region', 'service_type', 'demand_units', 'placed_capacity', 'utilization_pct', 'wasted_capacity']]
        .sort_values('utilization_pct', ascending=False).head(50),
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Capacity Planning Insights & Related Chart
    st.subheader("Capacity Planning Insights")
    
    # Chart: Top 10 regions by average utilization
    top_regions = filtered_df.groupby('azure_region')['utilization_pct'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_top_regions = px.bar(
        top_regions, x='azure_region', y='utilization_pct', color='utilization_pct',
        color_continuous_scale='Reds', text='utilization_pct'
    )
    fig_top_regions.update_layout(xaxis_title="Region", yaxis_title="Avg Utilization (%)")
    st.plotly_chart(apply_dark_theme(fig_top_regions), use_container_width=True)
    