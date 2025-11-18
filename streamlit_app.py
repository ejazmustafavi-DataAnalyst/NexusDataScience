import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
#import shap

# ======================================================================================
# PAGE CONFIG
# ======================================================================================
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Sales Forecasting & KPI Dashboard")
st.markdown("An interactive product analytics dashboard powered by your regression model.")

# ======================================================================================
# LOAD DATA & MODEL
# ======================================================================================
@st.cache_data
def load_data():
   
    df = pd.read_excel("final_clean_sales.xlsx", parse_dates=["date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("best_sales_model.joblib")

df = load_data()
model = load_model()

feature_names = model.feature_names_in_

# ======================================================================================
# SIDEBAR FILTERS
# ======================================================================================
st.sidebar.header("🔎 Filters")

# Product dropdown
product_col = "Product" if "Product" in df.columns else st.sidebar.selectbox(
    "Select product column", df.columns)

all_products = df[product_col].unique().tolist()
selected_products = st.sidebar.multiselect(
    "Select Product(s)", all_products, default=all_products[:3])

# Date range selector
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Filter data
filtered_df = df[
    (df[product_col].isin(selected_products)) &
    (df["date"].between(start_date, end_date))
]

# ======================================================================================
# KPIs
# ======================================================================================
st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

total_sales = filtered_df["Revenue"].sum()
avg_sales = filtered_df["Revenue"].mean()
unique_products = filtered_df[product_col].nunique()

col1.metric("Total Revenue", f"${total_sales:,.0f}")
col2.metric("Average Daily Revenue", f"${avg_sales:,.0f}")
col3.metric("Products Selected", unique_products)

# ======================================================================================
# FORECASTING USING MODEL
# ======================================================================================
st.subheader("📈 Model Forecast")

# Build feature frame required by model
def prepare_features(df):
    missing_cols = [col for col in feature_names if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    return df[feature_names]

# Predict + Confidence Interval
features = prepare_features(filtered_df.copy())
predictions = model.predict(features)

# Apply ±1.96 * std_error (approx 95% CI)
if hasattr(model, "predict"):
    residuals = filtered_df["Revenue"] - predictions
    std_err = np.std(residuals)
else:
    std_err = 0

lower_ci = predictions - 1.96 * std_err
upper_ci = predictions + 1.96 * std_err

filtered_df["Predicted"] = predictions
filtered_df["Lower_CI"] = lower_ci
filtered_df["Upper_CI"] = upper_ci

# ======================================================================================
# FORECAST CHART
# ======================================================================================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["Revenue"],
    mode="lines+markers", name="Actual Revenue"
))

fig.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["Predicted"],
    mode="lines", name="Predicted Revenue"
))

fig.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["Upper_CI"],
    line=dict(width=0), showlegend=False
))
fig.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["Lower_CI"],
    fill='tonexty',
    fillcolor='rgba(0, 100, 250, 0.15)',
    line=dict(width=0),
    name="Confidence Interval"
))

fig.update_layout(
    title="Sales Forecast with Confidence Interval",
    xaxis_title="Date",
    yaxis_title="Revenue"
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# FEATURE IMPORTANCE (SHAP VALUES)
# ======================================================================================
st.subheader("🧠 Feature Importance")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

shap_fig = shap.summary_plot(shap_values, features, plot_type="bar", show=False)
st.pyplot(shap_fig)

# ======================================================================================
# PRODUCT TRENDS
# ======================================================================================
st.subheader("📦 Product Sales Trends")

trend_fig = px.line(
    filtered_df,
    x="date",
    y="Revenue",
    color=product_col,
    title="Revenue Trend by Product"
)

st.plotly_chart(trend_fig, use_container_width=True)

# ======================================================================================
# CSV DOWNLOAD
# ======================================================================================
st.subheader("📥 Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_sales_view.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")
