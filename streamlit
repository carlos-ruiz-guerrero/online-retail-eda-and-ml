# ============================================================
# ONLINE RETAIL CUSTOMER SEGMENTATION DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Online Retail Analytics",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "online_retail_clean.csv"
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )

    df["Revenue"] = (
        df["Quantity"]
        * df["UnitPrice"]
    )

    return df

df = load_data()

# ============================================================
# HEADER
# ============================================================

st.title("🛍️ Online Retail Analytics Dashboard")

st.markdown("""
Customer segmentation and sales analytics using the
UCI Online Retail Dataset.

**Machine Learning Methodology**
- Data Cleaning
- RFM Feature Engineering
- K-Means Clustering
- PCA Visualization
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

countries = st.sidebar.multiselect(
    "Select Country",
    sorted(df["Country"].unique()),
    default=sorted(df["Country"].unique())
)

date_range = st.sidebar.date_input(
    "Date Range",
    [
        df["InvoiceDate"].min(),
        df["InvoiceDate"].max()
    ]
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df.copy()

filtered = filtered[
    filtered["Country"].isin(countries)
]

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

filtered = filtered[
    (filtered["InvoiceDate"] >= start_date)
    &
    (filtered["InvoiceDate"] <= end_date)
]

# ============================================================
# KPIs
# ============================================================

total_revenue = filtered["Revenue"].sum()

total_orders = filtered["InvoiceNo"].nunique()

total_customers = filtered["CustomerID"].nunique()

average_order_value = (
    total_revenue / total_orders
)

st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Revenue",
    f"£{total_revenue:,.0f}"
)

col2.metric(
    "Orders",
    f"{total_orders:,}"
)

col3.metric(
    "Customers",
    f"{total_customers:,}"
)

col4.metric(
    "Avg Order Value",
    f"£{average_order_value:,.2f}"
)

# ============================================================
# MONTHLY REVENUE
# ============================================================

st.markdown("---")
st.subheader("📈 Revenue Trend")

monthly_revenue = (
    filtered
    .set_index("InvoiceDate")
    .resample("M")
    ["Revenue"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_revenue,
    x="InvoiceDate",
    y="Revenue",
    markers=True,
    title="Monthly Revenue"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# COUNTRY ANALYSIS
# ============================================================

st.markdown("---")
st.subheader("🌍 Country Analysis")

country_sales = (
    filtered
    .groupby("Country")
    ["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

fig = px.bar(
    country_sales,
    x="Revenue",
    y="Country",
    orientation="h",
    color="Revenue",
    title="Top Countries"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# PRODUCT ANALYSIS
# ============================================================

st.markdown("---")
st.subheader("🏆 Top Products")

top_products = (
    filtered
    .groupby("Description")
    ["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

fig = px.bar(
    top_products,
    x="Revenue",
    y="Description",
    orientation="h",
    title="Top Products by Revenue"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

st.markdown("---")
st.subheader("🤖 Customer Segmentation")

rfm = (
    filtered.groupby("CustomerID")
    .agg(
        Recency=("InvoiceDate",
                 lambda x:
                 (filtered["InvoiceDate"].max()
                  - x.max()).days),

        Frequency=("InvoiceNo","nunique"),

        Monetary=("Revenue","sum")
    )
)

# basic segmentation

rfm["Segment"] = pd.qcut(
    rfm["Monetary"],
    4,
    labels=[
        "Low Value",
        "Medium",
        "High",
        "VIP"
    ]
)

segment_counts = (
    rfm["Segment"]
    .value_counts()
    .reset_index()
)

fig = px.pie(
    segment_counts,
    names="Segment",
    values="count",
    title="Customer Segments"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# RFM ANALYSIS
# ============================================================

st.markdown("---")
st.subheader("📊 Customer Value Matrix")

fig = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color="Segment",
    size="Monetary",
    hover_data=["Recency"]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# PARETO ANALYSIS
# ============================================================

st.markdown("---")
st.subheader("💰 Pareto Analysis")

customer_sales = (
    filtered
    .groupby("CustomerID")
    ["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

cumulative = (
    customer_sales.cumsum()
    /
    customer_sales.sum()
    * 100
)

customer_pct = (
    np.arange(
        1,
        len(customer_sales)+1
    )
    /
    len(customer_sales)
    * 100
)

pareto = pd.DataFrame({
    "Customers (%)": customer_pct,
    "Revenue (%)": cumulative.values
})

fig = px.line(
    pareto,
    x="Customers (%)",
    y="Revenue (%)",
    title="Pareto Revenue Distribution"
)

fig.add_hline(
    y=80,
    line_dash="dash"
)

fig.add_vline(
    x=20,
    line_dash="dash"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# RAW DATA
# ============================================================

st.markdown("---")
st.subheader("📄 Dataset Preview")

st.dataframe(
    filtered.head(100),
    use_container_width=True
)

# ============================================================
# DOWNLOAD SECTION
# ============================================================

st.download_button(
    label="⬇️ Download Clean Dataset",
    data=filtered.to_csv(index=False),
    file_name="online_retail_clean.csv",
    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption("""
Online Retail Customer Segmentation Dashboard
Built with Python, Streamlit, Plotly and Scikit-Learn
""")
