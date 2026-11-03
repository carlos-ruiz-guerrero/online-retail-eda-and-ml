# ============================================================
# VISUALIZATION MODULE
# Generates all charts for README and presentation
# Images are saved under /images
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

os.makedirs("images", exist_ok=True)

# ============================================================
# 1. MONTHLY REVENUE TREND
# ============================================================

monthly_revenue = (
    df_clean
    .set_index("InvoiceDate")
    .resample("M")
    ["Revenue"]
    .sum()
)

plt.figure(figsize=(12,6))

plt.plot(
    monthly_revenue.index,
    monthly_revenue.values,
    linewidth=3,
    color="#1f77b4"
)

plt.title(
    "Monthly Revenue Trend",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Date")
plt.ylabel("Revenue (£)")

plt.tight_layout()

plt.savefig(
    "images/monthly_revenue.png",
    dpi=300
)

plt.close()

# ============================================================
# 2. TOP COUNTRIES BY REVENUE
# ============================================================

country_revenue = (
    df_clean
    .groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

sns.barplot(
    x=country_revenue.values,
    y=country_revenue.index,
    palette="Blues_r"
)

plt.title(
    "Top 10 Countries by Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Revenue (£)")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    "images/country_revenue.png",
    dpi=300
)

plt.close()

# ============================================================
# 3. TOP PRODUCTS BY REVENUE
# ============================================================

top_products = (
    df_clean
    .groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(12,8))

sns.barplot(
    x=top_products.values,
    y=top_products.index,
    palette="viridis"
)

plt.title(
    "Top 15 Products by Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Revenue (£)")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig(
    "images/top_products.png",
    dpi=300
)

plt.close()

# ============================================================
# 4. REVENUE DISTRIBUTION
# ============================================================

customer_revenue = (
    df_clean
    .groupby("CustomerID")["Revenue"]
    .sum()
)

plt.figure(figsize=(10,6))

sns.histplot(
    np.log1p(customer_revenue),
    bins=50,
    kde=True,
    color="darkgreen"
)

plt.title(
    "Customer Revenue Distribution (Log Scale)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Log Revenue")

plt.tight_layout()

plt.savefig(
    "images/revenue_distribution.png",
    dpi=300
)

plt.close()

# ============================================================
# 5. ELBOW METHOD
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    range(2,11),
    inertia,
    marker="o",
    linewidth=3
)

plt.title(
    "Elbow Method",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.tight_layout()

plt.savefig(
    "images/elbow_method.png",
    dpi=300
)

plt.close()

# ============================================================
# 6. CUSTOMER SEGMENTS (PCA)
# ============================================================

plt.figure(figsize=(12,8))

sns.scatterplot(
    data=rfm,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2",
    s=100
)

plt.title(
    "Customer Segmentation using PCA",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "images/customer_segments.png",
    dpi=300
)

plt.close()

# ============================================================
# 7. CLUSTER PROFILE HEATMAP
# ============================================================

cluster_profile = (
    rfm.groupby("Cluster")
    [["Recency","Frequency","Monetary"]]
    .mean()
)

plt.figure(figsize=(10,6))

sns.heatmap(
    cluster_profile,
    annot=True,
    cmap="RdYlGn",
    fmt=".0f"
)

plt.title(
    "Cluster Characteristics",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "images/cluster_profile.png",
    dpi=300
)

plt.close()

# ============================================================
# 8. CLUSTER SIZE DISTRIBUTION
# ============================================================

cluster_counts = (
    rfm["Cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8,6))

sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values,
    palette="Set2"
)

plt.title(
    "Customers per Cluster",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "images/cluster_sizes.png",
    dpi=300
)

plt.close()

# ============================================================
# 9. PARETO ANALYSIS (80/20 RULE)
# ============================================================

customer_sales = (
    df_clean
    .groupby("CustomerID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

cumulative_percentage = (
    customer_sales.cumsum()
    / customer_sales.sum()
) * 100

customer_percentage = (
    np.arange(len(customer_sales))
    / len(customer_sales)
) * 100

plt.figure(figsize=(12,6))

plt.plot(
    customer_percentage,
    cumulative_percentage,
    linewidth=3,
    color="red"
)

plt.axvline(
    20,
    linestyle="--",
    color="black"
)

plt.axhline(
    80,
    linestyle="--",
    color="black"
)

plt.title(
    "Pareto Analysis (Customer Revenue Contribution)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("% of Customers")
plt.ylabel("% of Revenue")

plt.tight_layout()

plt.savefig(
    "images/pareto_analysis.png",
    dpi=300
)

plt.close()

# ============================================================
# 10. RFM PAIRPLOT
# ============================================================

sample_size = min(3000, len(rfm))

rfm_sample = rfm.sample(
    sample_size,
    random_state=42
)

pairplot = sns.pairplot(
    rfm_sample,
    vars=["Recency","Frequency","Monetary"],
    hue="Cluster",
    corner=True
)

pairplot.fig.suptitle(
    "RFM Feature Relationships",
    y=1.02,
    fontsize=16
)

pairplot.savefig(
    "images/rfm_pairplot.png",
    dpi=300
)

plt.close()

# ============================================================
# SUMMARY
# ============================================================

print("\nCharts generated successfully.")
print("\nSaved files:")

print("""
images/
│
├── monthly_revenue.png
├── country_revenue.png
├── top_products.png
├── revenue_distribution.png
├── elbow_method.png
├── customer_segments.png
├── cluster_profile.png
├── cluster_sizes.png
├── pareto_analysis.png
└── rfm_pairplot.png
""")
