# ============================================================
# ONLINE RETAIL DATASET
# COMPLETE EDA + DATA CLEANING + CUSTOMER SEGMENTATION
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# 2. DOWNLOAD DATASET
# ------------------------------------------------------------

print("Downloading UCI Online Retail Dataset...")

online_retail = fetch_ucirepo(id=352)

# Original dataframe
df = online_retail.data.original.copy()

print("\nDataset Loaded Successfully")
print(df.head())

# ------------------------------------------------------------
# 3. BASIC DATA INSPECTION
# ------------------------------------------------------------

print("\n====================")
print("DATASET INFORMATION")
print("====================")

print(df.info())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

print("\n====================")
print("MISSING VALUES")
print("====================")

print(df.isnull().sum())

# ------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n====================")
print("DESCRIPTIVE STATISTICS")
print("====================")

print(df.describe(include="all"))

# ------------------------------------------------------------
# 6. DUPLICATES
# ------------------------------------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ------------------------------------------------------------
# 7. CONVERT DATE COLUMN
# ------------------------------------------------------------

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ------------------------------------------------------------
# 8. CREATE REVENUE COLUMN
# ------------------------------------------------------------

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# ------------------------------------------------------------
# 9. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\nGenerating EDA Charts...")

# ------------------------------------------------------------
# Revenue Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["Revenue"],
    bins=100
)

plt.title("Revenue Distribution")
plt.show()

# ------------------------------------------------------------
# Quantity Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["Quantity"],
    bins=100
)

plt.title("Quantity Distribution")
plt.xlim(-100, 200)

plt.show()

# ------------------------------------------------------------
# Unit Price Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["UnitPrice"],
    bins=100
)

plt.title("Unit Price Distribution")
plt.xlim(0,100)

plt.show()

# ------------------------------------------------------------
# Top Countries
# ------------------------------------------------------------

top_countries = (
    df["Country"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top Countries")
plt.show()

# ------------------------------------------------------------
# Monthly Revenue Trend
# ------------------------------------------------------------

monthly_revenue = (
    df
    .set_index("InvoiceDate")
    .resample("M")
    ["Revenue"]
    .sum()
)

monthly_revenue.plot(
    figsize=(12,6),
    title="Monthly Revenue"
)

plt.show()

# ============================================================
# DATA CLEANING
# ============================================================

print("\nStarting Data Cleaning...")

df_clean = df.copy()

# ------------------------------------------------------------
# Remove cancelled invoices
# CXXXX invoices indicate cancellations
# ------------------------------------------------------------

df_clean = df_clean[
    ~df_clean["InvoiceNo"]
    .astype(str)
    .str.startswith("C")
]

# ------------------------------------------------------------
# Remove missing customers
# ------------------------------------------------------------

df_clean = df_clean.dropna(
    subset=["CustomerID"]
)

# ------------------------------------------------------------
# Remove negative quantity
# ------------------------------------------------------------

df_clean = df_clean[
    df_clean["Quantity"] > 0
]

# ------------------------------------------------------------
# Remove zero/negative prices
# ------------------------------------------------------------

df_clean = df_clean[
    df_clean["UnitPrice"] > 0
]

# ------------------------------------------------------------
# Remove duplicates
# ------------------------------------------------------------

df_clean = df_clean.drop_duplicates()

# ------------------------------------------------------------
# Recalculate revenue
# ------------------------------------------------------------

df_clean["Revenue"] = (
    df_clean["Quantity"]
    * df_clean["UnitPrice"]
)

print("\nClean Dataset Shape:")
print(df_clean.shape)

# ------------------------------------------------------------
# Save Clean Dataset
# ------------------------------------------------------------

df_clean.to_csv(
    "online-retail-clean-data.csv",
    index=False
)

print("\nSaved:")
print("online-retail-clean-data.csv")

# ============================================================
# FEATURE ENGINEERING
# CUSTOMER SEGMENTATION USING RFM
# ============================================================

print("\nBuilding RFM Features...")

# ------------------------------------------------------------
# Snapshot date
# ------------------------------------------------------------

snapshot_date = (
    df_clean["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)

# ------------------------------------------------------------
# RFM Construction
# ------------------------------------------------------------

rfm = df_clean.groupby(
    "CustomerID"
).agg({

    "InvoiceDate": lambda x:
        (snapshot_date - x.max()).days,

    "InvoiceNo": "nunique",

    "Revenue": "sum"

})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

print("\nRFM Dataset")
print(rfm.head())

# ------------------------------------------------------------
# Scale Features
# ------------------------------------------------------------

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm
)

# ============================================================
# FIND BEST NUMBER OF CLUSTERS
# ============================================================

print("\nFinding Best K...")

inertia = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)

    inertia.append(
        model.inertia_
    )

plt.figure(figsize=(10,6))

plt.plot(
    range(2,11),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("Inertia")

plt.show()

# ============================================================
# KMEANS CLUSTERING
# ============================================================

print("\nTraining KMeans...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

# ------------------------------------------------------------
# Silhouette Score
# ------------------------------------------------------------

score = silhouette_score(
    rfm_scaled,
    rfm["Cluster"]
)

print("\nSilhouette Score:")
print(score)

# ------------------------------------------------------------
# Cluster Summary
# ------------------------------------------------------------

cluster_summary = (
    rfm.groupby("Cluster")
    .mean()
)

print("\nCluster Summary")
print(cluster_summary)

# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\nRunning PCA...")

pca = PCA(
    n_components=2
)

rfm_pca = pca.fit_transform(
    rfm_scaled
)

rfm["PCA1"] = rfm_pca[:,0]
rfm["PCA2"] = rfm_pca[:,1]

plt.figure(figsize=(10,8))

sns.scatterplot(
    data=rfm,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2"
)

plt.title(
    "Customer Segments Using PCA"
)

plt.show()

# ============================================================
# ALTERNATIVE CLUSTERING
# ============================================================

print("\nTesting Alternative Models")

# ------------------------------------------------------------
# Hierarchical Clustering
# ------------------------------------------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=4
)

hier_labels = hierarchical.fit_predict(
    rfm_scaled
)

hier_score = silhouette_score(
    rfm_scaled,
    hier_labels
)

print("\nHierarchical Silhouette Score:")
print(hier_score)

# ------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

db_labels = dbscan.fit_predict(
    rfm_scaled
)

unique_clusters = len(
    set(db_labels)
)

print("\nDBSCAN Clusters:")
print(unique_clusters)

# ============================================================
# SAVE CUSTOMER SEGMENTS
# ============================================================

rfm.to_csv(
    "customer-segments.csv"
)

print("\nSaved:")
print("customer-segments.csv")

# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

print("\n=================================================")
print("MODEL TYPE")
print("=================================================")

print("Machine Learning Category:")
print("UNSUPERVISED LEARNING")

print("\nAlgorithm:")
print("CUSTOMER SEGMENTATION (CLUSTERING)")

print("\nNeed Class Balancing?")
print("NO")

print("\nNeed SMOTE?")
print("NO")

print("\nReason:")
print("There is no target variable; this is clustering.")

print("\n=================================================")
print("RECOMMENDED IMPROVEMENTS")
print("=================================================")

print("""
1. Remove outliers using IQR.
2. Log-transform Monetary values.
3. Use RobustScaler.
4. Try Gaussian Mixture Models.
5. Try HDBSCAN.
6. Add Customer Lifetime Value features.
7. Add Average Order Value.
8. Add Basket Size.
9. Add Products Per Order.
10. Evaluate cluster stability.
""")

print("\nPipeline Complete.")