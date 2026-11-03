# Online Retail Customer Segmentation

## Project Overview

This project analyzes the UCI Online Retail Dataset and applies machine learning techniques to identify customer segments based on purchasing behavior.

The analysis follows a complete data science workflow:

1. Data acquisition
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Feature engineering using RFM metrics
5. Customer segmentation using clustering algorithms
6. Model evaluation and comparison
7. Interactive visualization with Streamlit

The objective is to generate actionable customer insights that can support marketing, retention, and sales strategies.

---

## Dataset

Source: UCI Machine Learning Repository

Dataset: Online Retail

Description:

A transactional dataset containing all purchases made by a UK-based online retailer between December 2010 and December 2011.

Key variables:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

Dataset size:

- 541,909 transactions
- Thousands of unique customers
- Multiple countries represented

---

## Business Problem

Retail companies often have large customer bases with very different purchasing patterns.

Without segmentation:

- Marketing campaigns become less effective
- Retention efforts are difficult to prioritize
- High-value customers are harder to identify

This project aims to identify distinct customer groups through unsupervised machine learning.

---

## Methodology

### 1. Exploratory Data Analysis

EDA was performed to:

- Understand data distributions
- Detect missing values
- Detect duplicated records
- Identify outliers
- Understand country sales patterns
- Analyze revenue trends over time

### 2. Data Cleaning

The following cleaning rules were applied:

- Removed cancelled invoices
- Removed duplicate rows
- Removed rows with missing customer identifiers
- Removed transactions with negative quantities
- Removed transactions with non-positive prices

### 3. Feature Engineering

Revenue was created as:

Revenue = Quantity × UnitPrice

A customer-level RFM table was then generated:

#### Recency

Days since the customer's latest purchase

#### Frequency

Number of unique orders

#### Monetary

Total spending

### 4. Feature Scaling

RFM variables were standardized using StandardScaler.

### 5. Customer Segmentation

Primary algorithm:

- K-Means Clustering

Alternative models:

- Hierarchical Clustering
- DBSCAN

### 6. Cluster Evaluation

The following techniques were used:

- Elbow Method
- Silhouette Score
- PCA Visualization

---

## Machine Learning Approach

### Learning Type

Unsupervised Learning

### Task

Customer Segmentation

### Problem Category

Clustering

### Target Variable

None

Since no target variable exists, this is not a classification or regression problem.

---

## Class Imbalance Analysis

No class imbalance handling was required.

Why?

Class imbalance techniques such as:

- SMOTE
- Random Oversampling
- Undersampling

are only applicable to supervised learning problems.

Because customer segmentation is an unsupervised clustering task, no class balancing methods were used.

---

## Project Structure

├── data/
│ ├── online-retail-clean-data.csv
│
├── notebooks/
│ ├── EDA.ipynb
│
├── app.py
├── analysis.py
├── customer-segments.csv
├── requirements.txt
├── README.md

---

## Results

The clustering model successfully identified distinct customer groups based on purchasing behavior.

Examples of segment archetypes include:

- High-value loyal customers
- Frequent medium-spending customers
- Inactive customers
- Low-engagement customers

The segmentation provides a foundation for targeted marketing strategies.

---

## Streamlit Dashboard

Run:
