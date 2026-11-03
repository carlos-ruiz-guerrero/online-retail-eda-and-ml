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

Business Impact
Business Impact

Although this project was developed as an analytical exercise, the customer segmentation framework demonstrates how transactional data can be transformed into actionable business insights.

Customer-Centric Marketing

The clustering model groups customers according to their purchasing behavior, enabling more targeted marketing campaigns instead of a one-size-fits-all approach.

Potential applications include:

Personalized promotions
Customer loyalty programs
Cross-selling and upselling strategies
Product recommendation systems
Revenue Optimization

By identifying high-value customer segments, businesses can:

Prioritize retention efforts
Allocate marketing budgets more effectively
Maximize customer lifetime value (CLV)
Reduce acquisition costs through improved targeting
Churn Risk Identification

Customers with low purchasing frequency and long periods of inactivity can be flagged as potential churn risks.

This allows organizations to implement:

Re-engagement campaigns
Targeted discounts
Personalized communication strategies
Customer recovery initiatives
Data-Driven Decision Making

The project provides a repeatable analytical framework that supports strategic decisions in:

Marketing
Sales
Customer Relationship Management (CRM)
Business Intelligence
Operational Benefits

The Streamlit dashboard enables stakeholders to explore sales performance interactively without requiring technical knowledge.

Benefits include:

Faster reporting
Self-service analytics
Easier monitoring of key business metrics
Improved visibility of customer behavior patterns
Scalability

The methodology can be extended to:

Customer Lifetime Value prediction
Churn prediction models
Demand forecasting
Product recommendation engines
Real-time customer segmentation
Expected Business Outcomes

If deployed in a production environment, the solution could contribute to:

Objective	Potential ImpactCustomer Retention	Improved identification of at-risk customers
Marketing Efficiency	Better campaign targeting
Revenue Growth	Increased focus on high-value customer segments
Customer Experience	More personalized interactions
Decision Making	More data-driven commercial strategies
Key Takeaway

This project demonstrates how raw transactional data can be transformed into actionable customer intelligence through data cleaning, feature engineering, machine learning, and interactive visual analytics. The resulting segmentation framework provides a practical foundation for improving customer engagement, retention, and long-term business value.

In conclussion 
Developed an end-to-end customer segmentation solution using RFM analysis, K-Means clustering, and Streamlit visualization, transforming over 500,000 retail transactions into actionable customer insights.

---

## Streamlit Dashboard

Run:
