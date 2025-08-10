# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('Mall_Customers.csv')
    return data

data = load_data()

# Sidebar controls
st.sidebar.header("Controls")
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)
features = st.sidebar.multiselect(
    "Select features for clustering",
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    ['Annual Income (k$)', 'Spending Score (1-100)']
)

# Main app
st.title("🛍️ Customer Segmentation Dashboard")
st.write("""
This app performs customer segmentation using K-Means clustering.
Select different features and number of clusters using the sidebar controls.
""")

# Prepare data
if not features:
    st.warning("Please select at least one feature")
    st.stop()

X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

# Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(data)

# Visualizations
st.subheader("Cluster Visualizations")

# 2D Scatter plot
if len(features) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_clusters):
        ax.scatter(
            X[clusters == i][features[0]],
            X[clusters == i][features[1]],
            label=f'Cluster {i}'
        )
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200, c='black', marker='X', label='Centroids'
    )
    ax.set_title(f'Clusters by {features[0]} and {features[1]}')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Select at least 2 features for scatter plot")

# Cluster characteristics
st.subheader("Cluster Characteristics")
cluster_summary = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Gender': lambda x: x.mode()[0],
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})
st.dataframe(cluster_summary)

# Prediction section
st.subheader("Predict New Customer")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
with col2:
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
with col3:
    spending = st.number_input("Spending Score", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):
    input_data = [[income, spending]] if 'Age' not in features else [[age, income, spending]]
    scaled_input = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_input)[0]
    st.success(f"This customer belongs to Cluster {cluster}")
    st.write(cluster_summary.loc[cluster])