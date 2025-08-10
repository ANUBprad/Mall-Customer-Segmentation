# Mall-Customer-Segmentation
# Customer Segmentation Analysis

## Overview
This project performs customer segmentation using K-Means clustering on the [Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle. It identifies distinct customer groups based on spending behavior and demographics.

## Dataset
The dataset contains 200 records with:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Features
- **Jupyter Notebook**: Complete analysis pipeline
  - Exploratory Data Analysis
  - Feature scaling
  - Elbow method for optimal clusters
  - 2D/3D visualizations
- **Streamlit App**: Interactive dashboard
  - Adjustable cluster count
  - Feature selection
  - Real-time visualization
  - Cluster statistics

## Technologies Used
- Python 3.8+
- Scikit-learn (KMeans, StandardScaler)
- Pandas/Numpy (Data manipulation)
- Matplotlib/Seaborn (Visualization)
- Streamlit (Interactive dashboard)
- Jupyter (Analysis notebook)

## Installation
```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
