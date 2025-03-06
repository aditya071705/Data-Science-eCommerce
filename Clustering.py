import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
transactions = pd.read_csv('Transactions.csv')
customers = pd.read_csv('Customers.csv')

# Step 2: Aggregate Transaction Data
# Create features such as total spend, number of transactions, and unique products per customer
transaction_features = (
    transactions.groupby("CustomerID")
    .agg(
        total_spent=("TotalValue", "sum"),
        total_transactions=("TransactionID", "count"),
        unique_products=("ProductID", "nunique")
    )
    .reset_index()
)

# Step 3: Merge with Customer Profile Data
# Merge transaction features with customer profile data
customer_data = pd.merge(customers, transaction_features, on="CustomerID", how="left")
customer_data.fillna(0, inplace=True)  # Handle missing values

# Step 4: Feature Engineering
# Select relevant features for clustering
# Verify column names in Customers.csv before selecting features
print("Columns in Customers.csv:", customers.columns.tolist())
features = ["total_spent", "total_transactions", "unique_products"]
data = customer_data[features]

# Normalize the features for clustering
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Step 5: Apply Clustering
# Use K-Means clustering to segment customers
num_clusters = 5  # Example: Choose 5 clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(normalized_data)

# Add cluster labels to the customer data
customer_data["Cluster"] = kmeans.labels_

# Step 6: Evaluate Clustering Metrics
# Davies-Bouldin Index (Lower is better)
db_index = davies_bouldin_score(normalized_data, kmeans.labels_)
# Silhouette Score (Higher is better)
silhouette_avg = silhouette_score(normalized_data, kmeans.labels_)
# Calinski-Harabasz Score (Higher is better)
calinski_harabasz = calinski_harabasz_score(normalized_data, kmeans.labels_)

print(f"Davies-Bouldin Index: {db_index:.4f} (Lower is better)")
print(f"Silhouette Score: {silhouette_avg:.4f} (Higher is better)")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f} (Higher is better)")

# Step 7: Visualize Clusters
# Plot clusters using the first two features (e.g., total_spent and total_transactions)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=normalized_data[:, 0], y=normalized_data[:, 1], hue=customer_data["Cluster"], palette="viridis"
)
plt.title("Customer Segmentation Clusters")
plt.xlabel("Normalized Feature 1 (Total Spent)")
plt.ylabel("Normalized Feature 2 (Total Transactions)")
plt.legend(title="Cluster")
plt.show()

# Additional Visualizations
# Pairplot of all features with cluster labels
sns.pairplot(customer_data[features + ["Cluster"]], hue="Cluster", palette="viridis")
plt.show()

# Step 8: Save Results
# Save customer data with cluster labels
customer_data.to_csv("Customer_Segmentation.csv", index=False)

# Step 9: Evaluation Criteria
# Clustering Logic and Metrics
print("--- Clustering Metrics and Evaluation ---")
print(f"Number of Clusters: {num_clusters}")
print(f"Davies-Bouldin Index: {db_index:.4f} (Lower is better; indicates compact and well-separated clusters)")
print(f"Silhouette Score: {silhouette_avg:.4f} (Higher is better; indicates cohesion and separation)")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f} (Higher is better; indicates dense and well-separated clusters)")

# Visual Representation of Clusters
print("--- Visualizations ---")
print("1. Scatter plot showing clusters using normalized features.")
print("2. Pairplot of all features with cluster labels for detailed insights.")

# Final Output and Notes
# - The clustering results, including cluster labels for each customer, are saved in `Clustering.csv`.
# - Davies-Bouldin Index, Silhouette Score, and Calinski-Harabasz Score are calculated to evaluate the clustering performance.
# - Visualizations show how customers are segmented based on the chosen features.
