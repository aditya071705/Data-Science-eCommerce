# Import the necessary Libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load Data
# Load datasets
transactions = pd.read_csv('Transactions.csv')
products = pd.read_csv('Products.csv')
customers = pd.read_csv('Customers.csv')

# Step 2: Merge Data
# Create a comprehensive dataset by merging transactions, products, and customer information
transactions_products = pd.merge(transactions, products, on="ProductID", how="left")
full_data = pd.merge(transactions_products, customers, on="CustomerID", how="left")

# Step 3: Feature Engineering
# Create customer profiles with aggregated statistics
customer_profiles = (
    full_data.groupby("CustomerID")
    .agg(
        total_spent=("TotalValue", "sum"),
        total_transactions=("TransactionID", "count"),
        unique_products=("ProductID", "nunique"),
        unique_categories=("Category", "nunique"),
        avg_spent_per_transaction=("TotalValue", lambda x: x.sum() / x.count())
    )
    .reset_index()
)

# Step 4: Normalize Features
# Select numerical features for similarity calculation
numeric_features = [
    "total_spent",
    "total_transactions",
    "unique_products",
    "unique_categories",
    "avg_spent_per_transaction",
]
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(customer_profiles[numeric_features])

# Step 5: Calculate Similarity
# Use cosine similarity to find similarities between customers
similarity_matrix = cosine_similarity(normalized_features)
customer_ids = customer_profiles["CustomerID"].tolist()
similarity_df = pd.DataFrame(similarity_matrix, index=customer_ids, columns=customer_ids)

# Step 6: Get Recommendations
# Function to retrieve top 3 similar customers for a given customer ID
def get_top_3_similar(customers_df, similarity_df, customer_id):
    if customer_id not in similarity_df.index:
        return []

    top_similar = (
        similarity_df[customer_id]
        .sort_values(ascending=False)
        .iloc[1:4]  # Exclude the customer itself
    )
    return list(zip(top_similar.index, top_similar.values))

# Generate recommendations for the first 20 customers (C0001–C0020)
lookalike_results = {}
for customer_id in customer_ids[:20]:
    lookalike_results[customer_id] = get_top_3_similar(customer_profiles, similarity_df, customer_id)

# Save results to CSV
lookalike_df = pd.DataFrame(
    [{"cust_id": k, "lookalikes": v} for k, v in lookalike_results.items()]
)
lookalike_df.to_csv("Lookalike.csv", index=False)

# Step 7: Define Recommendation Function
# Function to recommend lookalikes for any customer ID
def recommend_lookalikes(customer_id):
    if customer_id not in similarity_df.index:
        return "Customer not found."

    recommendations = get_top_3_similar(customer_profiles, similarity_df, customer_id)
    return {"customer_id": customer_id, "recommendations": recommendations}

# Example Recommendation for a Customer
example_recommendation = recommend_lookalikes("C0001")
print("Example Recommendation:", example_recommendation)

# Step 8: Evaluation Criteria
# Model Accuracy and Logic
# 1. Average similarity score for all recommendations
mean_similarity = similarity_df.values[np.triu_indices_from(similarity_df.values, k=1)].mean()
print(f"Mean Similarity Score (Overall): {mean_similarity:.4f}")

# 2. Validate that top recommendations have high similarity
for customer_id, recommendations in lookalike_results.items():
    avg_score = np.mean([score for _, score in recommendations])
    print(f"Customer {customer_id} - Avg Recommendation Score: {avg_score:.4f}")

# Quality of Recommendations
# Example: Check if recommended customers align with similar behavior
# Example for first customer
first_customer_id = "C0001"
first_customer_recommendations = lookalike_results.get(first_customer_id, [])
print(f"Recommendations for {first_customer_id}: {first_customer_recommendations}")

# Results and Explanation:
# The model outputs lookalikes with similarity scores for each of the first 20 customers in `Lookalike.csv`. The function `recommend_lookalikes` can be used dynamically to retrieve recommendations for any customer.
# - Mean Similarity Score provides an overview of model accuracy.
# - Individual scores for top recommendations validate the quality of the model.