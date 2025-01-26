# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers_path = 'Customers.csv'
products_path = 'Products.csv'
transactions_path = 'Transactions.csv'

customers_df = pd.read_csv(customers_path)
products_df = pd.read_csv(products_path)
transactions_df = pd.read_csv(transactions_path)

# Convert date columns to datetime for proper analysis
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Merge datasets for EDA
transactions_customers = pd.merge(transactions_df, customers_df, on='CustomerID', how='inner')
full_data = pd.merge(transactions_customers, products_df, on='ProductID', how='inner')

# Set a theme for the visualizations
sns.set_theme(style="whitegrid")

# EDA: Customer Analysis
# Distribution of Customers by Region
plt.figure(figsize=(8, 5))
sns.countplot(data=customers_df, x="Region", palette="pastel", order=customers_df['Region'].value_counts().index)
plt.title("Number of Customers by Region")
plt.xlabel("Region")
plt.ylabel("Count of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# EDA: Product Analysis
# Top 10 Most Expensive Products
top_expensive_products = products_df.nlargest(10, 'Price')
plt.figure(figsize=(10, 6))
sns.barplot(data=top_expensive_products, x="Price", y="ProductName", palette="viridis")
plt.title("Top 10 Most Expensive Products")
plt.xlabel("Price")
plt.ylabel("Product Name")
plt.tight_layout()
plt.show()

# EDA: Transaction Analysis
# Total Sales Over Time
transactions_df['MonthYear'] = transactions_df['TransactionDate'].dt.to_period('M').astype(str)
monthly_sales = transactions_df.groupby('MonthYear')['TotalValue'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x="MonthYear", y="TotalValue", marker="o", color="b")
plt.title("Total Sales Over Time")
plt.xlabel("Month-Year")
plt.ylabel("Total Sales Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Most Popular Product Categories
popular_categories = full_data.groupby('Category')['Quantity'].sum().reset_index().sort_values(by='Quantity', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=popular_categories, x="Quantity", y="Category", palette="coolwarm")
plt.title("Most Popular Product Categories")
plt.xlabel("Quantity Sold")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# Top Customers by Spending
top_customers = full_data.groupby('CustomerName')['TotalValue'].sum().reset_index().nlargest(10, 'TotalValue')

plt.figure(figsize=(10, 6))
sns.barplot(data=top_customers, x="TotalValue", y="CustomerName", palette="mako")
plt.title("Top 10 Customers by Spending")
plt.xlabel("Total Spending")
plt.ylabel("Customer Name")
plt.tight_layout()
plt.show()
