import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
data = pd.read_csv('stock_data\MSFT_2020-01-01_2025-12-01_1d.csv')

# Display basic information about the dataset
print(data.head())
print(data.info())
print(data.describe())

# Convert date column
data['Date'] = pd.to_datetime(data['Date'])

# Initial Data Visualization
# Plot 1 - Open and Close Prices of time
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data['Date'], data['Open'], label="Open", color="blue")
ax.plot(data['Date'], data['Close'], label="Close", color="red")
ax.set_title("Open-Close Price over Time")
ax.legend()
# plt.show()

# Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['Date'],data['Volume'], color="orange")
plt.title("Stock Volume over Time")
# plt.show()


# Drop non-numeric columns
numeric_data = data.select_dtypes(include=["int64","float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show()

# Plot 4 - Distribution of Features
fig, ax = plt.subplots(2, 3, figsize=(12,8))
features = numeric_data.columns
for i, feature in enumerate(features):
    sns.histplot(numeric_data[feature], bins=30, kde=True, ax=ax[i//3, i%3])
    ax[i//3, i%3].set_title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()