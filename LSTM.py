from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations 

# Load and preprocess data
data = pd.read_csv('stock_data\MSFT_2020-01-01_2025-12-01_1d.csv')

# Convert date column and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Convert to numeric
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Clean data
data = data.ffill().dropna()

print(f"Data shape: {data.shape}")
print(f"Date range: {data.index.min()} to {data.index.max()}")