import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations 

# Load and preprocess data
df = pd.read_csv('stock_data\MSFT_2013-01-01_2025-12-01_1d.csv')

# Convert date column and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Convert to numeric
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean data
df = df.ffill().dropna()
print("-" * 50)
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Prepare for the LSTM Model (Sequential)
X = df.copy()
y = df['Close'].copy()
# Print first few rows
print("-" * 50)
print("Features:", X.head())
print("Target:", y.head())

# Define date ranges for splitting
TRAIN_END = datetime(2023,12,31)
VAL_END = datetime(2024,12,31)
TEST_END = datetime(2025,12,1)
train_X = X[X.index <= TRAIN_END].copy()
train_labels = y[y.index <= TRAIN_END].copy()
val_X = X[(X.index > TRAIN_END) & (X.index <= VAL_END)].copy()
val_labels = y[(y.index > TRAIN_END) & (y.index <= VAL_END)].copy()
test_X = X[(X.index > VAL_END) & (X.index < TEST_END)].copy()
test_labels = y[(y.index > VAL_END) & (y.index < TEST_END)].copy()
# Print shapes
print("-" * 50)
print(f"Train data shape: {train_X.shape}, Train labels shape: {train_labels.shape}")
print(f"Validation data shape: {val_X.shape}, Validation labels shape: {val_labels.shape}")
print(f"Test data shape: {test_X.shape}, Test labels shape: {test_labels.shape}")

# Preprocessing Stages
# Scale features
scaler = MinMaxScaler()
scaler.fit(train_X)  # learn from training data only
train_X_scaled = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns, index=train_X.index)
val_X_scaled = pd.DataFrame(scaler.transform(val_X), columns=val_X.columns, index=val_X.index)
test_X_scaled = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns, index=test_X.index)

# Create sequences
WINDOW_SIZE = 30
def create_sequences(features, labels, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(features)):
        X_seq.append(features.iloc[i-window_size:i].values)
        y_seq.append(labels.iloc[i])
    return torch.tensor(X_seq).float(), torch.tensor(y_seq).float()

X_train, y_train = create_sequences(train_X_scaled, train_labels, WINDOW_SIZE)
X_val, y_val = create_sequences(val_X_scaled, val_labels, WINDOW_SIZE)
X_test, y_test = create_sequences(test_X_scaled, test_labels, WINDOW_SIZE)
# Print shapes
print("-" * 50)
print("Training shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape", X_test.shape, y_test.shape)

class TorchLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=1, batch_first=True) # 5 features, 50 hidden units, 1 layer
        self.linear = nn.Linear(50, 1) # Input features: 50, output features: 1 (Close price)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
 
model = TorchLSTM()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
 
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 20 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
 
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(y) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[WINDOW_SIZE:WINDOW_SIZE+len(X_train)] = model(X_train)[:, -1, :].squeeze()
    test_plot = np.ones_like(y) * np.nan
    test_plot[len(X_train)+WINDOW_SIZE:len(y)] = model(X_test)[:, -1, :]
# plot
plt.plot(y, label='Actual Close Price')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()