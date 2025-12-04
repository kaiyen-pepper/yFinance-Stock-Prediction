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
data = pd.read_csv('stock_data\MSFT_2013-01-01_2025-12-01_1d.csv')

# Convert date column and set as index
data['Date'] = pd.to_datetime(data['Date'])

# Convert to numeric
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Clean data
data = data.ffill().dropna()

print(f"Data shape: {data.shape}")
print(f"Date range: {data.index.min()} to {data.index.max()}")

prediction = data.loc[
    (data['Date'] > datetime(2013,1,1)) &
    (data['Date'] < datetime(2018,1,1))
]

# Prepare for the LSTM Model (Sequential)
stock_close = data.filter(["Close"])
dataset = stock_close.values #convert to numpy array
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Preprocessing Stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len] #95% of all out data

X_train, y_train = [], []


# Create a sliding window for our stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Build the Model
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3rd Layer (Dense)
model.add(keras.layers.Dense(128, activation="relu"))

# 4th Layer (Dropout)
model.add(keras.layers.Dropout(0.5))

# Final Output Layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",
              loss="mae",
              metrics=[keras.metrics.RootMeanSquaredError()])


training = model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prep the test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]


for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

# Make a Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data
train = data[:training_data_len]
test =  data[training_data_len:]
test = test.copy()
test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['Date'], train['Close'], label="Train (Actual)", color='blue')
plt.plot(test['Date'], test['Close'], label="Test (Actual)", color='orange')
plt.plot(test['Date'], test['Predictions'], label="Predictions", color='red')
plt.title("Actual vs Predicted Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()

# Print Root Mean Squared Error (RMSE) and plot training history
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print("RMSE:", rmse)
plt.figure(figsize=(12,8))
plt.plot(predictions, label='Predicted Prices')
plt.plot(y_test, label='Actual Prices')
plt.title('Predicted vs Actual Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

# Print Mean Absolute Error (MAE)
mae = np.mean(np.abs(predictions - y_test))
print("MAE:", mae)
plt.figure(figsize=(12,8))
plt.plot(training.history['loss'], label='Train Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Compare predictions to the previous day's actual price
previous_day_baseline = y_test[:-1]
actual_for_comparison = y_test[1:]
predictions_for_comparison = predictions[1:]

baseline_rmse = np.sqrt(np.mean((previous_day_baseline - actual_for_comparison)**2))
model_rmse = np.sqrt(np.mean((predictions_for_comparison - actual_for_comparison)**2))

print(f"Baseline RMSE (just predicting previous day): {baseline_rmse}")
print(f"Model RMSE: {model_rmse}")
print(f"Improvement over baseline: {baseline_rmse - model_rmse}")