import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

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

# Prepare for the LSTM Model (Sequential)
X = data.filter(["Close"])
y = data['Close']
print(X.head())
print(y.head())

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
SEQUENCE_LENGTH = 30
def create_sequences(features, labels, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(features)):
        X_seq.append(features.iloc[i-window_size:i].values)
        y_seq.append(labels.iloc[i])
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(train_X_scaled, train_labels, SEQUENCE_LENGTH)
X_val, y_val = create_sequences(val_X_scaled, val_labels, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(test_X_scaled, test_labels, SEQUENCE_LENGTH)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Build the Model
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(30, len(X.columns))),
    Dropout(.2),
    LSTM(64, return_sequences=False),
    Dropout(.2),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(
    loss='mse', 
    optimizer=Adam(0.001), 
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

# Display model summary
print("MODEL ARCHITECTURE SUMMARY")
model.summary()


# Fit the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the Model
test_loss, test_rmse = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test RMSE: {test_rmse}")

# Print Root Mean Squared Error (RMSE) and plot training history
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print("RMSE:", rmse)
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()