import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

plt.figure(figsize=(12,6))
plt.plot(data.index, data['Close'],color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")

# Prepare for the LSTM Model (Sequential)
X = data.filter(["Close"])
y = data['Close']
print(X.head())
print(y.head())

# Define date ranges for splitting
TRAIN_END = datetime(2023,12,31)
VAL_END = datetime(2024,12,31)
TEST_END = datetime(2025,12,1)
train_data = X[X.index <= TRAIN_END].copy()
train_labels = y[y.index <= TRAIN_END].copy()
val_data = X[(X.index > TRAIN_END) & (X.index <= VAL_END)].copy()
val_labels = y[(y.index > TRAIN_END) & (y.index <= VAL_END)].copy()
test_data = X[(X.index > VAL_END) & (X.index < TEST_END)].copy()
test_labels = y[(y.index > VAL_END) & (y.index < TEST_END)].copy()
print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
print(f"Validation data shape: {val_data.shape}, Validation labels shape: {val_labels.shape}")
print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

# Preprocessing Stages
scaler = StandardScaler()
train_data_scaled = pd.DataFrame(
    scaler.fit_transform(train_data),
    columns=train_data.columns,
    index=train_data.index
)
val_data_scaled = pd.DataFrame(
    scaler.transform(val_data),
    columns=val_data.columns,
    index=val_data.index
)
test_data_scaled = pd.DataFrame(
    scaler.transform(test_data),
    columns=test_data.columns,
    index=test_data.index
)

# Create sequences
def create_sequences(features, labels, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(features)):
        X_seq.append(features.iloc[i-window_size:i].values)
        y_seq.append(labels.iloc[i])
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(train_data_scaled, train_labels, 30)
X_val, y_val = create_sequences(val_data_scaled, val_labels, 30)
X_test, y_test = create_sequences(test_data_scaled, test_labels, 30)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(30, len(X.columns))),
    Dropout(.2),
    LSTM(64, return_sequences=False),
    Dropout(.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Display model summary
print("="*60)
print("MODEL ARCHITECTURE SUMMARY")
print("="*60)
model.summary()

# Visualize model architecture
try:
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='models/model_architecture.png', show_shapes=True, 
               show_layer_names=True, rankdir='TB', dpi=150)
    print("\n✓ Model architecture diagram saved to models/model_architecture.png")
except Exception as e:
    print(f"\nNote: Could not generate architecture diagram: {e}")

# Create custom visualization using matplotlib
fig, ax = plt.subplots(figsize=(10, 12))

# Define layer positions and sizes
y_pos = 0.95
layer_height = 0.12
spacing = 0.15

# Input layer
ax.add_patch(plt.Rectangle((0.35, y_pos - 0.05), 0.3, 0.05, 
                           facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(0.5, y_pos - 0.025, f'Input\n({30}, {len(X.columns)})', 
        ha='center', va='center', fontsize=10, fontweight='bold')
y_pos -= spacing

# LSTM Layer 1
ax.add_patch(plt.Rectangle((0.3, y_pos - 0.05), 0.4, 0.05, 
                           facecolor='lightgreen', edgecolor='black', linewidth=2))
ax.text(0.5, y_pos - 0.025, f'LSTM Layer 1\n{32} units\n(return_sequences=True)', 
        ha='center', va='center', fontsize=9, fontweight='bold')
y_pos -= spacing

# Dropout 1
ax.add_patch(plt.Rectangle((0.35, y_pos - 0.04), 0.3, 0.04, 
                           facecolor='lightyellow', edgecolor='black', linewidth=2))
ax.text(0.5, y_pos - 0.02, f'Dropout\n{.2*100:.0f}%', 
        ha='center', va='center', fontsize=9, fontweight='bold')
y_pos -= spacing

# LSTM Layer 2
ax.add_patch(plt.Rectangle((0.3, y_pos - 0.05), 0.4, 0.05, 
                           facecolor='lightgreen', edgecolor='black', linewidth=2))
ax.text(0.5, y_pos - 0.025, f'LSTM Layer 2\n{64} units\n(return_sequences=False)', 
        ha='center', va='center', fontsize=9, fontweight='bold')
y_pos -= spacing

# Dropout 2
ax.add_patch(plt.Rectangle((0.35, y_pos - 0.04), 0.3, 0.04, 
                           facecolor='lightyellow', edgecolor='black', linewidth=2))
ax.text(0.5, y_pos - 0.02, f'Dropout\n{.2*100:.0f}%', 
        ha='center', va='center', fontsize=9, fontweight='bold')
y_pos -= spacing

# Dense Layer 1
ax.add_patch(plt.Rectangle((0.35, y_pos - 0.05), 0.3, 0.05, 
                           facecolor='lightcoral', edgecolor='black', linewidth=2))
ax.text(0.5, y_pos - 0.025, f'Dense Layer\n{32} units\n(ReLU)', 
        ha='center', va='center', fontsize=9, fontweight='bold')
y_pos -= spacing

# Output Layer
ax.add_patch(plt.Rectangle((0.35, y_pos - 0.05), 0.3, 0.05, 
                           facecolor='lightpink', edgecolor='black', linewidth=2))
ax.text(0.5, y_pos - 0.025, 'Output Layer\n1 unit\n(Sigmoid)', 
        ha='center', va='center', fontsize=9, fontweight='bold')

# Draw arrows between layers
arrow_props = dict(arrowstyle='->', lw=2, color='black')
# Calculate precise arrow positions: from bottom of one layer to top of next
# Input layer: y=0.95, height=0.05, bottom=0.90
# LSTM1: y=0.80, height=0.05, top=0.80, bottom=0.75
# Dropout1: y=0.65, height=0.04, top=0.65, bottom=0.61
# LSTM2: y=0.50, height=0.05, top=0.50, bottom=0.45
# Dropout2: y=0.35, height=0.04, top=0.35, bottom=0.31
# Dense: y=0.20, height=0.05, top=0.20, bottom=0.15
# Output: y=0.05, height=0.05, top=0.05

arrow_starts = [0.90, 0.75, 0.61, 0.45, 0.31, 0.15]  # Bottom of each layer
arrow_ends = [0.80, 0.65, 0.50, 0.35, 0.20, 0.05]    # Top of next layer

for y_start, y_end in zip(arrow_starts, arrow_ends):
    ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start), 
                arrowprops=arrow_props)

# Add title and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('LSTM Model Architecture for Volatility Prediction', 
             fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='Input'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='LSTM Layer'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightyellow', edgecolor='black', label='Dropout'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='black', label='Dense Layer'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightpink', edgecolor='black', label='Output')
]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
          ncol=5, fontsize=9)

plt.tight_layout()
plt.show()