import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess data
df = pd.read_csv('stock_data/MSFT_2013-01-01_2025-12-01_1d.csv')

# Convert date column and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Convert to numeric
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean data
df = df.ffill().dropna()
print("=" * 70)
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Prepare features and target
X = df.copy()
y = df['Close'].copy()

# Define date ranges for splitting
TRAIN_END = datetime(2023, 12, 31)
VAL_END = datetime(2024, 12, 31)
TEST_END = datetime(2025, 12, 1)

train_X = X[X.index <= TRAIN_END].copy()
train_labels = y[y.index <= TRAIN_END].copy()
val_X = X[(X.index > TRAIN_END) & (X.index <= VAL_END)].copy()
val_labels = y[(y.index > TRAIN_END) & (y.index <= VAL_END)].copy()
test_X = X[(X.index > VAL_END) & (X.index < TEST_END)].copy()
test_labels = y[(y.index > VAL_END) & (y.index < TEST_END)].copy()

print("=" * 70)
print(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")

# Scale features
scaler = MinMaxScaler()
scaler.fit(train_X)

train_X_scaled = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns, index=train_X.index)
val_X_scaled = pd.DataFrame(scaler.transform(val_X), columns=val_X.columns, index=val_X.index)
test_X_scaled = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns, index=test_X.index)

# Create sequences
WINDOW_SIZE = 15

def create_sequences(features, labels, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(features)):
        X_seq.append(features.iloc[i-window_size:i].values)
        y_seq.append(labels.iloc[i])
    return torch.tensor(np.array(X_seq)).float(), torch.tensor(np.array(y_seq)).float()

X_train, y_train = create_sequences(train_X_scaled, train_labels, WINDOW_SIZE)
X_val, y_val = create_sequences(val_X_scaled, val_labels, WINDOW_SIZE)
X_test, y_test = create_sequences(test_X_scaled, test_labels, WINDOW_SIZE)

print("=" * 70)
print(f"Training sequences: {X_train.shape}, {y_train.shape}")
print(f"Validation sequences: {X_val.shape}, {y_val.shape}")
print(f"Test sequences: {X_test.shape}, {y_test.shape}")

# Improved LSTM Model
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take only the last timestep
        last_timestep = lstm_out[:, -1, :]
        x = self.dropout(last_timestep)
        x = self.linear(x)
        return x.squeeze()

# Initialize model
model = ImprovedLSTM(input_size=5, hidden_size=32, num_layers=1, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# Create data loaders
train_loader = data.DataLoader(
    data.TensorDataset(X_train, y_train),
    shuffle=True,
    batch_size=32
)
val_loader = data.DataLoader(
    data.TensorDataset(X_val, y_val),
    shuffle=False,
    batch_size=32
)

# Training with early stopping
n_epochs = 200
patience = 10
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

print("=" * 70)
print("Starting training...")
print("=" * 70)

for epoch in range(n_epochs):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Print progress
    if epoch % 10 == 0:
        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        print(f"Epoch {epoch:3d}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation function
def evaluate_model(model, X, y, set_name="Test"):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    
    actuals = y.numpy()
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)
    
    # Directional accuracy
    actual_direction = np.diff(actuals) > 0
    pred_direction = np.diff(predictions) > 0
    directional_acc = np.mean(actual_direction == pred_direction) * 100
    
    print("=" * 70)
    print(f"{set_name} Set Evaluation Metrics:")
    print("-" * 70)
    print(f"MAE (Mean Absolute Error):        ${mae:.2f}")
    print(f"RMSE (Root Mean Squared Error):   ${rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error):     {mape:.2f}%")
    print(f"R² Score:                         {r2:.4f}")
    print(f"Directional Accuracy:             {directional_acc:.2f}%")
    print("=" * 70)
    
    # Write results to CSV
    results_df = pd.DataFrame({
        'Set': [set_name],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape],
        'R2_Score': [r2],
        'Directional_Accuracy': [directional_acc]
    })
    if not os.path.exists('model_evaluation_results.csv'):
        results_df.to_csv('model_evaluation_results.csv', index=False)
    else:
        results_df.to_csv('model_evaluation_results.csv', mode='a', header=False, index=False)

    return predictions, actuals

# Evaluate on all sets
print("\n")
train_preds, train_actuals = evaluate_model(model, X_train, y_train, "Training")
val_preds, val_actuals = evaluate_model(model, X_val, y_val, "Validation")
test_preds, test_actuals = evaluate_model(model, X_test, y_test, "Test")

# Naive baseline comparison (predict yesterday's price)
naive_predictions = test_actuals[:-1]
naive_actuals = test_actuals[1:]
naive_mae = mean_absolute_error(naive_actuals, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(naive_actuals, naive_predictions))

print("\nNaive Baseline (Predict Yesterday's Price):")
print(f"MAE:  ${naive_mae:.2f}")
print(f"RMSE: ${naive_rmse:.2f}")
print(f"\nModel Improvement over Naive Baseline:")
print(f"MAE:  {((naive_mae - mean_absolute_error(test_actuals, test_preds)) / naive_mae * 100):.2f}%")
print(f"RMSE: {((naive_rmse - np.sqrt(mean_squared_error(test_actuals, test_preds))) / naive_rmse * 100):.2f}%")

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Training history
axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
axes[0].plot(val_losses, label='Validation Loss', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training and Validation Loss Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Full predictions vs actuals
full_predictions = np.full(len(y), np.nan)
full_predictions[WINDOW_SIZE:WINDOW_SIZE+len(train_preds)] = train_preds
full_predictions[WINDOW_SIZE+len(train_preds)+len(val_preds):WINDOW_SIZE+len(train_preds)+len(val_preds)+len(test_preds)] = test_preds

axes[1].plot(y.index, y.values, label='Actual Close Price', linewidth=2)
axes[1].plot(y.index[WINDOW_SIZE:WINDOW_SIZE+len(train_preds)], train_preds, 
             label='Train Predictions', alpha=0.7, linewidth=1)
axes[1].plot(y.index[WINDOW_SIZE+len(train_preds)+len(val_preds):WINDOW_SIZE+len(train_preds)+len(val_preds)+len(test_preds)], 
             test_preds, label='Test Predictions', alpha=0.7, linewidth=1)
axes[1].axvline(x=TRAIN_END, color='gray', linestyle='--', alpha=0.5, label='Train End')
axes[1].axvline(x=VAL_END, color='gray', linestyle='--', alpha=0.5, label='Val End')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price ($)')
axes[1].set_title('MSFT Stock Price: Actual vs Predictions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Test set zoom
test_indices = y.index[WINDOW_SIZE+len(train_preds)+len(val_preds):WINDOW_SIZE+len(train_preds)+len(val_preds)+len(test_preds)]
axes[2].plot(test_indices, test_actuals, label='Actual', linewidth=2, marker='o', markersize=3)
axes[2].plot(test_indices, test_preds, label='Predicted', linewidth=2, marker='s', markersize=3, alpha=0.7)
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Price ($)')
axes[2].set_title('Test Set Predictions (Zoomed)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
try:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_title = now + ".png"
    plt.savefig(figure_title, dpi=300, bbox_inches='tight')
    print("\nPlot saved successfully as 'stock_prediction_results.png'")
except Exception as e:
    print(f"Error saving plot: {e}")
plt.show()

# Save evaluations to CSV
