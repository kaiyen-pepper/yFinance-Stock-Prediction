# LSTM Stock Prediction Exploratory Analysis

An exploratory analysis of Long Short-Term Memory (LSTM) neural networks for stock price prediction using S&P 500 data. This project investigates different LSTM architectures, frameworks (TensorFlow/Keras and PyTorch), and preprocessing techniques to evaluate the viability of LSTMs for financial time series forecasting.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

## Overview

This project explores LSTM models for predicting stock prices, specifically using Microsoft (MSFT) S&P 500 historical data. Through multiple iterations, the project addresses common pitfalls in time series forecasting (such as data leakage) and compares different model architectures and frameworks.

**Main Goal:** Determine whether LSTM networks can effectively predict stock prices and identify best practices for financial time series modeling.

## Dataset

The dataset is sourced from the **yfinance** library and includes the following features:
- **Open**: Opening price
- **Close**: Closing price (target variable)
- **High**: Highest price during the trading day
- **Low**: Lowest price during the trading day
- **Volume**: Number of shares traded

**Primary Dataset Used:** Microsoft (MSFT) stock data from 2013-01-01 to 2025-12-01 with daily intervals.

## Project Structure

```
.
├── download_data.py          # CLI tool for downloading stock data
├── stock_data/               # Directory containing downloaded CSV files
│   └── MSFT_2013-01-01_2025-12-01_1d.csv
├── stock_plots.py            # Initial data visualization and EDA
├── LSTM_simple.py            # Initial LSTM (with data leakage issue)
├── LSTM_no_cheating.py       # Corrected LSTM with proper train/val/test split
├── LSTM_2.py                 # Enhanced Keras LSTM with dropout and early stopping
├── LSTM_pytorch.py           # PyTorch implementation for comparison
├── Final_LSTM.py             # Final improved PyTorch model
├── best_lstm_model.h5        # Saved Keras model
├── best_model.pth            # Saved PyTorch model
└── graphs/                   # Output visualizations
    ├── training_validation_loss.png
    ├── actual_vs_predictions.png
    └── zoomed_predictions.png
```

## Installation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn yfinance scikit-learn tensorflow torch
```

### Full Dependencies

<details>
<summary>Click to expand complete package list</summary>

- beautifulsoup4 (4.14.3)
- pandas (2.3.3)
- numpy (2.3.5)
- matplotlib (3.10.7)
- seaborn (0.13.2)
- yfinance (0.2.66)
- scikit-learn (1.7.2)
- torch (2.9.1)
- TensorFlow/Keras (via pip install tensorflow)

</details>

## Usage

### 1. Download Stock Data

Use the CLI tool to download historical stock data:

```bash
python download_data.py --tickers MSFT AAPL GOOGL \
                        --start_date 2013-01-01 \
                        --end_date 2025-12-01 \
                        --interval 1d \
                        --output_dir stock_data
```

**Arguments:**
- `--tickers`: Space-separated list of stock ticker symbols
- `--start_date`: Start date in YYYY-MM-DD format
- `--end_date`: End date in YYYY-MM-DD format
- `--interval`: Data interval (default: 1d) - options: 1d, 1h, 15m, etc.
- `--output_dir`: Directory to save CSV files (default: stock_data)

### 2. Visualize Data

Explore the dataset distribution, correlation, and closing prices:

```bash
python stock_plots.py
```

### 3. Run LSTM Models

Each script runs independently and generates visualizations:

```bash
# Initial LSTM (demonstrates data leakage problem)
python LSTM_simple.py

# Corrected LSTM with proper data splitting
python LSTM_no_cheating.py

# Enhanced Keras LSTM with MinMaxScaler and dropout
python LSTM_2.py

# PyTorch implementation
python LSTM_pytorch.py

# Final improved PyTorch model
python Final_LSTM.py
```

All scripts automatically load `stock_data/MSFT_2013-01-01_2025-12-01_1d.csv` and generate prediction plots.

## Methodology

### Model Evolution

#### 1. LSTM_simple.py (Baseline with Data Leakage)
- **Issue Identified:** Training data included future values due to incorrect slicing
- **Problem Line:** `training_data = scaled_data[:training_data_len]`
- **Result:** ~98% accuracy, but predictions "lagged" actual values (model was "cheating")
- **Architecture:** Simple 2-layer LSTM (64 neurons) + Dense layer (128 neurons, ReLU) + Output layer

#### 2. LSTM_no_cheating.py (Fixed Data Leakage)
- **Improvement:** Proper train/validation/test split by year
- **Scaler:** StandardScaler
- **Result:** RMSE ~500, Model improvement vs naive baseline: **-2400%**
- **Conclusion:** Fixing data leakage revealed poor actual performance

#### 3. LSTM_2.py (Enhanced Keras Model)
- **Key Changes:**
  - MinMaxScaler instead of StandardScaler (better for skewed distributions)
  - Added Dropout layers (0.2) for regularization
  - Early stopping with patience=10 on validation loss
- **Architecture:**
```python
Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 5)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1)
])
```
- **Results:** Improved generalization with dropout and early stopping

#### 4. LSTM_pytorch.py (PyTorch Comparison)
- **Purpose:** Compare TensorFlow/Keras vs PyTorch implementation
- **Architecture:**
```python
LSTM(input_size=5, hidden_size=50, num_layers=1) + Linear(50, 1)
```
- **Training:** 1000 epochs, batch_size=8, Adam optimizer, MSELoss
- **Results:** Similar performance to Keras (RMSE ~500, MAE ~100)
- **Insight:** Framework choice had minimal impact on results

#### 5. Final_LSTM.py (Best Model)
- **Enhancements:**
  - Increased dropout to 0.5 for stronger regularization
  - Reduced hidden size to 32 (prevent overfitting)
  - Larger batch size (32) for stable training
  - Weight decay (1e-4) for L2 regularization
  - Early stopping (patience=10)
  - Takes only last timestep from LSTM output
- **Architecture:**
```python
ImprovedLSTM(
    input_size=5,
    hidden_size=32,
    num_layers=1,
    dropout=0.5
)
```
- **Results:** 
  - Test RMSE: ~100
  - Training RMSE: <10
  - **Issue:** Significant overfitting despite regularization
  - Model improvement vs naive: **-900%**

## Results

### Performance Metrics Summary

| Model | Test RMSE | MAE | Model Improvement | Key Issue |
|-------|-----------|-----|-------------------|-----------|
| LSTM_simple | Low (~2%) | N/A | N/A | Data leakage (invalid) |
| LSTM_no_cheating | ~500 | N/A | -2400% | Poor generalization |
| LSTM_2 | ~500 | N/A | N/A | Baseline with dropout |
| LSTM_pytorch | ~500 | ~100 | N/A | Framework comparison |
| Final_LSTM | ~100 | N/A | -900% | Overfitting (train RMSE <10) |

### Visualizations

The `Final_LSTM.py` script generates three key plots in the `/graphs` directory:

1. **Training and Validation Loss Over Time**: Shows learning progression and early stopping point
2. **MSFT Stock Price: Actual vs Predictions**: Full test set comparison
3. **Test Set Predictions: Zoomed**: Detailed view of prediction accuracy

## Key Findings

### What We Learned

1. **Data Leakage is Critical**: The initial model's 98% accuracy was entirely due to training on future data, demonstrating the importance of proper temporal splitting in time series

2. **LSTMs Struggle with Stock Prediction**: 
   - Even with proper methodology, models showed negative improvement over naive baselines
   - Best model improvement: -900% (performed worse than simply predicting previous day's closing price)
   - Market variability and noise make accurate prediction extremely difficult

3. **Overfitting is Persistent**:
   - Despite dropout (0.5), weight decay, early stopping, and reduced model capacity
   - Training RMSE <10 vs Test RMSE ~100 shows models memorize training patterns but fail to generalize

4. **Framework Choice is Secondary**: TensorFlow/Keras and PyTorch yielded similar results (~500 RMSE), suggesting the fundamental challenge lies in the problem itself, not implementation

5. **Feature Engineering Matters**: Using MinMaxScaler for skewed distributions improved stability over StandardScaler

### Why LSTMs Fail for Stock Prediction

- **Market Efficiency**: Stock prices incorporate all available information, making patterns difficult to exploit
- **High Noise-to-Signal Ratio**: Daily price movements are heavily influenced by unpredictable external factors
- **Non-Stationary Data**: Financial markets exhibit regime changes that violate stationarity assumptions
- **Temporal Dependencies are Weak**: Unlike language or sensor data, stock prices don't have strong sequential dependencies that LSTMs excel at capturing

## Future Work

Based on this exploratory analysis, alternative approaches may yield better results:

1. **Random Forests / XGBoost**: Tree-based models with engineered features (technical indicators, moving averages)
2. **Transformers**: Attention mechanisms to capture long-range dependencies
3. **Simple Regression**: Linear models with carefully selected features may outperform complex neural networks
4. **Hybrid Approaches**: Combine technical analysis indicators with machine learning models
5. **Classification Instead of Regression**: Predict direction (up/down) rather than exact prices
6. **Incorporate External Data**: News sentiment, market indicators, macroeconomic factors

## Conclusion

This project demonstrates that **LSTMs are not effective for stock price prediction** given market variability and the efficient market hypothesis. While LSTMs excel in many sequence modeling tasks (NLP, speech recognition, sensor data), financial time series present unique challenges that require different approaches.

The journey from a "98% accurate" but fundamentally flawed model to a properly validated model with negative improvement highlights the critical importance of:
- Rigorous evaluation methodology
- Awareness of data leakage
- Realistic baseline comparisons
- Understanding problem domain constraints

---

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: [yfinance](https://github.com/ranaroussi/yfinance)
- Inspired by TensorFlow/Keras stock prediction tutorials
- Developed as part of coursework on deep learning and time series analysis