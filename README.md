# Stock Price Predictor

## Overview

This project predicts the next day's closing price of a stock based on historical stock data and news sentiment analysis. It uses machine learning models such as Linear Regression, Random Forest, and XGBoost to make these predictions.

## Features

- **Stock Data Features**: Open, High, Low, Close, Volume
- **Technical Indicators**: 
  - Moving Averages (MA10, MA50)
  - Exponential Moving Averages (EMA10, EMA50)
  - Relative Strength Index (RSI)
  - Lag features (Lag1)
- **Sentiment Analysis**: Sentiment of news headlines using VADER

## Models Used

- **Linear Regression** (Baseline model)
- **Random Forest Regressor** (With hyperparameter tuning)
- **XGBoost Regressor**

## Results

The Mean Absolute Error (MAE) for each model is:
- **Linear Regression**: 2.04
- **Random Forest (Default)**: 2.28
- **Random Forest (Tuned)**: 2.28
- **XGBoost**: 2.33

### Insights:
The simpler **Linear Regression** model provided the best results for this specific dataset. More complex models like **Random Forest** and **XGBoost** did not significantly outperform Linear Regression, showing that a simpler model might be more suitable for this task.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/stock-price-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stock-price-predictor
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python main.py
   ```
