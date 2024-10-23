import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fetch historical stock price data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    # Let's get the last 5 years of stock data
    data = stock.history(period="5y")
    # Select relevant columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Moving Average (10-day and 50-day)
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Exponential Moving Average (EMA)
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # RSI (Relative Strength Index)
    data['RSI'] = calculate_rsi(data['Close'], window=14)

    # Lag Features (1-day lag)
    data['Lag1'] = data['Close'].shift(1)
    
    # Create a column for the next day's closing price (our target)
    data['Next Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)  # Drop rows with NaN values (due to lags/indicators)
    return data

# Analyze sentiment for a given headline
def analyze_sentiment(headline):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(headline)
    return sentiment_score

# Example: Fetching data for Apple (AAPL)
stock_data = get_stock_data('AAPL')

# Mock sentiment integration (for demonstration purposes)
sample_headline = "Apple launches new iPhone with groundbreaking features"
sentiment = analyze_sentiment(sample_headline)
# Add the compound sentiment score to the stock data
stock_data['Sentiment'] = sentiment['compound']

# Prepare the features (X) and the target (y)
X = stock_data[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'EMA10', 'EMA50', 'RSI', 'Lag1', 'Sentiment']]
y = stock_data['Next Close']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression (Baseline) ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f"Linear Regression MAE: {mae_lr}")

# --- Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest MAE: {mae_rf}")

# --- Hyperparameter Tuning for Random Forest ---
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid_rf, cv=3, scoring='neg_mean_absolute_error')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf_best = best_rf.predict(X_test)
mae_rf_best = mean_absolute_error(y_test, y_pred_rf_best)
print(f"Best Random Forest MAE after tuning: {mae_rf_best}")

# --- XGBoost Regressor ---
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost MAE: {mae_xgb}")

# Display a few predictions vs actual values for Random Forest
comparison_rf = pd.DataFrame({'Actual': y_test, 'Predicted_RF': y_pred_rf})
print("\nRandom Forest: Comparison of Actual vs Predicted closing prices:")
print(comparison_rf.head())
