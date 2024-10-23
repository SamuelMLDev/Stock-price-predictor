import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Fetch historical stock price data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    # Let's get the last 5 years of stock data
    data = stock.history(period="5y")
    # Select relevant columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    # Create a column for the next day's closing price (our target)
    data['Next Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)  # Drop the last row where we don't have 'Next Close'
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
X = stock_data[['Open', 'High', 'Low', 'Volume', 'Sentiment']]
y = stock_data['Next Close']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Display a few predictions vs actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted closing prices:")
print(comparison.head())
