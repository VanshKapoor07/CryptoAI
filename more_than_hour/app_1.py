from flask import Flask, request, jsonify, render_template, url_for
import tensorflow as tf
import numpy as np
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure 'static' directory exists for saving plots
if not os.path.exists("static"):
    os.makedirs("static")

# Load saved LSTM model
model = tf.keras.models.load_model("LSTM_more_model.keras")

# Define constants
symbol = 'APT/USDT'
timeframe = '1h'
n_input = 18  # Number of past hours to consider
n_future_hours = 5  # Predict next 5 hours

# Function to fetch latest data
def get_latest_data():
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=n_input)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df['close'].values.reshape(-1, 1)  # Extract close prices

# Route for home page
@app.route("/")
def home():
    return render_template("index_1.html")  # Ensure this file exists

# Route for prediction
@app.route("/predict", methods=["GET"])
def predict():
    # Fetch latest data
    window = get_latest_data()
    current_price = window[len(window) - 1,0]

    # Scale the data (use the same scaler used during training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(window)  # Fit scaler (or load from file)
    scaled_input = scaler.transform(window)

    # Reshape to match LSTM input shape
    scaled_window = scaled_input.reshape(1, n_input, 1)

    # Make a prediction (returns 5 future values)
    predicted_price_scaled = model.predict(scaled_window)[0]  # Get first row of output

    # Convert predictions back to original scale
    predicted_prices = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1)).flatten()

    profit_loss = [round(predicted_prices[i] - current_price, 4) for i in range(n_future_hours)]

    # Generate timestamps for future predictions
    future_timestamps = [(pd.Timestamp.now() + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M') for i in range(n_future_hours)]
    
    # Save predictions as a dictionary
    predictions_dict = {future_timestamps[i]: round(float(predicted_prices[i]), 4) for i in range(n_future_hours)}

    # Generate and save the plot
    plt.figure(figsize=(8, 5))
    plt.plot(future_timestamps, predicted_prices, marker='o', linestyle='-', color='b', label="Predicted Prices")

    for i in range(n_future_hours):
        plt.text(future_timestamps[i], predicted_prices[i], f"${profit_loss[i]}", fontsize=10, ha='right' if i % 2 == 0 else 'left', color='red' if profit_loss[i] < 0 else 'green')

    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.title("APT Price Prediction for Next 5 Hours")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    
    plot_path = "static/predictions.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return jsonify({
        "predictions": predictions_dict,
        "plot_url": url_for("static", filename="predictions.png")  # Returns /static/predictions.png
    })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
