from flask import Flask, jsonify, render_template
import numpy as np
import random
from sklearn.ensemble import IsolationForest
import math

app = Flask(__name__)

# Initialize global variables
data = []  # List to store the data points
max_data_size = 500  # Maximum number of data points to retain
model = IsolationForest(contamination=0.01)  # Anomaly detection model
current_time = 0  # Current time for data simulation

def simulate_row(time):
    """
    Simulate a row of data with seasonal, trend, and noise components.
    
    Args:
        time (int): The current time step for generating data.
        
    Returns:
        list: A list containing simulated temperature, humidity, and power demand.
    """
    seasonality = 10 * math.sin(2 * math.pi * time / 24)  # Seasonal pattern
    trend = 0.05 * time  # Trend component
    noise_temp = random.uniform(-2, 2)  # Random noise for temperature
    noise_humidity = random.uniform(-1, 1)  # Random noise for humidity
    noise_power = random.uniform(-5, 5)  # Random noise for power demand
    
    temperature = 50 + seasonality + noise_temp  # Simulated temperature
    humidity = 30 + noise_humidity  # Simulated humidity
    power_demand = 200 + seasonality * 2 + trend + noise_power  # Simulated power demand
    
    return [temperature, humidity, power_demand]

@app.route('/')
def home():
    """
    Render the home page of the application.
    
    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template('index.html')

@app.route('/stream_data', methods=['GET'])
def stream_data():
    """
    Handle requests for streaming data and perform anomaly detection.
    
    Returns:
        json: JSON object containing new rows of data and detected anomalies.
    """
    global data, model, current_time
    try:
        batch_size = 10  # Number of data points per batch
        new_rows = [simulate_row(current_time + i) for i in range(batch_size)]  # Generate new data
        current_time += batch_size  # Update the time step
        data.extend(new_rows)  # Add new data to the list
        
        # Retain only the most recent data up to max_data_size
        if len(data) > max_data_size:
            data = data[-max_data_size:]
        
        data_np = np.array(data)  # Convert data to NumPy array
        anomalies = [0] * batch_size  # Initialize list to hold anomaly flags

        # Fit model and detect anomalies if there is enough data
        if len(data_np) > 50:
            if len(data_np) % (5 * batch_size) == 0:
                model.fit(data_np)  # Fit the model with recent data
            predictions = model.predict(data_np)  # Predict anomalies
            anomalies = [int(p == -1) for p in predictions[-batch_size:]]  # Flag anomalies
        
        return jsonify({
            'new_rows': new_rows,
            'anomalies': anomalies
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message in case of exception

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application
