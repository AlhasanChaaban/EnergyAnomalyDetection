from flask import Flask, jsonify, render_template
import numpy as np
import random
from sklearn.ensemble import IsolationForest
import math

app = Flask(__name__)

data = []
max_data_size = 500
model = IsolationForest(contamination=0.01)
current_time = 0

def simulate_row(time):
    seasonality = 10 * math.sin(2 * math.pi * time / 24)
    trend = 0.05 * time
    noise_temp = random.uniform(-2, 2)
    noise_humidity = random.uniform(-1, 1)
    noise_power = random.uniform(-5, 5)
    temperature = 50 + seasonality + noise_temp
    humidity = 30 + noise_humidity
    power_demand = 200 + seasonality * 2 + trend + noise_power
    return [temperature, humidity, power_demand]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stream_data', methods=['GET'])
def stream_data():
    global data, model, current_time
    batch_size = 10
    new_rows = [simulate_row(current_time + i) for i in range(batch_size)]
    current_time += batch_size
    data.extend(new_rows)
    if len(data) > max_data_size:
        data = data[-max_data_size:]
    data_np = np.array(data)
    anomalies = [0] * batch_size

    if len(data_np) > 50:
        if current_time % (5 * batch_size) == 0:
            model.fit(data_np)
        predictions = model.predict(data_np)
        anomalies = [int(p == -1) for p in predictions[-batch_size:]]

    return jsonify({
        'new_rows': new_rows,
        'anomalies': anomalies
    })

if __name__ == '__main__':
    app.run(debug=True)
