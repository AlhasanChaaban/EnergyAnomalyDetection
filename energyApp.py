from flask import Flask, jsonify, render_template
import numpy as np
import random

app = Flask(__name__)


def simulate_energy_data(n_points=500):
    time = np.arange(n_points)
    seasonality = 10 * np.sin(2 * np.pi * time / 24)
    trend = 0.05 * time
    noise = np.random.normal(0, 1, n_points)
    data = 50 + seasonality + trend + noise

    
    for i in range(5):
        anomaly_index = random.randint(0, n_points - 1)
        data[anomaly_index] += random.choice([15, -15])

    return time.tolist(), data.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    time, data = simulate_energy_data()
    return jsonify({'time': time, 'data': data})

if __name__ == '__main__':
    app.run(debug=True)
