# Anomaly Detection in Real-Time Data Stream

##  Overview

This project implements an anomaly detection system that simulates a data stream and identifies anomalies using an Exponential Moving Average (EMA) approach. The system generates data points that incorporate trends, seasonality, and random noise, and it visualizes the data along with detected anomalies in real-time.

Features

- Simulates a continuous data stream with trends and seasonality.
- Detects anomalies based on the Exponential Moving Average and standard deviation.
- Real-time visualization of data points and detected anomalies using Matplotlib.

## Algorithm

Exponential Moving Average (EMA)

The EMA is a moving average that places greater weight and significance on the most recent data points. It is used to produce buy and sell signals based on crossovers and divergences from the historical average.

## Usage

To run the script, simply execute the following command in your terminal:

```python anomaly_detection.py```

This will start the anomaly detection system and open a window displaying the real-time visualization of the data stream along with detected anomalies.

## Code Structure

Key Components

- generate_data_stream(length=1000): A generator function that yields simulated data points with trend, seasonality, random noise, and occasional spikes.
- AnomalyDetector: A class for detecting anomalies in a data stream based on an Exponential Moving Average and standard deviation. It maintains a recent history of values to calculate the EMA and identify anomalies based on a defined threshold.
- main(): The main function that orchestrates the anomaly detection process. It initializes the anomaly detector, generates the data stream, and visualizes the data and detected anomalies in real-time.

## Example Output

Upon running the script, a window will open showing the data stream and any detected anomalies highlighted in red. The Exponential Moving Average will be displayed as a dashed orange line.

## Customization

You can customize the following parameters in the AnomalyDetector:

- alpha: The smoothing factor for the EMA (0 < alpha <= 1).
- threshold: The multiple of the standard deviation used for detecting anomalies.

Modify the generate_data_stream function to change the behavior of the simulated data stream.

## Dependencies

- NumPy: A fundamental package for numerical computations in Python.
- Matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python.

## Acknowledgments

- The algorithm and methods used in this project are inspired by common practices in time series analysis and anomaly detection.

Feel free to modify any sections or add additional information as needed!
