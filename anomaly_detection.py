"""
Anomaly Detection in Real-Time Data Stream

This script implements an anomaly detection system that simulates a data stream
and identifies anomalies using an Exponential Moving Average (EMA) approach.
The system generates data points that incorporate trends, seasonality, and random noise,
and it visualizes the data along with detected anomalies in real-time.

Algorithm: Exponential Moving Average (EMA)
- The EMA is a moving average that places a greater weight and significance
   on the most recent data points.
- Like all moving averages, this technical indicator is used to produce
   buy and sell signals based on crossovers and divergences from the historical average.

Classes:
- AnomalyDetector:
    A class for detecting anomalies in a data stream based on an
    Exponential Moving Average and standard deviation. It maintains
    a recent history of values to calculate the EMA and identify
    anomalies based on a defined threshold.

Functions:
- generate_data_stream(length=1000):
    A generator function that yields simulated data points with trend, seasonality, and noise.
    Anomalies may be artificially introduced into the data stream to test
    the detector's performance.

- main():
    The main function that orchestrates the anomaly detection process.
    It initializes the anomaly detector, generates the data stream,
    and visualizes the data and detected anomalies in real-time.

Usage:
Run this script as the main module to start the anomaly detection system
and observe the real-time visualization of the data stream and detected anomalies.

Dependencies:
- NumPy
- Matplotlib
"""

from collections import deque
import random
from typing import Generator
import numpy as np
import matplotlib.pyplot as plt

def generate_data_stream(length: int = 1000) -> Generator[float, None, None]:
    """
    Generates a simulated data stream with trend, seasonality, random noise, and occasional spikes.

    This function yields a continuous sequence of data points designed to mimic
    stock market-like behavior. The generated data includes:
    - Trend: A slowly increasing or decreasing component that changes over time,
             simulating market growth or decline. Default
    - Seasonality: A periodic component with a 24-time-step cycle, representing
              daily trading patterns.
    - Noise: Random fluctuations to simulate small variations in stock prices.
    - Spikes: Occasional sudden increases or decreases (up to ±15) with
             a low probability (5%), representing unexpected market events.

    Parameters:
    length (int): The number of data points to generate in the stream. Default is 1000.

    Yields:
    float: The next data point in the simulated data stream,
    which is a combination of trend, seasonality, noise, and possible spikes.
    """
    for time in range(length):
        trend = 0.05 * time + random.uniform(-0.5, 0.5) * (time % 100 == 0)
        seasonality = 10 * np.sin(2 * np.pi * time / 24)
        spike = random.choice([0, random.uniform(-15, 15)]) if random.random() < 0.05 else 0
        noise = random.uniform(-5, 5)

        yield trend + seasonality + noise + spike


class AnomalyDetector:
    """
    A class for detecting anomalies in a time series using the
    Exponential Moving Average (EMA) algorithm.

    This class tracks an Exponential Moving Average (EMA)
    and uses it to detect anomalies based on a specified threshold.

    An anomaly is detected when a new data point deviates significantly from the EMA
    by more than a specified multiple of the standard deviation.

    Attributes:
    alpha (float): Smoothing factor for the EMA (0 < alpha <= 1).
    A higher alpha gives more weight to recent values.
    ema (float or None): Current EMA value, initialized to None until the first value is added.
    deviation (float): Standard deviation of recent values, used to determine the anomaly threshold.
    threshold (float): The multiple of the standard deviation
        that a value must exceed to be considered an anomaly.
    recent_values (deque): A deque to store recent values for calculating the standard deviation.

    Methods:
    update_ema(value):
        Updates the EMA with a new value.

    is_anomaly(value):
        Checks if a given value is an anomaly based on the deviation from the EMA.

    add_value(value):
        Adds a new value to the recent values, updates the EMA,
        and checks if the value is an anomaly.
    """

    def __init__(self, alpha: float = 0.1, threshold: float = 2.0) -> None:
        """
        Initializes the AnomalyDetector with a specified alpha for
        the EMA and an anomaly threshold.

        Parameters:
        alpha (float): Smoothing factor for the EMA (0 < alpha <= 1).
                       Default is 0.1.
        threshold (float): The multiple of the standard deviation used for
                           detecting anomalies. Default is 2.
        """
        self.alpha = alpha
        self.threshold = threshold
        self.ema = None
        self.deviation = 0
        self.recent_values = deque(maxlen=100)

    def update_ema(self, value: float) -> None:
        """
        Updates the Exponential Moving Average (EMA) with a new value.

        Parameters:
        value (float): The new value to be included in the EMA calculation.
        """
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

    def is_anomaly(self, value: float) -> bool:
        """
        Determines if a given value is an anomaly based on the EMA and recent deviations.

        A value is considered an anomaly if its absolute difference from the EMA is greater than the
        threshold multiplied by the standard deviation of recent values.

        Parameters:
        value (float): The value to check for anomaly.

        Returns:
        bool: True if the value is considered an anomaly, False otherwise.
        """
        if len(self.recent_values) < 2:
            return False

        self.deviation = np.std(self.recent_values)
        return abs(value - self.ema) > self.threshold * self.deviation

    def add_value(self, value: float) -> bool:
        """
        Adds a new value to the recent values, updates the EMA, and checks for anomalies.

        Parameters:
        value (float): The new value to be added to the recent values and used for EMA update.

        Returns:
        bool: True if the added value is considered an anomaly, False otherwise.
        """
        self.recent_values.append(value)
        self.update_ema(value)
        return self.is_anomaly(value)


def main() -> None:
    """
    Main function for running the anomaly detection system
    and visualizing the data stream in real-time.

    This function initializes an instance of the AnomalyDetector
    with specified alpha and threshold values, simulates a data stream
    using the `generate_data_stream` generator, and continuously updates
    the detector with incoming data points. It visualizes the data stream
    and highlights any detected anomalies in real-time.

    The process involves:
    1. Initializing the AnomalyDetector.
    2. Creating lists to store data points, detected anomalies, and EMA values.
    3. Setting up a real-time plot using Matplotlib.
    4. Iterating over generated data points from `generate_data_stream`.
    5. Adding each data point to the detector and checking for anomalies.
    6. Updating the plot to display the data stream and any detected anomalies.
    7. Using interactive mode for real-time visualization with a brief pause for rendering.

    Returns:
    None
    """
    detector = AnomalyDetector(alpha=0.1, threshold=2)

    data_points = []
    anomalies = []
    ema_values = []

    plt.ion()
    _, ax = plt.subplots(figsize=(12, 6))

    for value in generate_data_stream():
        data_points.append(value)
        if detector.add_value(value):
            anomalies.append(len(data_points) - 1)

        ema_values.append(detector.ema)

        ax.clear()
        ax.plot(data_points, label='Data Stream', color='blue')
        ax.plot(ema_values, label='EMA', color='orange', linestyle='--')
        ax.scatter(anomalies,
                   [data_points[i] for i in anomalies],
                   color='red', label='Anomaly', zorder=5)
        ax.set_title('Real-time Data Stream and Anomaly Detection')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='upper left')
        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
