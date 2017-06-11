#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib

from lstm import build_model

if __name__ == '__main__':
    test = 0  # 0 - 3

    Xscaler = joblib.load('models/lstm_scaler_X.dat')
    yscaler = joblib.load('models/lstm_scaler_y.dat')

    X_test = np.load('data/sin2X_test%d_365.npy' % test)
    y_test = np.load('data/sin2y_test%d_365.npy' % test)
    X_test_scaled = Xscaler.transform(X_test)

    lstm = build_model()
    lstm.load_weights('models/lstm.dat')

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    max_input = np.amax(X_test)
    min_input = 0
    samples_count = X_test.shape[0]

    Xs = []
    deltas = []
    inputs = []

    X_anomalies = []
    Y_anomalies = []

    for i in range(X_test_scaled.shape[0]):
        y = y_test[i]
        sample_scaled = np.reshape(X_test_scaled[i], (1, -1, 1))
        predicted = lstm.predict(sample_scaled)
        predicted_real = yscaler.inverse_transform(predicted).reshape(1,)

        delta = np.abs(y - predicted_real)
        deltas.append(delta)
        inputs.append(y)
        Xs.append(i)

        if delta > 150:
            X_anomalies.append(i)
            Y_anomalies.append(y)

        ax1.clear()
        ax1.set_title('Test data')
        ax1.plot(Xs, inputs)
        ax1.plot(X_anomalies, Y_anomalies, 'ro')
        ax2.clear()
        ax2.set_title('Absolute error of the prediction')
        ax2.plot(Xs, deltas)

        ax1.set_xlim([0, samples_count + 1])
        ax1.set_ylim([min_input, max_input])
        ax2.set_xlim([0, samples_count + 1])
        ax2.set_ylim([min_input, max_input])

        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
