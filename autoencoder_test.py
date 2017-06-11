#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib

from autoencoder import window_size, build_model

if __name__ == '__main__':
    scaler = joblib.load('auto_encoder_scaler.dat')

    X_test = np.load('sin2X_test.npy')
    X_test_scaled = scaler.transform(X_test)

    auto_encoder = build_model()
    auto_encoder.load_weights('auto_encoder.dat')

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
        sample = np.reshape(X_test[i], (1, -1))
        sample_scaled = np.reshape(X_test_scaled[i], (1, -1))
        predicted = auto_encoder.predict(sample_scaled)
        predicted_real = scaler.inverse_transform(predicted)

        delta = np.abs(sample - predicted_real)[0, window_size - 1]
        deltas.append(delta)
        input = X_test[i, window_size - 1]
        inputs.append(input)
        Xs.append(i)

        if delta > 150:
            X_anomalies.append(i)
            Y_anomalies.append(input)

        ax1.clear()
        ax1.set_title('Test data')
        ax1.plot(Xs, inputs)
        ax1.plot(X_anomalies, Y_anomalies, 'ro')
        ax2.clear()
        ax2.set_title('Absolute error of the representation')
        ax2.plot(Xs, deltas)

        ax1.set_xlim([0, samples_count + 1])
        ax1.set_ylim([min_input, max_input])
        ax2.set_xlim([0, samples_count + 1])
        ax2.set_ylim([min_input, max_input])

        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
