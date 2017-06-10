#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    window_size = 32
    encoding_dim = 4

    X_train = np.load('sin2X_train.npy')
    X_train_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = X_train_scaler.fit_transform(X_train)

    encoder_input = Input(shape=(window_size,))

    encoded = Dense(16, activation='relu')(encoder_input)
    encoded = Dense(8, activation='relu')(encoded)
    encoded = Dense(4, activation='relu')(encoded)

    decoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(decoded)
    decoded = Dense(window_size, activation='sigmoid')(decoded)

    auto_encoder = Model(encoder_input, decoded)
    auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    auto_encoder.fit(X_train_scaled, X_train_scaled, epochs=100)
    auto_encoder.save_weights('auto_encoder.dat')

    X_test = np.load('sin2X_test.npy')
    X_test_scaler = MinMaxScaler(feature_range=(0, 1))
    X_test_scaled = X_test_scaler.fit_transform(X_test)

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
        predicted_real = X_test_scaler.inverse_transform(predicted)

        delta = np.amax(np.abs(sample - predicted_real), axis=1)
        deltas.append(delta)
        input = X_test[i, window_size - 1]
        inputs.append(input)
        Xs.append(i)

        if delta > 200:
            X_anomalies.append(i)
            Y_anomalies.append(input)

        ax1.clear()
        ax1.plot(Xs, inputs)
        ax1.plot(X_anomalies, Y_anomalies, 'ro')
        ax2.clear()
        ax2.plot(Xs, deltas)

        ax1.set_xlim([0, samples_count + 1])
        ax1.set_ylim([min_input, max_input])
        ax2.set_xlim([0, samples_count + 1])
        ax2.set_ylim([min_input, max_input])

        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
