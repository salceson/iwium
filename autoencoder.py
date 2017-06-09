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

    predicted = auto_encoder.predict(X_test_scaled)
    predicted_real = X_test_scaler.inverse_transform(predicted)

    delta = np.abs(X_test - predicted_real)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(X_test)

    plt.subplot(212)
    plt.plot(delta)
    plt.show()
