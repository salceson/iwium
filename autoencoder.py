#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

window_size = 32


def build_model():
    encoder_input = Input(shape=(window_size,))

    encoded = Dense(16, activation='relu')(encoder_input)
    encoded = Dense(8, activation='relu')(encoded)
    encoded = Dense(4, activation='relu')(encoded)

    decoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(decoded)
    decoded = Dense(window_size, activation='sigmoid')(decoded)

    return Model(encoder_input, decoded)


if __name__ == '__main__':
    X_train = np.load('data/sin2X_train0_7.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)

    auto_encoder = build_model()
    auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    auto_encoder.fit(X_train_scaled, X_train_scaled, epochs=100)

    auto_encoder.save_weights('models/auto_encoder.dat')
    joblib.dump(scaler, 'models/auto_encoder_scaler.dat')
