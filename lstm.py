#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

window_size = 32


def build_model():
    model = Sequential()
    model.add(LSTM(4, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    X_train = np.load('data/sin2X_train0_365.npy')
    y_train = np.load('data/sin2y_train0_365.npy')

    Xscaler = MinMaxScaler(feature_range=(0, 1))
    yscaler = MinMaxScaler(feature_range=(0, 1))
    X_train = Xscaler.fit_transform(X_train)
    y_train = yscaler.fit_transform(y_train)

    X_train = X_train.reshape(X_train.shape + (1,))

    model = build_model()
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=5)
    model.save_weights('models/lstm.dat')

    joblib.dump(Xscaler, 'models/lstm_scaler_X.dat')
    joblib.dump(yscaler, 'models/lstm_scaler_y.dat')
