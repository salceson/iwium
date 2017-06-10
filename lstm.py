#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

window_size = 32

X_train = np.load('sin2X_train.npy')
y_train = np.load('sin2y_train.npy')

Xscaler = MinMaxScaler(feature_range=(0, 1))
yscaler = MinMaxScaler(feature_range=(0, 1))
X_train = Xscaler.fit_transform(X_train)
y_train = yscaler.fit_transform(y_train)

X_train = X_train.reshape(X_train.shape + (1,))

model = Sequential()
model.add(LSTM(4, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(X_train, y_train, epochs=2)

# TEST

X_test = np.load('sin2X_test.npy')
y_test = np.load('sin2y_test.npy')

Xscaler = MinMaxScaler(feature_range=(0, 1))
yscaler = MinMaxScaler(feature_range=(0, 1))
X_test = Xscaler.fit_transform(X_test)
y_test = yscaler.fit_transform(y_test)

X_test = X_test.reshape(X_test.shape + (1,))

y_real = yscaler.inverse_transform(y_test)
predicted = model.predict(X_test)
predicted_real = np.reshape(yscaler.inverse_transform(predicted), (len(y_real),))
print(predicted_real, y_real)

xs = np.array([0.5 * x for x in range(len(y_real))])

plt.plot(xs[:500], y_real[:500], 'g-')
plt.plot(xs[:500], predicted_real[:500], 'r-')
plt.show()
