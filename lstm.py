#!/usr/bin/env/python
# coding: utf-8

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

window_size = 32

X = np.load('sin2X.npy')
y = np.load('sin2y.npy')

Xscaler = MinMaxScaler(feature_range=(0, 1))
yscaler = MinMaxScaler(feature_range=(0, 1))
X = Xscaler.fit_transform(X)
y = yscaler.fit_transform(y)

X = X.reshape(X.shape + (1,))

max_X = np.max(np.max(X))

model = Sequential()
model.add(LSTM(4, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print model.summary()

model.fit(X, y, epochs=2)

y_real = yscaler.inverse_transform(y)
predicted = model.predict(X)
predicted_real = np.reshape(yscaler.inverse_transform(predicted), (len(y_real),))
print predicted_real, y_real

xs = np.array([0.5 * x for x in xrange(len(y_real))])

plt.plot(xs[:100], y_real[:100], 'g-')
plt.plot(xs[:100], predicted_real[:100], 'r-')
plt.show()
