#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def distort(distort_level):
    return 1 + random.uniform(-distort_level, distort_level)


if __name__ == '__main__':
    window_size = 32
    days = 14
    hours = days * 24
    half_hours = hours * 2
    distortion = 0.1
    observations = np.array(
        [distort(distortion) * 1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in range(half_hours)], dtype=int)
    X = rolling_window(observations, window_size)[:-1, :]
    y = np.array(observations[window_size:])
    np.save('sin2X_train', X)
    np.save('sin2y_train', y)

    test_days = 7
    half_h = test_days * 24 * 2
    anomaly_observation = np.array(
        [distort(distortion) * 1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in range(int(half_h * 0.45))] +
        [distort(distortion) * 200 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 + 250 for x in range(int(half_h * 0.45), int(half_h * 0.7))] +
        [distort(distortion) * 1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in range(int(half_h * 0.7), half_h)],
        dtype=int)
    X_anomaly = rolling_window(anomaly_observation, window_size)[:-1, :]
    y_anomaly = np.array(anomaly_observation[window_size:])
    np.save('sin2X_test', X_anomaly)
    np.save('sin2y_test', y_anomaly)
