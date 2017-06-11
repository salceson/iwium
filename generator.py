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


def test1(half_h):
    return [distort(distortion) * 0 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 + 250 for x in
            range(int(half_h * 0.45), int(half_h * 0.7))]


def test2(half_h):
    return [distort(distortion) * -250 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 + 500 for x in
            range(int(half_h * 0.45), int(half_h * 0.7))]


def test3(half_h):
    return [distort(distortion) * 800 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in
            range(int(half_h * 0.45), int(half_h * 0.7))]


def test4(half_h):
    return [distort(distortion * 3) * 1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 + 0 for x in
            range(int(half_h * 0.45), int(half_h * 0.7))]


tests = [
    test1, test2, test3, test4
]

lengths = [
    7, 15, 365
]

if __name__ == '__main__':
    for test_no in range(len(tests)):
        window_size = 32
        for days in lengths:
            hours = days * 24
            half_hours = hours * 2
            distortion = 0.1
            observations = np.array(
                [distort(distortion) * 1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in range(half_hours)],
                dtype=int)
            X = rolling_window(observations, window_size)[:-1, :]
            y = np.array(observations[window_size:])
            np.save('in/sin2X_train' + str(test_no) + "_" + str(days), X)
            np.save('in/sin2y_train' + str(test_no) + "_" + str(days), y)

            test_days = 7
            half_h = test_days * 24 * 2
            anomaly_observation = np.array(
                [distort(distortion) * 1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in range(int(half_h * 0.45))] +
                tests[test_no](half_h) +
                [distort(distortion) * 1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in
                 range(int(half_h * 0.7), half_h)],
                dtype=int)
            X_anomaly = rolling_window(anomaly_observation, window_size)[:-1, :]
            y_anomaly = np.array(anomaly_observation[window_size:])
            np.save('in/sin2X_test' + str(test_no) + "_" + str(days), X_anomaly)
            np.save('in/sin2y_test' + str(test_no) + "_" + str(days), y_anomaly)
