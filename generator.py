#!/usr/bin/env python
# coding: utf-8

import numpy as np


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


if __name__ == '__main__':
    window_size = 32
    days = 700
    hours = days * 24
    half_hours = hours * 2
    observations = np.array([1000 * np.sin(np.pi / 24.0 * x * 0.5) ** 2 for x in xrange(half_hours)], dtype=int)
    X = rolling_window(observations, window_size)[:-1, :]
    y = np.array(observations[window_size:])
    np.save('sin2X', X)
    np.save('sin2y', y)
