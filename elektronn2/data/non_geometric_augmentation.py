# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import numpy as np
from scipy import ndimage


class InvalidNonGeomAugmentParameters(Exception):
    pass


def noiseAugment(data, level=0.15, data_overwrite=False):
    """
    The function adds random noise to the original raw data passed to the function.

    By default the raw data will be copied in order to avoid of overwriting of the
    original data. However, the user can disable that and allow the function
    makes changes on the passed data. The function doesn't require to change
    the corresponding labels of the raw data

    If the noise level is too high (more than 1.0) or too low (less than 0.0)
    the function throws the corresponding error

    Parameters
    ----------
    data : 4d np.ndarray float32
        The array represents the current field of view.
        The array has to have the following format: [ num_channels, z, x, y ]
    level : float
        The value determines determines the strength of the noise. The maximum value
        must be 1.0.
    data_overwrite : boolean
        The value determines whether the raw data might be
        overwritten or can be modified. If the "data_overwrite" is equal to True
        the original data passed to the function will be overwritten


     Returns
    -------
    data : 4d np.ndarray float32
        The array has   the following format: [ num_channels, z, x, y ]
    """
    MIN_NOISE = 0
    MAX_NOISE = 1

    if level < MIN_NOISE or level > MAX_NOISE:
        raise InvalidNonGeomAugmentParameters("Noise level exceeds either "
                                              "Min or Max level")

    if not data_overwrite:
        data = data.copy()

    num_channels = data.shape[0]

    # add noise to all channels of the data
    for channel in range(num_channels):
        shape = data[channel].shape
        data[channel] += level * (np.random.random(shape) - 0.5)
    return data


def blurAugment(data, level=1, data_overwrite=False):
    """
    The function performs Gaussian smoothing on the original data.

    By default the raw data will be copied in order to avoid of overwriting of the
    original data. However, the user can disable that and allow the function
    makes changes on the passed data. The function doesn't require to change
    the corresponding labels of the raw data

    Parameters
    ----------
    data : 4d np.ndarray float32
        The array represents the current field of view.
        The array has to have the following format: [ num_channels, z, x, y ]
    level : int
        The value determines the strength of the gaussian smoothing.
    data_overwrite : boolean
        The value determines whether the raw data might be
        overwritten or can be modified. If the "data_overwrite" is equal to True
        the original data passed to the function will be overwritten


    Returns
    -------
    data : 4d np.ndarray float32
        The array has   the following format: [ num_channels, z, x, y ]
    """

    if not data_overwrite:
        data = data.copy()

    num_channels = data.shape[0]

    for channel in range(num_channels):
        data[channel] = ndimage.gaussian_filter(data[channel], level)

    return data


def mixBlurNoiseAugment(data,
                        noise_level=0.15,
                        smoothing_level=1,
                        data_overwrite=False):
    """
    The function performs Gaussian smoothing and adding random noise respectively.

    By default the raw data will be copied in order to avoid of overwriting of the
    original data. However, the user can disable that and allow the function
    makes changes on the passed data. The function doesn't require to change
    the corresponding labels of the raw data

    Parameters
    ----------
    data : 4d np.ndarray float32
        The array represents the current field of view.
        The array has to have the following format: [ num_channels, z, x, y ]
    noise_level : float
        The value determines determines the strength of the noise. The maximum value
        must be 1.0.
    smoothing_level : int
        The value determines the strength of the gaussian smoothing.
    data_overwrite : boolean
        The value determines whether the raw data might be
        overwritten or can be modified. If the "data_overwrite" is equal to True
        the original data passed to the function will be overwritten


    Returns
    -------
    data : 4d np.ndarray float32
        The array has   the following format: [ num_channels, z, x, y ]
    """
    if not data_overwrite:
        data = data.copy()

    blurAugment(data, smoothing_level, data_overwrite=True)
    noiseAugment(data, noise_level, data_overwrite=True)

    return data
