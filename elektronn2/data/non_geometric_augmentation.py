# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import numpy as np
from scipy import ndimage


class InvalidNonGeomAugmentParameters(Exception):
    pass


def noise_augment(data, level=0.15, data_overwrite=False):
    """
    Adds random noise to the original raw data passed to the function.

    By default the raw data will be copied in order to avoid of overwriting of the
    original data. However, the user can disable that and allow the function
    to make changes on the passed data. The function doesn't require to change
    the corresponding labels of the raw data

    If the noise level is too high (more than 1.0) or too low (less than 0.0)
    the function throws the corresponding error

    Parameters
    ----------
    data: np.ndarray
        Current field of view.
        Has to have the following format: [num_channels, z, x, y]
    level: float
        Strength of the noise. The maximum value is 1.0.
    data_overwrite: bool
        Determines whether the input data may be
        modified. If ``data_overwrite`` is true
        the original data passed to the function will be overwritten.


     Returns
    -------
    data: np.ndarray
        The array has the following format: [num_channels, z, x, y]
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


def blur_augment(data, level=1, data_overwrite=False):
    """
    The function performs Gaussian smoothing on the original data.

    By default the raw data will be copied in order to avoid overwriting the
    original data. However, the user can disable that and allow the function
    to make changes to the passed data. The function doesn't require to change
    the corresponding labels of the raw data.

    Parameters
    ----------
    data: np.ndarray
        Current field of view.
        Has to have the following format: [num_channels, z, x, y]
    level: int
        Strength of the gaussian smoothing.
    data_overwrite: bool
        Determines whether the input data may be
        modified. If ``data_overwrite`` is true
        the original data passed to the function will be overwritten.


    Returns
    -------
    data: np.ndarray
        The array has the following format: [num_channels, z, x, y]
    """

    if not data_overwrite:
        data = data.copy()

    num_channels = data.shape[0]

    for channel in range(num_channels):
        data[channel] = ndimage.gaussian_filter(data[channel], level)

    return data


def add_blobs(data,
              num_blobs=10,
              max_blob_size=32,
              min_blob_size=10,
              diffuseness=None,
              data_overwrite=False):
    """
    Generates random blobs across the given (raw) input data.

    A blob is a cube of the size that will be randomly drawn from the range
    [min_blob_size, max_blob_size]. The area within the blob will be affected by
    Gausssian smoothing. Depending on the diffuseness level the blob can stay
    almost transparent (in case of low diffuseness value) or be filled with the mean
    color value of the blob region (in case of high diffuseness value)

    By default the raw data will be copied in order to avoid overwriting the
    original data. However, the user can disable that and allow the function
    to make changes to the passed data. The function doesn't require to change
    the corresponding labels of the raw data

    If the size of a blob exceeds the length of any dimension of the give volume
    the blob size will be assign to the length of the corresponding dimension

    Parameters
    ----------
    data: np.ndarray
        Current field of view.
        It has to have the following format: [num_channels, z, x, y]
    num_blobs: int
        Number of blobs that have to be generated
    max_blob_size: int
        Maximum size of a blob
    min_blob_size: int
        Minimum size of a blob
    diffuseness: int
        Determines the level of the gaussian smoothing which will be
        performed within a randomly generated blob
    data_overwrite: bool
        Determines whether the input data may be
        modified. If ``data_overwrite`` is true
        the original data passed to the function will be overwritten.


    Returns
    -------
    data: np.ndarray
        Augmented data of the following format: [num_channels, z, x, y]
    """

    if not data_overwrite:
        data = data.copy()

    num_channels, depth, width, height = data.shape

    # blob_size_restriction - the minimal dimension of the given volume
    blob_size_restriction = np.min(data.shape[1:])

    # generate random blobs for each individual channel
    for channel in range(num_channels):
        for i in range(num_blobs):

            # get an appropriate blob size
            # get a blob size drawn from the distribution
            blob_size = np.random.randint(low=min_blob_size,
                                          high=max_blob_size,
                                          dtype=np.int16)

            # check whether the chosen blob size fits
            # to the given volume. If it doesn't assign
            # the blob size to the blob_size_restriction
            # and subtract 2 to as the offset to guarantee
            # the fit
            if blob_size >= blob_size_restriction:
                blob_size = blob_size_restriction - 2

            make_blob(data[channel],
                      depth,
                      width,
                      height,
                      blob_size,
                      diffuseness)

    return data


def make_blob(data, depth, width, height, blob_size, diffuseness=None):
    """
    Generates a random blob within the given field of view.

    The user can control the level of diffuseness or it can be generated
    automatically drawing from a distribution.

    Parameters
    ----------
    data: np.ndarray
        Represents the current field of view.
        It has to have the following format: [num_channels, z, x, y]
    depth: int
        The depth of the field of view
    width: int
        The width of the field of view
    height: int
        The height of the field of view
    blob_size: int
        A particular size of a blob
    diffuseness: int
        Level of blob diffuseness ( transparency )


    Returns
    -------
    """

    delta = blob_size // 2

    # select a random center
    z = np.random.randint(low=delta, high=depth - delta, dtype=np.int16)
    x = np.random.randint(low=delta, high=width - delta, dtype=np.int16)
    y = np.random.randint(low=delta, high=height - delta, dtype=np.int16)
    center = np.array([z, x, y])

    # make a slice and extract a cube
    bottom = center - delta
    top = center + delta

    snippet = data[bottom[0]: top[0],
                   bottom[1]: top[1],
                   bottom[2]: top[2]]

    if not diffuseness:
        diffuseness = np.random.randint(low=1, high=6, dtype=np.int16)

    snippet = ndimage.gaussian_filter(snippet, diffuseness)

    data[bottom[0]: top[0],
         bottom[1]: top[1],
         bottom[2]: top[2]] = snippet
