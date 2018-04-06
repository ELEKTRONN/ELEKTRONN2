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


def blurry_blobs(data, diffuseness, num_blob_range=(5, 15),
                 blob_size_range=(10, 32), data_overwrite=False):
    """
    Generates random blobs across the given (raw) input data.

    A blob is a cube of the size that will be randomly drawn from the blob_size_range.
    The area within the blob will be affected by Gausssian smoothing.
    Depending on the diffuseness level the blob can stay
    almost transparent (in case of low diffuseness value) or be filled with the mean
    color value of the blob region (in case of high diffuseness value)

    By default the raw data will be copied in order to avoid overwriting the
    original data. However, the user can disable that and allow the function
    to make changes to the passed data. The function doesn't require to change
    the corresponding labels of the raw data

    If the size of a blob exceeds the length of any dimension of the give volume
    the blob size will be assigned to the length of the corresponding dimension.

    Parameters
    ----------
    data: np.ndarray
        Current field of view.
        It has to have the following format: [num_channels, z, x, y]
    diffuseness: int
        The standard deviation of the applied Gaussian kernel.
    num_blob_range: (int, int)
        Range of possible numbers of blobs generated
    blob_size_range: (int, int)
        Range of possible blob sizes
    data_overwrite: bool
        Determines whether the input data may be
        modified. If ``data_overwrite`` is true
        the original data passed to the function will be overwritten.
    Returns
    -------
    data: np.ndarray
        Augmented data of the following format: [num_channels, z, x, y]
    """
    return add_blobs(data, num_blob_range=num_blob_range,
            blob_size_range=blob_size_range, data_overwrite=data_overwrite,
            blob_operator=blur_blob, blob_operator_args=(diffuseness,))

def noisy_random_erasing(data, noise_range=(0, 255),num_blob_range=(5, 15),
                         blob_size_range=(10, 32), data_overwrite=False):
    """
    Like blurry_blobs, but the blob area is filled with random noise.
    """
    return add_blobs(data, num_blob_range=num_blob_range,
            blob_size_range=blob_size_range, data_overwrite=data_overwrite,
            blob_operator=random_noise_blob, blob_operator_args=(noise_range,))

def uniform_random_erasing(data, value, num_blob_range=(5, 15),
                           blob_size_range=(10, 32), data_overwrite=False):
    """
    Like blurry blobs, but the blob area is filled with a uniform value.
    """
    return add_blobs(data, num_blob_range=num_blob_range,
             blob_size_range=blob_size_range, data_overwrite=data_overwrite,
             blob_operator=uniform_blob, blob_operator_args=(value,))

def add_blobs(data, num_blob_range, blob_size_range, data_overwrite,
              blob_operator, blob_operator_args=()):

    if not data_overwrite:
        data = data.copy()

    num_channels, depth, width, height = data.shape

    # blob_size_restriction - the minimal dimension of the given volume
    blob_size_restriction = np.min(data.shape[1:])
    num_blobs = np.random.randint(low=num_blob_range[0],
                                  high=num_blob_range[1],
                                  dtype=np.uint16)
    # generate random blobs for each individual channel
    for channel in range(num_channels):
        for i in range(num_blobs):

            # get a blob size drawn from the distribution
            blob_size = np.random.randint(low=blob_size_range[0],
                                          high=blob_size_range[1],
                                          dtype=np.int16)

            # check whether the chosen blob size fits
            # to the given volume. If it doesn't assign
            # the blob size to the blob_size_restriction
            # and subtract 2 to as the offset to guarantee
            # the fit
            if blob_size >= blob_size_restriction:
                blob_size = blob_size_restriction - 2
                break

            blob = make_blob(data[channel],
                      blob_operator,
                      blob_operator_args,
                      depth,
                      width,
                      height,
                      blob_size)

    return data

def make_blob(data, blob_operator, blob_operator_args, depth, width, height, blob_size):
    """
    Generates a random blob within the given field of view.

    The user can control the level of diffuseness or it can be generated
    automatically drawing from a distribution.

    Parameters
    ----------
    data: np.ndarray
        Represents the current field of view.
        It has to have the following format: [num_channels, z, x, y]
    blob_operator: Function to apply on the blob, i.e.
        blur_blob, random_noise_blob or uniform_blob
    blob_operator_args: tuple
        Arguments that are passed to the blob operator
    depth: int
        The depth of the field of view
    width: int
        The width of the field of view
    height: int
        The height of the field of view
    blob_size: int
        A particular size of a blob

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

    blob = data[bottom[0]: top[0],
                bottom[1]: top[1],
                bottom[2]: top[2]]

    data[bottom[0]: top[0],
    bottom[1]: top[1],
    bottom[2]: top[2]] = blob_operator(blob, *blob_operator_args)

def blur_blob(blob, diffuseness=None):
    if not diffuseness:
        diffuseness = np.random.randint(low=1, high=6, dtype=np.int16)

    return ndimage.gaussian_filter(blob, diffuseness)

def random_noise_blob(blob, noise_range=(0, 255)):
    return np.random.uniform(low=noise_range[0], high=noise_range[1], size=blob.shape)

def uniform_blob(blob, value):
    return np.ones(blob.shape) * value


