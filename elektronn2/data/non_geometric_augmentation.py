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


def add_blobs(data,
              num_blobs=10,
              max_blob_size=15,
              min_blob_size=8,
              diffuseness=None,
              data_overwrite=False):
    """
     The function generates random blobs across the given data.

     A blob is a cube of the size that will be randomly drawn from the range:
    [ min_blob_size, max_blob_size ]. The area within the blob will be affected by
    the Gausssian smoothing. Depending of the diffuseness level the blob can stay
    almost transparent (in case of low diffuseness value) or be filled with the mean
    color value of the blob region (in case of high diffuseness value)

    By default the raw data will be copied in order to avoid of overwriting of the
    original data. However, the user can disable that and allow the function
    makes changes on the passed data. The function doesn't require to change
    the corresponding labels of the raw data

    Parameters
    ----------
    data : 4d np.ndarray float32
        The array represents the current field of view.
        The array has to have the following format: [ num_channels, z, x, y ]
    num_blobs : int
        The value represents the number of blobs that have to be generated
    max_blob_size : int
        The value represents the maximum size of a blob
    min_blob_size : int
        The value represents the minimum size of a blob
    diffuseness : int
        Diffuseness determines the level of the gaussian smoothing which will be
        performed within a randomly generated blob
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

    num_channels, depth, width, height = data.shape

    # generate random blobs for each individual channel
    for channel in range(num_channels):
        for i in range(num_blobs):

            # get the blob size
            blob_size = np.random.randint(low=min_blob_size,
                                          high=max_blob_size,
                                          dtype=int)
            make_blob(data[channel],
                      depth,
                      width,
                      height,
                      blob_size,
                      diffuseness)

    return data


def make_blob(data, depth, width, height, blob_size, diffuseness=None):
    """
    The function generates a random blob within the given filed of view.

    The user can control the level of diffuseness or it can be generated
    automatically drawing from a distribution

    Parameters
    ----------
    data : 4d np.ndarray float32
        The array represents the current field of view.
        The array has to have the following format: [ num_channels, z, x, y ]
    depth : int
        The depth of the filed of view
    width : int
        The width of the filed of view
    height : int
        The height of the filed of view
    blob_size : int
        A particular size of a blob
    diffuseness : int
        Level of blob diffuseness ( transparency )


    Returns
    -------
    """

    delta = blob_size // 2

    seed = throw_seed(depth, width, height, delta)

    slice_z, slice_x, slice_y = make_slice(seed, blob_size)

    """
    blob = data[ slice_z[ "low" ] + delta : slice_z[ "high" ]  - delta,
                 slice_x[ "low" ] + delta : slice_x[ "high" ]  - delta,
                 slice_y[ "low" ] + delta : slice_y[ "high" ] - delta ]
    """

    snippet = data[slice_z["low"]: slice_z["high"],
                   slice_x["low"]: slice_x["high"],
                   slice_y["low"]: slice_y["high"]]

    if not diffuseness:
        diffuseness = np.random.randint(low=1, high=6, dtype=int)

    snippet = ndimage.gaussian_filter(snippet, diffuseness)

    data[slice_z["low"]: slice_z["high"],
         slice_x["low"]: slice_x["high"],
         slice_y["low"]: slice_y["high"]] = snippet


def throw_seed(depth, width, height, delta):
    """
    The function generates a random seed within the given volume which is represented
    by 3 integer values: depth, width, height. The function also takes care of
    the fact that a seed should not lay within the margin which is determined by the
    parameter "delta"

                _________________________________________________________
                |                            d                           |
                |                            e                           |
                |                            l                           |
                |                            t                           |
                |           _________________a________________           |
                |          |                                  |          |
                |          |                                  |          |
                |          |                                  |          |
                |          |                                  |          |
                |  delta   |  place where a seed can be taken |  delta   |
                |          |                                  |          |
                |          |                                  |          |
                |          |                                  |          |
                |          |__________________________________|          |
                |                            d                           |
                |                            e                           |
                |                            l                           |
                |                            t                           |
                |____________________________a___________________________|


    Parameters
    ----------
    depth : int
        depth represents the z dimension of the given vield of view
    width : int
        width represents the x dimension of the given vield of view
    height : int
        height represents the y dimension of the given vield of view
    delta : int
        delta represents the margin where a seed should not be taken


    Returns
    -------
    dictionary : 3 int values (coordinates of a seed)
    """

    z = np.random.randint(low=delta, high=depth-delta, dtype=int)
    x = np.random.randint(low=delta, high=width-delta, dtype=int)
    y = np.random.randint(low=delta, high=height-delta, dtype=int)

    return {"z": z, "x": x, "y": y}


def make_slice(seed, delta):
    """
    Given the coordinates of a seed the function generates three slices along three
    different axes: z, y, z. The slices can be used later to extract a particular place
    from the given data

    Parameters
    ----------
    seed : dictionary with three int values
        seed represents the coordinates of a seed
    delta : int
        delta represents a half of the slice length

    Returns
    -------
    3 dictionaries
        The dictionaries represent the slices along z, x, y respectively
        Each dictionary has the following keys: low and high; which determines the lower
        and the higher indices of the slice along a particular axis
    """

    slice_z = {}
    slice_x = {}
    slice_y = {}

    slice_z["low"] = seed["z"] - delta
    slice_z["high"] = seed["z"] + delta

    slice_x["low"] = seed["x"] - delta
    slice_x["high"] = seed["x"] + delta

    slice_y["low"] = seed["y"] - delta
    slice_y["high"] = seed["y"] + delta

    return slice_z, slice_x, slice_y
