# --------------------------------------------------------------------------------
# Copyright (C) 2022 ZauggGroup
#
# This file is a copy (or a modified version) of the original file from the
# following GitHub repository:
#
# Repository: https://github.com/ZauggGroup/DeePiCt
# Original file: https://github.com/ZauggGroup/DeePiCt/blob/main/spectrum_filter/FilterUtils.py
# Repository URL: https://github.com/ZauggGroup/DeePiCt
# Original author(s): de Teresa, I.*, Goetz S.K.*, Mattausch, A., Stojanovska, F.,
#   Zimmerli C., Toro-Nahuelpan M., Cheng, D.W.C., Tollervey, F. , Pape, C.,
#   Beck, M., Diz-Muñoz, A., Kreshuk, A., Mahamid, J. and Zaugg, J.
# License: Apache License 2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------------

from typing import List, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d


def hypotenuse_ndim(axes: List[np.ndarray], offset: float = 0.5) -> np.ndarray:
    """Function to compute the hypotenuse for n-dimensional axes.

    This is used to compute the distance of each voxel to the center point.
    Each 2D hypotenuse is computed by a^2 + b^2 = c^2
    Thus, this also gives the Euclidean distance of a 2D point.
    This is extended recursively to any dimension.

    Parameters
    ----------
    axes : List[np.ndarray]
        List of axes in n-dimensional space.
    offset : float, optional
        Offset to apply before calculating the hypotenuse, by default 0.5.


    Returns
    -------
    np.ndarray
        An n-dimensional numpy array representing the computed hypotenuse. The shape of
        the returned array is determined by the shape of the input axes.


    Notes
    -----
    The reason to use this function instead of simply computing the Euclidean distance
    directly is to calculate the hypotenuse with respect to an offset from the center
    point of the axes. This offset is subtracted from the maximum of each axis shape
    before the hypotenuse is calculated.
    """
    if len(axes) == 2:
        return np.hypot(
            axes[0] - max(axes[0].shape) * offset,
            axes[1] - max(axes[1].shape) * offset,
        )
    else:
        return np.hypot(
            hypotenuse_ndim(axes[1:], offset),
            axes[0] - max(axes[0].shape) * offset,
        )


def radial_average(image: np.ndarray) -> np.ndarray:
    """Compute the radially averaged intensity of an image.

    This function calculates the radially averaged intensity of an input image.
    It first calculates a radial grid of the image using the `hypotenuse_ndim` function,
    then uses this grid to compute the mean intensity in each radial bin.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array.

    Returns
    -------
    np.ndarray
        The radially averaged intensity of the input image.
        (1-D array with bins corresponding to maximum image axis)

    Notes
    -----
    The `radial_average` function operates on an image represented as a numpy array,
    which can have any number of dimensions. The function first creates a radial
    grid of the image using the `hypotenuse_ndim` function, with the size of the grid
    equal to half the maximum size of the input image's axes. The function then computes
    the mean intensity in each bin of this radial grid to obtain the radially
    averaged intensity.
    """
    bins = np.max(image.shape) / 2
    axes = np.ogrid[tuple(slice(0, s) for s in image.shape)]
    r = hypotenuse_ndim(axes)
    rbin = (bins * r / r.max()).astype(int)
    radial_mean = ndimage.mean(image, labels=rbin, index=np.arange(1, rbin.max() + 1))

    return radial_mean


def rotational_kernel(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Create a rotational kernel from an input array.

    This function uses given input array and extends its values
    symmetrically by rotating it to the desired shape

    Parameters
    ----------
    arr : np.ndarray
        Input array used to create the rotational kernel.
        This should be a 1D array as output by radial_average()
    shape : Tuple[int, ...]
        Shape of the desired rotational kernel.

    Returns
    -------
    np.ndarray
        The created rotational kernel.
    """
    func = interp1d(np.arange(len(arr)), arr, bounds_error=False, fill_value=0)

    axes = np.ogrid[tuple(slice(0, np.ceil(s / 2)) for s in shape)]
    kernel = hypotenuse_ndim(axes, offset=0).astype("f4")
    kernel = func(kernel).astype("f4")
    for idx, s in enumerate(shape):
        padding = [(0, 0)] * len(shape)
        padding[idx] = (int(np.floor(s / 2)), 0)

        mode = "reflect" if s % 2 else "symmetric"
        kernel = np.pad(kernel, padding, mode=mode)
    return kernel
