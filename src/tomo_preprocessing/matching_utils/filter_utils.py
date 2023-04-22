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


def hypot_nd(axes: List[np.ndarray], offset: float = 0.5) -> np.ndarray:
    """Function to compute the hypotenuse for n-dimensional axes.

    This is used to compute the distance of each voxel to the center point.
    Each 2D hypotenuse is computed by a^2 + b^2 = c^2
    Thus, this also gives the Euclidean distance of a 2D point.
    This is extended recursively to any dimension.

    But why do we not just compute the Euclidean distance??

    Parameters
    ----------
    axes : List[np.ndarray]
        List of axes in n-dimensional space.
    offset : float, optional
        Offset to apply before calculating the hypotenuse, by default 0.5.


    Returns
    -------
    np.ndarray
        The computed hypotenuse.
    """
    if len(axes) == 2:
        return np.hypot(
            axes[0] - max(axes[0].shape) * offset,
            axes[1] - max(axes[1].shape) * offset,
        )
    else:
        return np.hypot(
            hypot_nd(axes[1:], offset),
            axes[0] - max(axes[0].shape) * offset,
        )


def rad_avg(image: np.ndarray) -> np.ndarray:
    """Compute the radially averaged intensity of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array.

    Returns
    -------
    np.ndarray
        The radially averaged intensity of the input image.
        (1-D array with bins corresponding to maximum image axis)
    """
    bins = np.max(image.shape) / 2
    axes = np.ogrid[tuple(slice(0, s) for s in image.shape)]
    r = hypot_nd(axes)
    rbin = (bins * r / r.max()).astype(int)
    radial_mean = ndimage.mean(image, labels=rbin, index=np.arange(1, rbin.max() + 1))

    return radial_mean


def rot_kernel(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Create a rotational kernel from an input array.

    This function uses given input array and extends its values
    symmetrically by rotating it to the desired shape

    Parameters
    ----------
    arr : np.ndarray
        Input array used to create the rotational kernel.
        This should be a 1D array as output by rad_avg()
    shape : Tuple[int, ...]
        Shape of the desired rotational kernel.

    Returns
    -------
    np.ndarray
        The created rotational kernel.
    """
    func = interp1d(np.arange(len(arr)), arr, bounds_error=False, fill_value=0)

    axes = np.ogrid[tuple(slice(0, np.ceil(s / 2)) for s in shape)]
    kernel = hypot_nd(axes, offset=0).astype("f4")
    kernel = func(kernel).astype("f4")
    for idx, s in enumerate(shape):
        padding = [(0, 0)] * len(shape)
        padding[idx] = (int(np.floor(s / 2)), 0)

        mode = "reflect" if s % 2 else "symmetric"
        kernel = np.pad(kernel, padding, mode=mode)
    return kernel
