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


import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d


def hypot_nd(axes, offset=0.5):
    """Function to compute the hypotenuse for n-dimensional axes."""
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


def rad_avg(image):
    """Compute the radially averaged intensity of an image."""
    bins = np.max(image.shape) / 2
    axes = np.ogrid[tuple(slice(0, s) for s in image.shape)]
    r = hypot_nd(axes)
    rbin = (bins * r / r.max()).astype(int)
    radial_mean = ndimage.mean(image, labels=rbin, index=np.arange(1, rbin.max() + 1))

    return radial_mean


def rot_kernel(arr, shape):
    """Create a rotational kernel from an input array."""
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
