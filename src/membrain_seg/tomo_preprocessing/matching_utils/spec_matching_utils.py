# --------------------------------------------------------------------------------
# Copyright (C) 2022 ZauggGroup
#
# This file is a copy (or a modified version) of the original file from the
# following GitHub repository:
#
# Repository: https://github.com/ZauggGroup/DeePiCt
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


import warnings
from typing import Optional, Union

import numpy as np
import numpy.fft as fft
import pandas as pd
from skimage.util import img_as_float32

from membrain_seg.tomo_preprocessing.matching_utils.filter_utils import (
    radial_average,
    rotational_kernel,
)


def extract_spectrum(tomo: np.ndarray) -> pd.Series:
    """
    Extract the radially averaged amplitude spectrum from the input tomogram.

    This function normalizes the input tomogram to fall between 0 and 1,
    computes the absolute value of its Fourier transform, and then performs
    a radial average of the transformed data. The resulting spectrum is
    returned as a pandas Series.

    Parameters
    ----------
    tomo : np.ndarray
        Input tomogram as a 3D numpy array.

    Returns
    -------
    pd.Series
        Radially averaged amplitude spectrum as a pandas Series.

    Notes
    -----
    This function uses the FFT algorithms from numpy.fft for performing the Fourier
    transform, and it assumes that the input tomogram has voxel intensities that
    can be converted to floating point numbers.

    """
    # Normalize input tomogram intensities.
    tomo = img_as_float32(tomo)
    tomo -= tomo.min()
    tomo /= tomo.max()

    # Compute absolute value of Fourier transform of the normalized tomogram.
    t = fft.fftn(tomo)
    t = fft.fftshift(t)
    t = np.abs(t)

    # Compute the radially averaged amplitude spectrum.
    spectrum = radial_average(t)
    spectrum = pd.Series(spectrum, index=np.arange(len(spectrum)))

    return spectrum


def match_spectrum(
    tomo: np.ndarray,
    target_spectrum: np.ndarray,
    cutoff: Optional[int] = None,
    smooth: Union[float, int] = 0,
    almost_zero_cutoff: bool = True,
    shrink_excessive_value: Optional[int] = None,
) -> np.ndarray:
    """
    Match the amplitude spectrum of the input tomogram to the target spectrum.

    This function performs spectrum matching in the Fourier domain. The input
    tomogram is first Fourier-transformed and its amplitude spectrum is extracted.
    An equalization vector is computed from the ratio of the target spectrum to the
    input spectrum. The function then applies this equalization vector back to the
    Fourier-transformed tomogram, and performs inverse Fourier transformation to
    obtain the tomogram with a matched spectrum.

    Parameters
    ----------
    tomo : np.ndarray
        Input tomogram as a 3D numpy array.
    target_spectrum : np.ndarray
        Target amplitude spectrum as a 1D numpy array.
    cutoff : Optional[int], default=None
        Frequency index at which to apply the low-pass filter. The frequencies
        higher than this value will be removed.
    smooth : Union[float, int], default=0
        Smoothing factor for the low-pass filter. The higher the factor, the
        smoother the transition will be between the retained and removed frequencies.
    almost_zero_cutoff : bool, default=True
        If True, the function will automatically find a cutoff frequency based on
        the minimum frequency index where the input or target spectrum approaches zero.
    shrink_excessive_value : Optional[int], default=None
        If specified, any values in the equalization vector that are greater than
        this value will be replaced with this value. This is used to avoid applying
        excessive amplification to some frequencies.

    Returns
    -------
    np.ndarray
        The filtered tomogram as a 3D numpy array. The voxel intensities in the
        filtered tomogram now have a matched spectrum to the target spectrum.

    Notes
    -----
    This function uses the FFT algorithms from numpy.fft for performing the Fourier
    transformations. It also assumes that the input tomogram and target spectrum have
    intensities and amplitudes that can be converted to floating point numbers.

    """
    # Make a copy of the target spectrum and normalize the input tomogram
    target_spectrum = target_spectrum.copy()
    tomo = img_as_float32(tomo)
    tomo -= tomo.min()
    tomo /= tomo.max()

    # Compute the Fourier transform of the normalized tomogram
    t = fft.fftn(tomo)
    t = fft.fftshift(t)

    # Free memory occupied by the input tomogram
    del tomo

    # Compute the radially averaged amplitude spectrum of the input tomogram
    input_spectrum = radial_average(np.abs(t))

    # Resize the target spectrum to match the input spectrum's length
    target_spectrum = np.resize(target_spectrum, len(input_spectrum))

    almost_zeros_input = np.argwhere(input_spectrum < 1e-1)
    almost_zeros_target = np.argwhere(target_spectrum < 1e-4)
    if len(almost_zeros_input) == 0:
        almost_zeros_input = np.array([99999])
    if len(almost_zeros_target) == 0:
        almost_zeros_target = np.array([99999])

    almost_zero_cutoff_value = np.maximum(
        np.minimum(np.min(almost_zeros_input) - 4, np.min(almost_zeros_target) - 4), 0
    )

    # Compute the equalization vector
    equal_v = target_spectrum / input_spectrum
    if almost_zero_cutoff:
        if not cutoff:
            cutoff = almost_zero_cutoff_value
        else:
            cutoff = np.minimum(cutoff, almost_zero_cutoff_value)
    # Apply cutoff and smoothing if specified
    if cutoff:
        if smooth:
            slope = len(equal_v) / smooth
            offset = 2 * slope * ((cutoff - len(equal_v) / 2) / len(equal_v)) - 8
            cutoff_v = 1 / (
                1 + np.exp(np.linspace(-slope, slope, len(equal_v)) - offset)
            )
            try:
                equal_v[cutoff:] = 0
            except IndexError:
                warnings.warn("Flat cutoff is higher than maximum frequency")

        else:
            print("Were shrinking")
            cutoff_v = np.ones_like(equal_v)
            try:
                equal_v[cutoff:] = 0
            except IndexError:
                warnings.warn("Flat cutoff is higher than maximum frequency")

        equal_v *= cutoff_v

    if shrink_excessive_value:
        equal_v[equal_v > shrink_excessive_value] = shrink_excessive_value
    # Create the equalization kernel
    equal_kernel = rotational_kernel(equal_v, t.shape)

    # Apply the equalization kernel to the input tomogram's Fourier transform
    t *= equal_kernel
    del equal_kernel

    # Compute the inverse Fourier transform and return the filtered tomogram
    t = fft.ifftn(t)
    t = np.abs(t)
    t = img_as_float32(t)

    return t
