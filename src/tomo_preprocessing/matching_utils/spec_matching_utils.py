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

import numpy as np
import numpy.fft as fft
import pandas as pd
from tomo_preprocessing.matching_utils.FilterUtils import rad_avg, rot_kernel

def extract_spectrum(tomo):
    """ Extract radially averaged amplitude spectrum. """
    # Normalize input tomogram intensities.
    tomo -= tomo.min()
    tomo /= tomo.max()

    # Compute absolute value of Fourier transform of the normalized tomogram.
    t = fft.fftn(tomo)
    t = fft.fftshift(t)
    t = np.abs(t)

    # Compute the radially averaged amplitude spectrum.
    spectrum = rad_avg(t)
    spectrum = pd.Series(spectrum, index = np.arange(len(spectrum)))

    return spectrum

def match_spectrum(tomo, target_spectrum, cutoff=None, smooth=0):
    """ Match the amplitude spectrum of the input tomogram to the target spectrum. """

    # Make a copy of the target spectrum and normalize the input tomogram
    target_spectrum = target_spectrum.copy()
    tomo -= tomo.min()
    tomo /= tomo.max()

    # Compute the Fourier transform of the normalized tomogram
    t = fft.fftn(tomo)
    t = fft.fftshift(t)

    # Free memory occupied by the input tomogram
    del tomo

    # Compute the radially averaged amplitude spectrum of the input tomogram
    input_spectrum = rad_avg(np.abs(t))

    # Resize the target spectrum to match the input spectrum's length
    target_spectrum.resize(len(input_spectrum))

    # Compute the equalization vector
    equal_v = target_spectrum / input_spectrum


    # Apply cutoff and smoothing if specified
    if cutoff:
        if smooth:
            slope = len(equal_v)/smooth
            offset = 2 * slope * ((cutoff - len(equal_v) / 2) / len(equal_v))
        
            cutoff_v = 1/(1 + np.exp(np.linspace(-slope, slope, len(equal_v)) - offset))

        else:
            cutoff_v = np.ones_like(equal_v)
            try:
                equal_v[cutoff:] = 0
            except IndexError:
                warnings.warn("Flat cutoff is higher than maximum frequency")
            
        equal_v *= cutoff_v

    # Create the equalization kernel
    equal_kernel = rot_kernel(equal_v, t.shape)
    
    # Apply the equalization kernel to the input tomogram's Fourier transform
    t *= equal_kernel
    del equal_kernel
    
    # Compute the inverse Fourier transform and return the filtered tomogram
    t = fft.ifftn(t)
    t = np.abs(t).astype("f4")

    return t