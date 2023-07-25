# --------------------------------------------------------------------------------
# Copyright (C) 2022 ZauggGroup
#
# This file is a copy (or a modified version) of the original file from the
# following GitHub repository:
#
# Repository: https://github.com/ZauggGroup/DeePiCt
# Original file: https://github.com/ZauggGroup/DeePiCt/blob/main/spectrum_filter/match_spectrum.py
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

from typing import Union

import pandas as pd

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    normalize_tomogram,
    store_tomogram,
)
from membrain_seg.tomo_preprocessing.matching_utils.spec_matching_utils import (
    match_spectrum,
)


def match_amplitude_spectrum_for_files(
    input_path: str,
    target_path: str,
    output_path: str,
    cutoff: Union[int, bool],
    smoothen: Union[int, bool],
    almost_zero_cutoff: Union[int, bool],
    shrink_excessive_value: Union[int, bool],
) -> None:
    """
    Match the input tomogram's spectrum to the target spectrum.

    Parameters
    ----------
    input_path : str
        The file path to the input tomogram to be processed.
    target_path : str
        The file path to the target spectrum.
    output_path : str
        The file path where the processed tomogram will be stored.
    cutoff : int / False
        The cutoff frequency for the spectrum matching process. All frequencies
        above will be set to 0.
        If set to False, no cutoff is performed.
    smoothen : int / False
        The smoothing factor to be applied in the spectrum matching process.
        If set to False, no smoothing is performed.
    almost_zero_cutoff : int / False
        The value below which the amplitude is treated as almost zero during the
        spectrum matching process. The cutoff will then be set to the lowest frequency
        that is below the almost-zero-cutoff.
        If set to False, no almost-zero-cutoff is performed.
    shrink_excessive_value : int / False
        A limit to shrink the excessive amplitude values during the spectrum
        matching process. Serves as a regularization.
        If set to False, excessive values will not be excluded.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the file specified in `input_path` or `target_path` does not exist.

    Notes
    -----
    This function reads the input tomogram and the target spectrum from the given paths,
    matches the amplitude spectrum of the input tomogram to the target spectrum, and
    stores the processed tomogram to the specified output path. The matching process is
    controlled by several parameters including cutoff frequency, smoothing factor, an
    almost zero cutoff, and a shrink factor for excessive values.
    """
    # Read input tomogram
    tomo = load_tomogram(input_path, normalize_data=True)

    # Read target spectrum
    target_spectrum = pd.read_csv(target_path, sep="\t")["intensity"].values

    # Match the amplitude spectrum of the input tomogram to the target spectrum
    filtered_tomo = match_spectrum(
        tomo.data,
        target_spectrum,
        cutoff,
        smoothen,
        almost_zero_cutoff,
        shrink_excessive_value,
    )
    filtered_tomo = normalize_tomogram(filtered_tomo)
    tomo.data = filtered_tomo
    # Save the filtered tomogram to a file
    store_tomogram(output_path, tomo)
