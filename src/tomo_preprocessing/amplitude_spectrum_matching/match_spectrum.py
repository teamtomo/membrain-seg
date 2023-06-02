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


import pandas as pd
from membrain_seg.dataloading.data_utils import (
    load_tomogram,
    normalize_tomogram,
    store_tomogram,
)

from tomo_preprocessing.matching_utils.spec_matching_utils import match_spectrum


def match_amplitude_spectrum(
    input_path,
    target_path,
    output_path,
    cutoff,
    smoothen,
    almost_zero_cutoff,
    shrink_excessive_value,
):
    """Match the input tomogram's spectrum to the target spectrum."""
    # Read input tomogram
    tomo = load_tomogram(input_path, normalize_data=True)

    # Read target spectrum
    target_spectrum = pd.read_csv(target_path, sep="\t")["intensity"].values

    # Match the amplitude spectrum of the input tomogram to the target spectrum
    filtered_tomo = match_spectrum(
        tomo,
        target_spectrum,
        cutoff,
        smoothen,
        almost_zero_cutoff,
        shrink_excessive_value,
    )
    filtered_tomo = normalize_tomogram(filtered_tomo)
    # Save the filtered tomogram to a file
    store_tomogram(output_path, filtered_tomo)
