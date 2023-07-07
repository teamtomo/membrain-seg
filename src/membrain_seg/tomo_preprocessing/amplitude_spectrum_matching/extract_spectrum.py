# --------------------------------------------------------------------------------
# Copyright (C) 2022 ZauggGroup
#
# This file is a copy (or a modified version) of the original file from the
# following GitHub repository:
#
# Repository: https://github.com/ZauggGroup/DeePiCt
# Original file: https://github.com/ZauggGroup/DeePiCt/blob/main/spectrum_filter/extract_spectrum.py
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


from membrain_seg.segmentation.dataloading.data_utils import normalize_tomogram
from membrain_seg.tomo_preprocessing.matching_utils.spec_matching_utils import (
    extract_spectrum,
)


def extract_spectrum_from_file(input_path: str, output_path: str) -> None:
    """
    Extract the radially averaged Fourier spectrum from the target tomogram.

    Parameters
    ----------
    input_path : str
        The file path to the input tomogram to be processed.
    output_path : str
        The file path where the extracted spectrum will be stored.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the file specified in `input_path` does not exist.

    Notes
    -----
    This function reads the input tomogram from the given path, extracts its Fourier
    amplitude spectrum, and stores the spectrum to the specified output path.
    The extracted spectrum is saved as a CSV file.
    """
    # Parse command line arguments.

    # Read input tomogram.
    tomo = input_path
    tomo = normalize_tomogram(tomo)

    # Extract amplitude spectrum.
    spectrum = extract_spectrum(tomo)

    # Save the spectrum to a file
    spectrum.to_csv(output_path, sep="\t", header=["intensity"], index_label="freq")
