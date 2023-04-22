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

import argparse

import pandas as pd
from membrain_seg.dataloading.data_utils import load_tomogram, store_tomogram

from tomo_preprocessing.matching_utils.spec_matching_utils import match_spectrum


def main():
    """Match the input tomogram's spectrum to the target spectrum."""
    # Parse command line arguments
    parser = get_cli()
    args = parser.parse_args()

    # Read input tomogram
    tomo = load_tomogram(args.input)

    # Read target spectrum
    target_spectrum = pd.read_csv(args.target, sep="\t")["intensity"].values

    # Match the amplitude spectrum of the input tomogram to the target spectrum
    filtered_tomo = match_spectrum(tomo, target_spectrum, args.cutoff, args.smoothen)

    # Save the filtered tomogram to a file
    store_tomogram(args.output, filtered_tomo)


def get_cli():
    """Set up the command line interface."""
    parser = argparse.ArgumentParser(
        description="Match tomogram to another tomogram's amplitude spectrum."
    )

    parser.add_argument(
        "-i", "--input", required=True, help="Tomogram to match (.mrc/.rec)"
    )

    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Target spectrum to match the input tomogram to (.tsv)",
    )

    parser.add_argument(
        "-o", "--output", required=True, help="Output location for matched tomogram"
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        required=False,
        default=False,
        type=int,
        help="Lowpass cutoff to apply",
    )

    parser.add_argument(
        "-s",
        "--smoothen",
        required=False,
        default=0,
        type=float,
        help="Smoothening to apply to lowpass filter. Value roughly resembles sigmoid"
        " width in pixels",
    )

    return parser


if __name__ == "__main__":
    main()
