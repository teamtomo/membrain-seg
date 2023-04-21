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

import argparse
from membrain_seg.dataloading.data_utils import load_tomogram
from tomo_preprocessing.matching_utils.spec_matching_utils import extract_spectrum



def main():
    # Parse command line arguments.
    parser = get_cli()
    args = parser.parse_args()
    
    # Read input tomogram.
    tomo = args.input

    # Extract amplitude spectrum.
    spectrum = extract_spectrum(tomo)

    # Save the spectrum to a file
    spectrum.to_csv(args.output, sep="\t", header=["intensity"], index_label="freq")


def get_cli():
    """ Function to set up the command line interface. """
    
    parser = argparse.ArgumentParser(
        description="Extract radially averaged amplitude spectrum from cryo-ET data."
    )

    parser.add_argument( 
        "-i",
        "--input",
        required=True,
        type=load_tomogram,
        help="Tomogram to extract spectrum from (.mrc/.rec format)"
    )

    parser.add_argument( 
        "-o",
        "--output",
        required=True,
        help="Output destination for extracted spectrum (.tsv format)"
    )
    
    return parser


if __name__ == "__main__":
    main()