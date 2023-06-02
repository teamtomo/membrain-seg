import argparse
import os

from membrain_seg.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)
from scipy import ndimage


def main():
    """Match the pixel size if your input tomo to the target."""
    # Parse the command-line arguments
    parser = get_cli()
    args = parser.parse_args()

    # Load the input tomogram and its pixel size
    file_path = args.seg_path
    data = load_tomogram(file_path, return_pixel_size=False, normalize_data=False)

    # Get output shape from original tomogram
    match_tomo_path = args.orig_tomo_path
    orig_tomo = load_tomogram(
        match_tomo_path, return_pixel_size=False, normalize_data=False
    )
    output_shape = orig_tomo.shape

    print(
        "Matching input tomogram",
        os.path.basename(file_path),
        "from shape",
        data.shape,
        "to shape",
        output_shape,
        ".",
    )

    rescale_factors = [
        target_dim / original_dim
        for target_dim, original_dim in zip(output_shape, data.shape)
    ]
    resized_data = ndimage.zoom(data, rescale_factors, order=1)
    print(resized_data.shape, output_shape)
    # Save the resized tomogram to the specified output path
    store_tomogram(args.output_path, resized_data)


def get_cli():
    """Command line interface parser."""
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Match tomogram pixel size")
    parser.add_argument("seg_path", help="Path to the segmentation")
    parser.add_argument(
        "orig_tomo_path",
        help="Path to the tomogram the segmentation should be matched to.",
    )
    parser.add_argument(
        "output_path",
        help="Path to the where the reszied segmentation should be stored.",
    )
    return parser


if __name__ == "__main__":
    main()
