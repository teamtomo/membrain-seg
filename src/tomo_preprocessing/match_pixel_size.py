import argparse
import os

from membrain_seg.dataloading.data_utils import (
    load_tomogram,
    normalize_tomogram,
    store_tomogram,
)

from tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
    fourier_cropping,
    fourier_extend,
)


def main():
    """Match the pixel size if your input tomo to the target."""
    # Parse the command-line arguments
    parser = get_cli()
    args = parser.parse_args()

    # Load the input tomogram and its pixel size
    file_path = args.input_tomogram
    data, input_pixel_sizes = load_tomogram(
        file_path, return_pixel_size=True, normalize_data=True
    )
    pixel_size_in = args.pixel_size_in or input_pixel_sizes.x
    smoothing = not args.disable_smooth

    print(
        "Matching input tomogram",
        os.path.basename(file_path),
        "from pixel size",
        pixel_size_in,
        "to pixel size",
        args.pixel_size_out,
        ".",
    )

    # Calculate the output shape after pixel size matching
    output_shape = determine_output_shape(
        pixel_size_in, args.pixel_size_out, data.shape
    )

    # Perform Fourier-based resizing (cropping or extending) using the determined
    # output shape
    if (pixel_size_in / args.pixel_size_out) < 1.0:
        resized_data = fourier_cropping(data, output_shape, smoothing)
    else:
        resized_data = fourier_extend(data, output_shape, smoothing)

    resized_data = normalize_tomogram(resized_data)
    # Save the resized tomogram to the specified output path
    store_tomogram(args.output_path, resized_data)


def get_cli():
    """Command line interface parser."""
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Match tomogram pixel size")
    parser.add_argument("input_tomogram", help="Path to the input tomogram")
    parser.add_argument("output_path", help="Path to store the output files")
    parser.add_argument(
        "--pixel_size_out",
        type=float,
        default=10.0,
        help="Target pixel size (default: 20.0)",
    )
    parser.add_argument(
        "--pixel_size_in",
        type=float,
        default=None,
        help="Input pixel size (optional). If not specified, it will be read"
        "from the tomogram's header. ATTENTION: This can lead to severe errors if the"
        "header pixel size is not correct.",
    )
    parser.add_argument(
        "--disable_smooth",
        type=bool,
        default=False,
        help="Disable smoothing (ellipsoid mask + cosine decay). Disable if "
        "causing problems or for speed up",
    )


if __name__ == "__main__":
    main()
