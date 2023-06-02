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


def match_pixel_size(
    input_tomogram, output_path, pixel_size_in, pixel_size_out, disable_smooth
):
    """Match the pixel size if your input tomo to the target."""
    # Load the input tomogram and its pixel size
    file_path = input_tomogram
    data, input_pixel_sizes = load_tomogram(
        file_path, return_pixel_size=True, normalize_data=True
    )
    pixel_size_in = pixel_size_in or input_pixel_sizes.x
    smoothing = not disable_smooth

    print(
        "Matching input tomogram",
        os.path.basename(file_path),
        "from pixel size",
        pixel_size_in,
        "to pixel size",
        pixel_size_out,
        ".",
    )

    # Calculate the output shape after pixel size matching
    output_shape = determine_output_shape(pixel_size_in, pixel_size_out, data.shape)

    # Perform Fourier-based resizing (cropping or extending) using the determined
    # output shape
    if (pixel_size_in / pixel_size_out) < 1.0:
        resized_data = fourier_cropping(data, output_shape, smoothing)
    else:
        resized_data = fourier_extend(data, output_shape, smoothing)

    resized_data = normalize_tomogram(resized_data)
    # Save the resized tomogram to the specified output path
    store_tomogram(output_path, resized_data)
