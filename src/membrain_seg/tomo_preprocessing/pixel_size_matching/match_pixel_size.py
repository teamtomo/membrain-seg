import os

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    normalize_tomogram,
    store_tomogram,
)
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
    fourier_cropping,
    fourier_extend,
)


def match_pixel_size(
    input_tomogram: str,
    output_path: str,
    pixel_size_in: float,
    pixel_size_out: float,
    disable_smooth: bool,
) -> None:
    """
    Match the pixel size of the input tomogram to the target pixel size.

    Parameters
    ----------
    input_tomogram : str
        The file path to the input tomogram to be processed.
    output_path : str
        The file path where the processed tomogram will be stored.
    pixel_size_in : float
        The pixel size of the input tomogram. If None, it will be read
        from the tomogram file.
        ATTENTION: This can lead to errors if the header is not correct.
    pixel_size_out : float
        The target pixel size.
    disable_smooth : bool
        If True, smoothing will be disabled in the Fourier-based resizing process.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the file specified in `input_tomogram` does not exist.

    Notes
    -----
    This function reads the input tomogram from the given path, rescales the pixel size
    of the tomogram to match the target pixel size, and stores the processed tomogram to
    the specified output path. The rescaling process is achieved by calculating the
    output shape based on the input and target pixel sizes and performing a
    Fourier-based resizing (either cropping or extending) on the input tomogram.
    If `disable_smooth` is True, smoothing is disabled in the resizing process.
    """
    # Load the input tomogram and its pixel size
    file_path = input_tomogram
    tomo = load_tomogram(file_path, normalize_data=True)
    pixel_size_in = pixel_size_in or tomo.voxel_size.x
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
    output_shape = determine_output_shape(
        pixel_size_in, pixel_size_out, tomo.data.shape
    )

    # Perform Fourier-based resizing (cropping or extending) using the determined
    # output shape
    if (pixel_size_in / pixel_size_out) < 1.0:
        resized_data = fourier_cropping(tomo.data, output_shape, smoothing)
    elif (pixel_size_in / pixel_size_out) > 1.0:
        resized_data = fourier_extend(tomo.data, output_shape, smoothing)
    else:
        resized_data = tomo.data

    resized_data = normalize_tomogram(resized_data)
    tomo.data = resized_data
    # Save the resized tomogram to the specified output path
    store_tomogram(output_path, tomo, voxel_size=pixel_size_out)
