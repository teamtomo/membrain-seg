import logging
import os

from scipy import ndimage

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)


def match_segmentation_pixel_size_to_tomo(
    seg_path: str, orig_tomo_path: str, output_path: str
) -> None:
    """
    Match the pixel size of the input segmentation to the target tomogram.

    Note: This is normally used after matching your tomogram to the training pixel
    size range. You can use this function to map the resulting segmentation back
    to the original tomogram size.
    Generally, it is important that both input and target tomogram cover the same
    physical extent!

    Parameters
    ----------
    seg_path : str
        The file path to the input segmentation (e.g. .mrc file) to be processed.
    orig_tomo_path : str
        The file path to the target tomogram that the pixel size should be matched to.
    output_path : str
        The file path where the processed segmentation will be stored.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the file specified in `seg_path` or `orig_tomo_path` does not exist.

    Notes
    -----
    This function reads the input segmentation and the target tomogram from the given
    paths, rescales the pixel size of the input segmentation to match that of the
    target tomogram, and stores the processed tomogram to the specified output path.
    The rescaling process is achieved by calculating the rescaling factors for each
    dimension and applying a zoom operation to the input tomogram.
    """
    # Load the input tomogram and its pixel size
    file_path = seg_path
    tomo = load_tomogram(file_path, normalize_data=False)

    # Get output shape from original tomogram
    match_tomo_path = orig_tomo_path
    orig_tomo = load_tomogram(match_tomo_path, normalize_data=False)
    output_shape = orig_tomo.data.shape

    logging.info(
        "Matching input tomogram" + os.path.basename(file_path) + "from shape",
        tomo.data.shape,
        "to shape",
        output_shape,
        ".",
    )

    rescale_factors = [
        target_dim / original_dim
        for target_dim, original_dim in zip(output_shape, tomo.data.shape)
    ]
    resized_data = ndimage.zoom(tomo.data, rescale_factors, order=0, prefilter=False)
    orig_tomo.data = resized_data  # Keep the header of the original data
    store_tomogram(output_path, orig_tomo)
