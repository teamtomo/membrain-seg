import os

from membrain_seg.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)
from scipy import ndimage


def match_segmentation_pixel_size_to_tomo(seg_path, orig_tomo_path, output_path):
    """Match the pixel size if your input tomo to the target."""
    # Load the input tomogram and its pixel size
    file_path = seg_path
    data = load_tomogram(file_path, return_pixel_size=False, normalize_data=False)

    # Get output shape from original tomogram
    match_tomo_path = orig_tomo_path
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
    store_tomogram(output_path, resized_data)
