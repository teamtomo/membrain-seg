import os

import numpy as np
import torch

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    normalize_tomogram,
    store_tomogram,
)
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
    fourier_cropping,
    fourier_cropping_torch,
    fourier_extend,
    fourier_extend_torch,
)


def get_gaussian_kernel(patch_shape, sigma):
    """
    Generate a Gaussian kernel of a given shape and standard deviation.

    Parameters
    ----------
    patch_shape : tuple
        Shape of the patch for which the kernel should be generated.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Gaussian kernel of the given shape and standard deviation.
    """
    kernel = np.zeros(patch_shape, dtype=np.float32)
    for i in range(patch_shape[0]):
        for j in range(patch_shape[1]):
            for k in range(patch_shape[2]):
                kernel[i, j, k] = np.exp(
                    -(
                        (i - patch_shape[0] // 2) ** 2
                        + (j - patch_shape[1] // 2) ** 2
                        + (k - patch_shape[2] // 2) ** 2
                    )
                    / (2 * sigma**2)
                )
    return kernel


class SW_Rescaler(torch.nn.Module):
    """Rescale a tomogram using sliding window inference."""

    def __init__(self, input_pixel_size, output_pixel_size, process_patch_size):
        super().__init__()
        self.input_pixel_size = input_pixel_size
        self.output_pixel_size = output_pixel_size
        self.process_patch_size = process_patch_size
        self.crop = self.input_pixel_size < self.output_pixel_size
        self.gaussian_kernel = get_gaussian_kernel(
            process_patch_size, 0.25 * process_patch_size[0]
        )
        self.gaussian_kernel = np.ones_like(self.gaussian_kernel)

    def forward(self, tomogram: np.ndarray):
        """Rescale the input tomogram using sliding window inference."""
        tomo_shape = tomogram.shape
        crop_shape = determine_output_shape(
            pixel_size_in=self.output_pixel_size,
            pixel_size_out=self.input_pixel_size,
            orig_shape=self.process_patch_size,
        )

        output_shape = determine_output_shape(
            self.input_pixel_size, self.output_pixel_size, tomo_shape
        )

        rescaled_array = np.zeros(output_shape, dtype=np.float32)
        weight_array = np.zeros(output_shape, dtype=np.float32) + 0.001

        step_sizes = (crop_shape[0] // 2, crop_shape[1] // 2, crop_shape[2] // 2)

        x_positions = [
            *list(range(0, tomo_shape[0] - crop_shape[0], step_sizes[0])),
            tomo_shape[0] - crop_shape[0],
        ]
        y_positions = [
            *list(range(0, tomo_shape[1] - crop_shape[1], step_sizes[1])),
            tomo_shape[1] - crop_shape[1],
        ]
        z_positions = [
            *list(range(0, tomo_shape[2] - crop_shape[2], step_sizes[2])),
            tomo_shape[2] - crop_shape[2],
        ]

        x_positions_out = [
            int(entry * self.input_pixel_size / self.output_pixel_size)
            for entry in x_positions
        ]
        y_positions_out = [
            int(entry * self.input_pixel_size / self.output_pixel_size)
            for entry in y_positions
        ]
        z_positions_out = [
            int(entry * self.input_pixel_size / self.output_pixel_size)
            for entry in z_positions
        ]

        for k_x, x_pos in enumerate(x_positions):
            for k_y, y_pos in enumerate(y_positions):
                for k_z, z_pos in enumerate(z_positions):
                    crop = tomogram[
                        x_pos : x_pos + crop_shape[0],
                        y_pos : y_pos + crop_shape[1],
                        z_pos : z_pos + crop_shape[2],
                    ]
                    crop = torch.tensor(crop, dtype=torch.float32)
                    if self.crop:
                        rescaled = (
                            fourier_cropping_torch(crop, self.process_patch_size)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        rescaled = (
                            fourier_extend_torch(crop, self.process_patch_size)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    rescaled_array[
                        x_positions_out[k_x] : x_positions_out[k_x]
                        + self.process_patch_size[0],
                        y_positions_out[k_y] : y_positions_out[k_y]
                        + self.process_patch_size[1],
                        z_positions_out[k_z] : z_positions_out[k_z]
                        + self.process_patch_size[2],
                    ] += (
                        rescaled * self.gaussian_kernel
                    )
                    weight_array[
                        x_positions_out[k_x] : x_positions_out[k_x]
                        + self.process_patch_size[0],
                        y_positions_out[k_y] : y_positions_out[k_y]
                        + self.process_patch_size[1],
                        z_positions_out[k_z] : z_positions_out[k_z]
                        + self.process_patch_size[2],
                    ] += self.gaussian_kernel
        return rescaled_array / weight_array


def rescale_entire_tomogram(
    tomo: np.ndarray, pixel_size_in: float, pixel_size_out: float, smoothing: bool
):
    """Rescale the entire tomogram using Fourier-based resizing."""
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
    return resized_data


def rescale_sliding_window(
    tomo: np.ndarray,
    pixel_size_in: float,
    pixel_size_out: float,
):
    """Rescale a tomogram using sliding window inference."""
    rescaler = SW_Rescaler(
        pixel_size_in, pixel_size_out, process_patch_size=(160, 160, 160)
    )
    rescaled_array = rescaler(tomo)
    return rescaled_array


def match_pixel_size(
    input_tomogram: str,
    output_path: str,
    pixel_size_in: float,
    pixel_size_out: float,
    disable_smooth: bool,
    use_sliding_window: bool,
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
    use_sliding_window : bool
        If True, sliding window inference will be used for resizing.

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
    tomo.data = normalize_tomogram(tomo.data)
    if use_sliding_window:
        resized_data = rescale_sliding_window(tomo.data, pixel_size_in, pixel_size_out)
    else:
        resized_data = rescale_entire_tomogram(
            tomo.data, pixel_size_in, pixel_size_out, smoothing
        )

    resized_data = normalize_tomogram(resized_data)
    tomo.data = resized_data
    # Save the resized tomogram to the specified output path
    store_tomogram(output_path, tomo, voxel_size=pixel_size_out)
