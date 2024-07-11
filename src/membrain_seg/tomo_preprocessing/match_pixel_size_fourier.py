import numpy as np
import torch
from tqdm import tqdm

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
    fourier_cropping_torch,
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
            pixel_size_in=output_pixel_size,
            pixel_size_out=input_pixel_size,
            orig_shape=self.process_patch_size,
        )

        output_shape = determine_output_shape(
            input_pixel_size, output_pixel_size, tomo_shape
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
            int(entry * input_pixel_size / output_pixel_size) for entry in x_positions
        ]
        y_positions_out = [
            int(entry * input_pixel_size / output_pixel_size) for entry in y_positions
        ]
        z_positions_out = [
            int(entry * input_pixel_size / output_pixel_size) for entry in z_positions
        ]

        for k_x, x_pos in tqdm(enumerate(x_positions)):
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
        print("Returning")
        return rescaled_array / weight_array


tomo_path = (
    "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/4Lorenz/Tomo_1/Tomo1L1_bin4.rec"
)
tomo = load_tomogram(tomo_path)
input_pixel_size = float(tomo.voxel_size.x)
process_patch_size = (160, 160, 160)
output_pixel_size = 10.0
# output_pixel_size = input_pixel_size
tomo = tomo.data
tomo = np.array(tomo, dtype=float)
tomo -= np.mean(tomo)
tomo /= np.std(tomo)

rescaler = SW_Rescaler(input_pixel_size, output_pixel_size, process_patch_size)
rescaled_array = rescaler(tomo)
print("Storing")
store_tomogram("./test_rescale.mrc", rescaled_array)
