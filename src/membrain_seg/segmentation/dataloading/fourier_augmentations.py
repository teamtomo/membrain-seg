from typing import Tuple

import numpy as np
import numpy.fft as fft
import torch
from monai.transforms import (
    Randomizable,
    Transform,
)
from scipy.ndimage import gaussian_filter1d

from membrain_seg.segmentation.dataloading.transforms import sample_scalar
from membrain_seg.tomo_preprocessing.matching_utils.filter_utils import (
    rotational_kernel,
)


def wedge_mask(shape: Tuple[int, int, int], angle: float) -> np.ndarray:
    """
    Generates a 3D wedge mask based on the given shape and angle.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of the 3D mask to generate.
    angle : float
        Angle of the wedge in degrees.
        (Angles to keep; e.g. 90 degrees will keep the entire image,
        0 degrees will keep nothing.)

    Returns
    -------
    np.ndarray
        A boolean mask with the same shape as input, where 1 indicates the
        region inside the wedge and 0 indicates outside.
    """
    # Create the 3D coordinate grid
    x_coords, _, z_coords = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    x_center = np.mean(x_coords)
    z_center = np.mean(z_coords)

    angle_radians = np.radians(angle)
    m = np.tan(angle_radians)  # slope
    b = z_center  # intercept

    def above_wedge(x):
        return m * x + b

    x_left_side = x_coords[: int(x_coords.shape[0] / 2), :, :]
    x_right_side = x_coords[int(x_coords.shape[0] / 2) :, :, :]

    x_right_above_wedge = above_wedge(x_right_side - x_center)
    x_left_above_wedge = above_wedge(x_center - x_left_side)
    x_right_below_wedge = above_wedge(x_center - x_right_side)
    x_left_below_wedge = above_wedge(x_left_side - x_center)

    x_above_wedge = np.concatenate((x_left_above_wedge, x_right_above_wedge), axis=0)
    x_below_wedge = np.concatenate((x_left_below_wedge, x_right_below_wedge), axis=0)
    above_wedge_mask = x_above_wedge < z_coords - 1.5
    below_wedge_mask = x_below_wedge > z_coords + 1.5

    wedge_mask = above_wedge_mask + below_wedge_mask
    wedge_mask = wedge_mask > 0
    wedge_mask[
        int(x_coords.shape[0] / 2),
        int(x_coords.shape[2] / 2),
        int(x_coords.shape[2] / 2),
    ] = 1
    return ~wedge_mask


def get_line_plot(
    n_points: int, smooth_sigma: float, step_sigma: float, offset: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a line plot.

    This function applies a Gaussian smoothing filter to a random walk
    series, then offsets the result.

    Parameters
    ----------
    n_points : int
        Number of points in the line plot.
    smooth_sigma : float
        Standard deviation for Gaussian kernel, used in smoothing.
    step_sigma : float
        Standard deviation of the step sizes in the random walk.
    offset : float
        Offset to add to the smoothed random walk.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x values (range from 0 to n_points-1) and the corresponding y values
        of the line plot.
    """
    # Generate an array of x values from 0 to n_points
    x = np.linspace(0, n_points - 1, n_points)

    # Generate an array of y values using a random walk
    y = np.cumsum(np.random.randn(n_points) * step_sigma)

    # Apply Gaussian smoothing
    y_smoothed = gaussian_filter1d(y, sigma=smooth_sigma)
    return x, np.abs(y_smoothed + offset)


def normalize_and_fft_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalizes the given patch and applies the Fast Fourier Transform to it.

    Parameters
    ----------
    patch : np.ndarray
        2D or 3D data patch to be transformed.

    Returns
    -------
    np.ndarray
        FFT-transformed and normalized patch.
    """
    patch -= patch.min()
    patch /= patch.max()
    t = fft.fftn(patch)
    t = fft.fftshift(t)
    return t


def fft_patch_to_real(fft_patch: np.ndarray) -> np.ndarray:
    """
    Converts a patch from FFT space back to real space.

    Parameters
    ----------
    fft_patch : np.ndarray
        Patch in FFT space to be converted.

    Returns
    -------
    np.ndarray
        Converted real space patch.
    """
    fft_patch = fft.fftshift(fft_patch)  # Added this. Do I need it? What happens?
    t = fft.ifftn(fft_patch)
    t = np.real(t)
    # t = np.abs(t).astype("f4")
    return t


class MissingWedgeMaskAndFourierAmplitudeMatchingCombined(Randomizable, Transform):
    """
    A class for combining missing wedge mask and Fourier amplitude augmentations.

    Parameters
    ----------
    keys : list
        Keys to extract from input data for transformation.
    amplitude_aug : bool, optional
        Whether to perform amplitude augmentation.
    missing_wedge_aug : bool, optional
        Whether to perform missing wedge augmentation.
    smooth_sigma_range : Tuple[float, float], optional
        Range of standard deviation for Gaussian smoothing
        in amplitude augmentation.
    step_sigma_range : Tuple[float, float], optional
        Range of standard deviation for step sizes
        in amplitude augmentation.
    offset_range : Tuple[float, float], optional
        Range of offset values in amplitude augmentation.
    missing_angle_range : Tuple[float, float], optional
        Range of angles for missing wedge augmentation.
    missing_wedge_prob : float, optional
        Probability of applying missing wedge augmentation.
    amplitude_prob : float, optional
        Probability of applying amplitude augmentation.
    sample_kernel_prob : float, optional
        Probability of sampling a kernel for interpolation
        between original and modified patches.
    scale : callable, optional
        Function to determine the scaling factor in kernel generation.
    loc : Tuple[float, float], optional
        Location parameter for kernel generation.
    """

    def __init__(
        self,
        keys,
        amplitude_aug=True,
        missing_wedge_aug=True,
        smooth_sigma_range=(2.0, 4.0),
        step_sigma_range=(0.1, 4.0),
        offset_range=(2.0, 10.0),
        missing_angle_range=(55.0, 88.0),
        missing_wedge_prob=0.5,
        amplitude_prob=0.5,
        sample_kernel_prob=0.0,
        scale=lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
        loc=(0, 1),
    ):
        self.keys = keys

        self.amplitude_aug = amplitude_aug
        self.smooth_sigma_range = smooth_sigma_range
        self.step_sigma_range = step_sigma_range
        self.offset_range = offset_range

        self.missing_wedge_aug = missing_wedge_aug
        self.missing_angle_range = missing_angle_range

        self.amplitude_prob = amplitude_prob
        self.missing_wedge_prob = missing_wedge_prob

        self.sample_kernel_prob = sample_kernel_prob
        self.scale = scale
        self.loc = loc

    def randomize(self, data: None) -> None:
        """Randomly draw augmentation parameters."""
        if isinstance(self.smooth_sigma_range, float):
            self.smooth_sigma = self.smooth_sigma_range
        elif isinstance(self.smooth_sigma_range, tuple):
            self.smooth_sigma = np.random.uniform(
                self.smooth_sigma_range[0], self.smooth_sigma_range[1]
            )

        if isinstance(self.step_sigma_range, float):
            self.step_sigma = self.step_sigma_range
        elif isinstance(self.step_sigma_range, tuple):
            self.step_sigma = np.random.uniform(
                self.step_sigma_range[0], self.step_sigma_range[1]
            )

        if isinstance(self.offset_range, float):
            self.offset = self.offset_range
        elif isinstance(self.smooth_sigma_range, tuple):
            self.offset = np.random.uniform(self.offset_range[0], self.offset_range[1])

        if isinstance(self.missing_angle_range, float):
            self.missing_angle = self.missing_angle_range
        elif isinstance(self.smooth_sigma_range, tuple):
            self.missing_angle = np.random.uniform(
                self.missing_angle_range[0], self.missing_angle_range[1]
            )

        self._do_mw_transform = self.R.random() < self.missing_wedge_prob
        self._do_amp_transform = self.R.random() < self.amplitude_prob
        self._do_sample_kernel = self.R.random() < self.sample_kernel_prob
        if self._do_mw_transform:
            self._do_sample_kernel = True

    def __call__(self, data):
        """Apply both transform in one go."""
        self.randomize(None)
        if self.missing_wedge_aug + self.amplitude_aug == 0:
            return data
        for key in self.keys:
            for c in range(data[key].shape[0]):
                patch = data[key][c]
                fft_patch = normalize_and_fft_patch(patch)
                if self.amplitude_aug and self._do_amp_transform:
                    _, fake_spectrum = get_line_plot(
                        n_points=int(patch.shape[0] / 2),
                        smooth_sigma=self.smooth_sigma,
                        step_sigma=self.step_sigma,
                        offset=self.offset,
                    )
                    equal_kernel = rotational_kernel(fake_spectrum, patch.shape)
                    fft_patch *= equal_kernel
                if self.missing_wedge_aug and self._do_mw_transform:
                    missing_wedge_mask = wedge_mask(patch.shape, self.missing_angle)
                    fft_patch[~missing_wedge_mask] = 0.0
                real_patch = fft_patch_to_real(fft_patch)

                real_patch -= real_patch.mean()
                real_patch /= real_patch.std()
                # Sample a kernel and interpolate between original and modified patch
                if self._do_sample_kernel:
                    patch -= patch.mean()
                    patch /= patch.std()
                    kernel = self._generate_kernel(patch.shape)
                    real_patch = self.run_interpolation(patch, real_patch, kernel)

                real_patch -= real_patch.mean()
                real_patch /= real_patch.std()

                data[key][c] = torch.from_numpy(real_patch)
        data["kernel"] = kernel
        return data

    def _generate_kernel(self, img_shape):
        """Generate a kernel for interpolation."""
        img_shape = img_shape
        scale = [sample_scalar(self.scale, img_shape, i) for i in range(len(img_shape))]
        loc = [sample_scalar(self.loc, img_shape, i) for i in range(len(img_shape))]
        loc = np.array(loc) * np.array(img_shape)
        coords = [
            np.linspace(-loc[i], img_shape[i] - loc[i], img_shape[i])
            for i in range(len(img_shape))
        ]
        meshgrid = np.meshgrid(*coords, indexing="ij")
        kernel = np.exp(
            -0.5 * sum([(meshgrid[i] / scale[i]) ** 2 for i in range(len(img_shape))])
        )
        kernel = np.expand_dims(kernel, 0)
        return kernel

    def run_interpolation(self, img, img_modified, kernel):
        """Interpolate between image and modified image, weighted with kernel."""
        return img + kernel * (img_modified - img)
