from membrain_seg.tomo_preprocessing.matching_utils.filter_utils import rotational_kernel
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch 
import numpy.fft as fft
from monai.transforms import (
    Randomizable,
    Transform,
)


def wedge_mask(shape, angle):
    # Create the 3D coordinate grid
    x_coords, _, z_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    x_center = np.mean(x_coords)
    z_center = np.mean(z_coords)
    
    angle_radians = np.radians(angle)
    m = np.tan(angle_radians)  # slope
    b = z_center # intercept

    def above_wedge(x):
        return m * x + b
    
    x_left_side = x_coords[:int(x_coords.shape[0] / 2), :, :]
    x_right_side = x_coords[int(x_coords.shape[0] / 2):, :, :]

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
    wedge_mask[int(x_coords.shape[0] / 2), int(x_coords.shape[2] / 2), int(x_coords.shape[2] / 2)] = 1
    return ~wedge_mask


def get_line_plot(n_points, smooth_sigma, step_sigma, offset):
    # Generate an array of x values from 0 to n_points
    x = np.linspace(0, n_points-1, n_points)

    # Generate an array of y values using a random walk
    y = np.cumsum(np.random.randn(n_points) * step_sigma)

    # Apply Gaussian smoothing
    y_smoothed = gaussian_filter1d(y, sigma=smooth_sigma)
    return x, np.abs(y_smoothed + offset)


def normalize_and_fft_patch(patch):
    patch -= patch.min()
    patch /= patch.max()
    t = fft.fftn(patch)
    t = fft.fftshift(t)
    return t


def fft_patch_to_real(fft_patch):
    fft_patch  = fft.fftshift(fft_patch) # Added this. Do I need it? What happens?
    t = fft.ifftn(fft_patch)
    t = np.real(t)
    # t = np.abs(t).astype("f4")
    return t


class MissingWedgeMaskAndFourierAmplitudeMatchingCombined(Randomizable, Transform):
    """
    still empty
    """

    def __init__(self, keys, amplitude_aug=True, missing_wedge_aug=True, smooth_sigma_range=(2., 4.), step_sigma_range=(0.1, 4.), offset_range=(2., 10.),
            missing_angle_range=(55.,88.), missing_wedge_prob=0.5, amplitude_prob=0.5):
        self.keys = keys

        self.amplitude_aug = amplitude_aug
        self.smooth_sigma_range = smooth_sigma_range
        self.step_sigma_range=step_sigma_range
        self.offset_range = offset_range

        self.missing_wedge_aug = missing_wedge_aug
        self.missing_angle_range = missing_angle_range

        self.amplitude_prob = amplitude_prob
        self.missing_wedge_prob = missing_wedge_prob

    def randomize(self, data: None) -> None:
        """Randomly draw median filter range."""
        if isinstance(self.smooth_sigma_range, float):
            self.smooth_sigma = self.smooth_sigma_range
        elif isinstance(self.smooth_sigma_range, tuple):
            self.smooth_sigma = np.random.uniform(self.smooth_sigma_range[0], self.smooth_sigma_range[1])

        if isinstance(self.step_sigma_range, float):
            self.step_sigma = self.step_sigma_range
        elif isinstance(self.step_sigma_range, tuple):
            self.step_sigma = np.random.uniform(self.step_sigma_range[0], self.step_sigma_range[1])

        if isinstance(self.offset_range, float):
            self.offset = self.offset_range
        elif isinstance(self.smooth_sigma_range, tuple):
            self.offset = np.random.uniform(self.offset_range[0], self.offset_range[1])

        if isinstance(self.missing_angle_range, float):
            self.missing_angle = self.missing_angle_range
        elif isinstance(self.smooth_sigma_range, tuple):
            self.missing_angle = np.random.uniform(self.missing_angle_range[0], self.missing_angle_range[1])
        
        self._do_mw_transform = self.R.random() < self.missing_wedge_prob
        self._do_amp_transform = self.R.random() < self.amplitude_prob
        

    def __call__(self, data):
        """Apply median filter."""
        self.randomize(None)
        if self.missing_wedge_aug + self.amplitude_aug == 0:
            return data
        for key in self.keys:
            for c in range(data[key].shape[0]):
                patch = data[key][c]
                fft_patch = normalize_and_fft_patch(patch)
                if self.amplitude_aug and self._do_amp_transform:
                    _, fake_spectrum = get_line_plot(n_points=int(patch.shape[0] / 2), \
                        smooth_sigma=self.smooth_sigma, step_sigma=self.step_sigma, offset=self.offset)
                    equal_kernel = rotational_kernel(fake_spectrum, patch.shape)
                    fft_patch *= equal_kernel
                if self.missing_wedge_aug and self._do_mw_transform:
                    missing_wedge_mask = wedge_mask(patch.shape, self.missing_angle)
                    fft_patch[~missing_wedge_mask] = 0.
                real_patch = fft_patch_to_real(fft_patch)
                real_patch -= real_patch.mean()
                real_patch /= real_patch.std()
                data[key][c] = torch.from_numpy(real_patch)
        return data