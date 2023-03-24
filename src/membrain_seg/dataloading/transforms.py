from typing import Optional, Tuple, Union

import numpy as np
import torch
from monai.transforms import (
    Compose,
    MapTransform,
    OneOf,
    RandAdjustContrastd,
    Randomizable,
    Resize,
    Transform,
    Transposed,
    Zoomd,
)
from scipy.ndimage import convolve, median_filter


class RandApplyTransform(Randomizable, Transform):
    """Randomly apply given MONAI transform with given probability."""

    def __init__(
        self,
        transform: Transform,
        prob: float = 0.5,
        do_transform: Optional[bool] = None,
    ):
        if prob < 0.0 or prob > 1.0:
            raise ValueError("Probability must be a value between 0 and 1.")
        self.transform = transform
        self.prob = prob
        self.do_transform = do_transform

    def randomize(self, data: None) -> None:
        """Roll a dice and decide whether we actuall do the augmentation."""
        if self.do_transform is None:
            self._do_transform = self.R.random() < self.prob
        else:
            self._do_transform = self.do_transform

    def __call__(self, data):
        """Apply transform only with given probability."""
        self.randomize(None)
        if self._do_transform:
            return self.transform(data)
        return data


class MedianFilterd(Transform):
    """Median filter from batchgenerators package reimplemented."""

    def __init__(self, keys, radius=1):
        self.keys = keys
        self.radius_range = radius

    def randomize(self, data: None) -> None:
        """Randomly draw median filter range."""
        if isinstance(self.radius_range, int):
            self.radius = self.radius_range
        elif isinstance(self.radius_range, tuple):
            self.radius = np.random.randint(self.radius_range[0], self.radius_range[1])

    def __call__(self, data):
        """Apply median filter."""
        self.randomize(None)
        for key in self.keys:
            for c in range(data[key].shape[0]):
                data[key][c] = torch.from_numpy(
                    median_filter(data[key][c], size=self.radius, mode="reflect")
                )  # TODO: Is this very inefficient?
        return data


class RandomBrightnessTransformd(Transform):
    """Brightness transform from batchgenerators reimplemented."""

    def __init__(self, keys, mu, sigma, prob=1.0):
        super().__init__()
        self.keys = keys
        self.mu = mu
        self.sigma = sigma
        self.prob = prob

    def __call__(self, data):
        """Apply brightness transform."""
        if np.random.rand() < self.prob:
            for key in self.keys:
                add_const = np.random.normal(loc=self.mu, scale=self.sigma)
                data[key] = data[key] + add_const
        return data


class RandomContrastTransformd(Transform):
    """Randomly scale the data with a contrast factor."""

    def __init__(self, keys, contrast_range, preserve_range=False, prob=1.0):
        super().__init__()
        self.keys = keys
        self.contrast_range = contrast_range
        self.prob = prob
        self.preserve_range = preserve_range

    def __call__(self, data):
        """Apply the transform."""
        if np.random.rand() < self.prob:
            for key in self.keys:
                # Choose a random contrast factor from the given range
                contrast_factor = np.random.uniform(
                    self.contrast_range[0], self.contrast_range[1]
                )

                # Apply the contrast transformation
                mean = np.mean(data[key])
                if self.preserve_range:
                    minval = data[key].min()
                    maxval = data[key].max()
                data[key] = mean + contrast_factor * (data[key] - mean)
                if self.preserve_range:
                    data[key][data[key] < minval] = minval
                    data[key][data[key] > maxval] = maxval

        return data


class RandAdjustContrastWithInversionAndStats(RandAdjustContrastd):
    """
    Adjusted version of RandAdjustContrastd from MONAI.

    After the normal function call, the original mean and std are preserved
    and the result is multiplied by -1. It is therefore recommended to
    apply the function 2 times.
    (Without two time application and negation, the transform is non-symmetric)
    """

    def __call__(self, data):
        """Apply the transform."""
        for key in self.keys:
            if self._do_transform:
                img = data[key]
                mean_before = img.mean()
                std_before = img.std()

                img = super().__call__({key: img})[key]

                mean_after = img.mean()
                std_after = img.std()

                img = (img - mean_after) / std_after * std_before + mean_before
                img *= -1
                data[key] = img

        return data


class SimulateLowResolutionTransform(Randomizable, Transform):
    """
    Batchgenerators' transform for simulating low resolution upscaling.

    The image is downsampled using nearest neighbor interpolation using a
    random scale factor. Afterwards, it is upsampled again with trilinear
    interpolation, imitating the upscaling of low-resolution images.
    """

    def __init__(
        self,
        keys,
        downscale_factor_range,
        upscale_mode="trilinear",
        downscale_mode="nearest",
    ):
        self.keys = keys
        self.downscale_factor_range = downscale_factor_range
        self.downscale_factor = None
        self.upscale_mode = upscale_mode
        self.downscale_mode = downscale_mode

    def randomize(self, data):
        """Randomly choose the downsample factor."""
        self.downscale_factor = self.R.uniform(*self.downscale_factor_range)

    def __call__(self, data):
        """Apply the transform."""
        self.randomize(data)
        downscale_transform = Zoomd(
            self.keys, zoom=self.downscale_factor, mode=self.downscale_mode
        )
        upscale_transform = Zoomd(
            self.keys, zoom=1.0 / self.downscale_factor, mode=self.upscale_mode
        )

        data = downscale_transform(data)
        data = upscale_transform(data)
        return data


class BlankCuboidTransform(Randomizable, MapTransform):
    """
    Randomly blank out cuboids from the image.

    The number and sizes of cuboids are randomly drawn.
    """

    def __init__(
        self,
        keys,
        prob=0.5,
        cuboid_area=(10, 100),
        is_3d=False,
        max_cuboids=1,
        replace_with="mean",
    ):
        super().__init__(keys)
        self.prob = prob
        self.cuboid_area = cuboid_area
        self.is_3d = is_3d
        self.max_cuboids = max_cuboids
        self.replace_with = replace_with

    def randomize(self, data=None):
        """Randomly choose the number of cuboids & their sizes."""
        self.do_transform = self.R.random() < self.prob
        if self.do_transform:
            self.num_cuboids = self.R.randint(1, self.max_cuboids + 1)
            self.cuboids = []
            for _ in range(self.num_cuboids):
                height = self.R.randint(self.cuboid_area[0], self.cuboid_area[1] + 1)
                width = self.R.randint(self.cuboid_area[0], self.cuboid_area[1] + 1)
                depth = (
                    self.R.randint(self.cuboid_area[0], self.cuboid_area[1] + 1)
                    if self.is_3d
                    else 1
                )
                self.cuboids.append((height, width, depth))

    def __call__(self, data):
        """Apply the transform."""
        self.randomize()
        if not self.do_transform:
            return data

        d = dict(data)
        for key in self.keys:
            image = d[key]
            for height, width, depth in self.cuboids:
                if self.is_3d:
                    z_max, y_max, x_max = (
                        image.shape[-3],
                        image.shape[-2],
                        image.shape[-1],
                    )
                    z = self.R.randint(0, z_max - depth)
                else:
                    y_max, x_max = image.shape[-2], image.shape[-1]
                    z, depth = 0, 1
                y = self.R.randint(0, y_max - height)
                x = self.R.randint(0, x_max - width)
                if self.replace_with == "mean":
                    image[
                        ..., z : z + depth, y : y + height, x : x + width
                    ] = torch.mean(torch.Tensor(image))
                elif self.replace_with == 0.0:
                    image[..., z : z + depth, y : y + height, x : x + width] = 0.0
            d[key] = image

        return d


def sample_scalar(value, *args):
    """Sample scalar function from batchgenerators."""
    if isinstance(value, (tuple, list)):
        return np.random.uniform(value[0], value[1])
    elif callable(value):
        # return value(image, kernel)
        return value(*args)
    else:
        return value


class BrightnessGradientAdditiveTransform(Randomizable, MapTransform):
    """
    Add a brightness gradient to the image (also from batchgenerators).

    A scaled Gaussian kernel is added to the image. The center of the
    Gaussian kernel is randomly drawn an can also be outside of the image.
    """

    def __init__(
        self,
        keys,
        scale,
        loc=(-1, 2),
        max_strength=1.0,
        mean_centered=True,
        clip_intensities=False,
    ):
        super().__init__(keys)
        self.scale = scale
        self.loc = loc
        self.max_strength = max_strength
        self.mean_centered = mean_centered

    def _generate_kernel(self, img_shape):
        img_shape = img_shape[2:]
        scale = [sample_scalar(self.scale, img_shape, i) for i in range(len(img_shape))]
        loc = [sample_scalar(self.loc, img_shape, i) for i in range(len(img_shape))]
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

    def __call__(self, data):
        """Apply the transform."""
        d = dict(data)
        for key in self.keys:
            image = d[key]
            img_shape = image.shape
            kernel = self._generate_kernel(img_shape)
            if self.mean_centered:
                kernel -= kernel.mean()
            max_kernel_val = max(np.max(np.abs(kernel)), 1e-8)
            strength = sample_scalar(self.max_strength, image, kernel)
            kernel = kernel / max_kernel_val * strength
            image += kernel

            d[key] = image

        return d


class LocalGammaTransform(Randomizable, MapTransform):
    """Locally adjusts the Gamma value using Gaussian kernel.(from batchgenerators)."""

    def __init__(self, keys, scale, loc=(-1, 2), gamma=(0.5, 1)):
        super().__init__(keys)
        self.scale = scale
        self.loc = loc
        self.gamma = gamma

    def _generate_kernel(self, img_shape):
        img_shape = img_shape[2:]
        scale = [sample_scalar(self.scale, img_shape, i) for i in range(len(img_shape))]
        loc = [sample_scalar(self.loc, img_shape, i) for i in range(len(img_shape))]
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

    def _apply_gamma_gradient(self, img, kernel):
        mn, mx = img.min(), img.max()
        img = (img - mn) / (max(mx - mn, 1e-8))

        gamma = sample_scalar(self.gamma)
        img_modified = np.power(img, gamma)

        return self.run_interpolation(img, img_modified, kernel) * (mx - mn) + mn

    def run_interpolation(self, img, img_modified, kernel):
        """Interpolate between image and modified image, weighted with kernel."""
        return img + kernel * (img_modified - img)

    def __call__(self, data):
        """Apply the transform."""
        d = dict(data)
        for key in self.keys:
            image = d[key]
            img_shape = image.shape
            kernel = self._generate_kernel(img_shape)
            d[key] = self._apply_gamma_gradient(image, kernel)
        return d


class SharpeningTransformMONAI(MapTransform):
    """Laplacian Sharpening transform. (from batchgenerators)."""

    filter_2d = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filter_3d = np.array(
        [
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
        ]
    )

    def __init__(
        self,
        strength: Union[float, Tuple[float, float]] = 0.2,
        same_for_each_channel: bool = False,
        p_per_channel: float = 0.5,
        keys: str = "image",
    ):
        super().__init__(keys)
        self.strength = strength
        self.same_for_each_channel = same_for_each_channel
        self.p_per_channel = p_per_channel

    def __call__(self, data):
        """Apply the transform."""
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)
            if self.same_for_each_channel:
                strength_here = (
                    self.strength
                    if isinstance(self.strength, float)
                    else np.random.uniform(*self.strength)
                )
                filter_here = (
                    self.filter_2d if img.ndim == 3 else self.filter_3d
                ) * strength_here
                filter_here[tuple([1] * (img.ndim - 1))] += 1
                for c in range(img.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        img[c] = torch.from_numpy(
                            convolve(img[c], filter_here, mode="reflect")
                        )
                        img[c] = np.clip(img[c], img[c].min(), img[c].max())
            else:
                for c in range(img.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        strength_here = (
                            self.strength
                            if isinstance(self.strength, float)
                            else np.random.uniform(*self.strength)
                        )
                        filter_here = (
                            self.filter_2d if img.ndim == 3 else self.filter_3d
                        ) * strength_here
                        filter_here[tuple([1] * (img.ndim - 1))] += 1
                        img[c] = torch.from_numpy(
                            convolve(img[c], filter_here, mode="reflect")
                        )
                        img[c] = torch.from_numpy(
                            np.clip(img[c], img[c].min(), img[c].max())
                        )
            d[key] = img.squeeze() if len(img.shape) == 5 and img.shape[0] == 1 else img
        return d


class DownsampleSegForDeepSupervisionTransform(MapTransform):
    """Downsample labels for deep supervision (from nnUNet)."""

    def __init__(
        self,
        keys: Union[str, Tuple[str]],
        ds_scales: Tuple[float, float, float] = (1, 0.5, 0.25),
        order: int = 0,
        axes: Union[None, Tuple[int, int, int]] = None,
    ):
        super().__init__(keys)
        self.axes = axes
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, data_dict):
        """Apply the transform."""
        for key in self.keys:
            data_dict[key] = downsample_seg_for_ds_transform(
                data_dict[key], self.ds_scales, self.order, self.axes
            )
        return data_dict


def downsample_seg_for_ds_transform(
    seg: np.ndarray,
    ds_scales: Tuple[float, float, float] = (1, 0.5, 0.25),
    order: str = "nearest",
    axes: Union[None, Tuple[int, int, int]] = None,
):
    """Downsampling of segmentations."""
    if axes is None:
        axes = list(range(1, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            resize_transform = Resize(
                new_shape[1:],
                mode=order,
                align_corners=(None if order == "nearest" else False),
            )
            out_seg = np.zeros(new_shape, dtype=np.array(seg).dtype)
            for c in range(seg.shape[0]):
                out_seg[c] = resize_transform(np.expand_dims(seg[c], 0)).squeeze()
            output.append(out_seg)
    return output


AxesSwap = OneOf(
    [
        Transposed(keys=("image", "label"), indices=(0, 2, 1, 3)),
        Transposed(keys=("image", "label"), indices=(0, 1, 3, 2)),
        Transposed(keys=("image", "label"), indices=(0, 3, 2, 1)),
    ]
)
AxesShuffle = Compose(
    [
        RandApplyTransform(transform=AxesSwap, prob=0.5),
        RandApplyTransform(transform=AxesSwap, prob=0.5),
        RandApplyTransform(transform=AxesSwap, prob=0.5),
    ]
)
