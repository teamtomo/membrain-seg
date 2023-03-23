from time import time
from typing import Optional

import numpy as np
from monai.transforms import (
    Compose,
    MapTransform,
    OneOf,
    RandAdjustContrastd,
    RandAxisFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Randomizable,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    Transform,
    Transposed,
    Zoomd,
)
from scipy.ndimage import median_filter


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
        if isinstance(radius, int):
            self.radius = radius
        elif isinstance(radius, tuple):
            self.radius = np.random.randint(radius[0], radius[1])

    def __call__(self, data):
        """Apply median filter."""
        for key in self.keys:
            data[key] = median_filter(data[key], size=self.radius, mode="reflect")
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
            if self._do_transform():
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
            self.keys, zoom=1.0 / self.downscale_factor, mode=self.downscale_mode
        )
        upscale_transform = Zoomd(
            self.keys, zoom=self.downscale_factor, mode=self.upscale_mode
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
                    image[..., z : z + depth, y : y + height, x : x + width] = np.mean(
                        image
                    )
                elif self.replace_with == 0.0:
                    image[..., z : z + depth, y : y + height, x : x + width] = 0.0
            d[key] = image

        return d


def sample_scalar(value, image=None, kernel=None):
    """Sample scalar function from batchgenerators."""
    if isinstance(value, (tuple, list)):
        return np.random.uniform(value[0], value[1])
    elif callable(value):
        return value(image, kernel)
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
        p_per_sample=1.0,
        clip_intensities=False,
    ):
        super().__init__(keys)
        self.scale = scale
        self.loc = loc
        self.max_strength = max_strength
        self.p_per_sample = p_per_sample
        self.mean_centered = mean_centered

    def _generate_kernel(self, img_shape):
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
        return kernel

    def __call__(self, data):
        """Apply the transform."""
        d = dict(data)
        for key in self.keys:
            image = d[key]
            img_shape = image.shape
            if np.random.uniform() < self.p_per_sample:
                kernel = self._generate_kernel(img_shape)
                if self.mean_centered:
                    kernel -= kernel.mean()
                max_kernel_val = max(np.max(np.abs(kernel)), 1e-8)
                strength = sample_scalar(self.max_strength, image, kernel)
                kernel = kernel / max_kernel_val * strength
                image += kernel

                d[key] = image

        return d


data_aug_params = {}
data_aug_params["rotation_x"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_y"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_z"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)

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

aug_sequence = [
    RandRotated(
        keys=("image", "label"),
        range_x=data_aug_params["rotation_x"],
        range_y=data_aug_params["rotation_y"],
        range_z=data_aug_params["rotation_x"],
        prob=0.5,
        mode=("bilinear", "nearest"),
    ),
    RandZoomd(
        keys=("image", "label"),
        prob=0.3,
        min_zoom=0.7,
        max_zoom=1.4,
        mode=("trilinear", "nearest"),
    ),
    RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3, spatial_axes=(0, 1)),
    RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3, spatial_axes=(0, 2)),
    RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3, spatial_axes=(1, 2)),
    AxesShuffle,
    OneOf(
        [
            RandApplyTransform(
                transform=MedianFilterd(keys=("image"), radius=(2, 8)), prob=0.2
            ),
            RandGaussianSmoothd(
                keys=("image"),
                sigma_x=(0.3, 1.5),
                sigma_y=(0.3, 1.5),
                sigma_z=(0.3, 1.5),
                prob=0.2,
            ),
        ]
    ),
    RandGaussianNoised(keys=("image"), prob=0.1, mean=0.0, std=0.1),
    RandomBrightnessTransformd(keys=("image"), mu=0.0, sigma=0.5, prob=0.1),
    OneOf(
        [
            RandomContrastTransformd(
                keys=("image"), contrast_range=(0.5, 2.0), preserve_range=True, prob=0.1
            ),
            RandomContrastTransformd(
                keys=("image"),
                contrast_range=(0.5, 2.0),
                preserve_range=False,
                prob=0.1,
            ),
        ]
    ),
    SimulateLowResolutionTransform(
        keys=("image"),
        downscale_factor_range=(0.25, 1.0),
        upscale_mode="trilinear",
        downscale_mode="nearest",
    ),
    RandApplyTransform(
        Compose(
            [
                RandAdjustContrastWithInversionAndStats(keys=("image"), prob=1.0),
                RandAdjustContrastWithInversionAndStats(keys=("image"), prob=1.0),
            ]
        ),
        prob=0.1,
    ),
    RandAxisFlipd(keys=("image", "label"), prob=0.5),
    BlankCuboidTransform(
        keys=("image"),
        prob=0.2,
        cuboid_area=(160 // 10, 160 // 3),
        is_3d=True,
        max_cuboids=5,
        replace_with="mean",
    ),  # patch size of 160 hard-coded. Should we make it flexible?
    RandApplyTransform(
        BrightnessGradientAdditiveTransform(
            keys=("image"),
            scale=lambda x, y: np.exp(
                np.random.uniform(np.log(x[y] // 6), np.log(x[y]))
            ),
            loc=(-0.5, 1.5),
            max_strength=lambda x, y: np.random.uniform(-5, -1)
            if np.random.uniform() < 0.5
            else np.random.uniform(1, 5),
            mean_centered=False,
        ),
        prob=0.15,
    ),
]


imgs, labels = np.random.randn(2, 1, 160, 160, 160), np.random.randint(
    0, 2, (2, 1, 160, 160, 160)
)

for img, label in zip(imgs, labels):
    cur_dict = {"image": img, "label": label}
    time_zero = time()
    for aug in aug_sequence:
        cur_dict = aug(cur_dict)
        print(cur_dict["image"].shape)
        print(cur_dict["label"].shape)
        print(type(aug))
        print("")
    print("This sample took", time() - time_zero, "seconds.")
