from time import time
from typing import Optional

import numpy as np
from monai.transforms import (
    Compose,
    OneOf,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Randomizable,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    Transform,
    Transposed,
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

    def _randomize(self, data: None) -> None:
        if self.do_transform is None:
            self._do_transform = self.R.random() < self.prob
        else:
            self._do_transform = self.do_transform

    def __call__(self, data):
        """Apply transform only with given probability."""
        self._randomize(None)
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
