from typing import Callable, List, Union

import numpy as np
import torch
from monai.transforms import (
    Compose,
    OneOf,
    RandAxisFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    ToTensor,
    ToTensord,
)

from membrain_seg.segmentation.dataloading.transforms import (
    AxesShuffle,
    BlankCuboidTransform,
    BrightnessGradientAdditiveTransform,
    DownsampleSegForDeepSupervisionTransform,
    LocalGammaTransform,
    MedianFilterd,
    RandAdjustContrastWithInversionAndStats,
    RandApplyTransform,
    RandomBrightnessTransformd,
    RandomContrastTransformd,
    SharpeningTransformMONAI,
    SimulateLowResolutionTransform,
)

### Hard-coded area

# Hard-coding kernel sizes for pooling operations (work also with smaller network sizes)
pool_op_kernel_sizes = [
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
]  # hard-coded
net_num_pool_op_kernel_sizes = pool_op_kernel_sizes
deep_supervision_scales = [[1, 1, 1]] + [
    list(i) for i in 1 / np.cumprod(np.vstack(net_num_pool_op_kernel_sizes), axis=0)
][:-1]

# Define 3D rotation ranges
data_aug_params = {}
data_aug_params["rotation_x"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_y"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_z"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)

# Which axes should be used for mirroring?
mirror_axes = (0, 1, 2)


def get_mirrored_img(img: torch.Tensor, mirror_idx: int) -> torch.Tensor:
    """
    Get mirrored images for test time augmentation.

    There are 8 possible cases, enumerated from 0 to 7.
    The function supports mirroring across three axes,
    and combinations thereof.

    Parameters
    ----------
    img : torch.Tensor
        Input tensor to be mirrored.
    mirror_idx : int
        Integer index to select mirror case.
        Should be within the range [0, 7] inclusive.

    Returns
    -------
    torch.Tensor
        The mirrored image tensor.

    Raises
    ------
    AssertionError
        If the mirror index is not in the range [0, 7].

    """
    assert mirror_idx < 8 and mirror_idx >= 0
    if mirror_idx == 0:
        return img

    if mirror_idx == 1 and (2 in mirror_axes):
        return torch.flip(img, (4,))

    if mirror_idx == 2 and (1 in mirror_axes):
        return torch.flip(img, (3,))

    if mirror_idx == 3 and (2 in mirror_axes) and (1 in mirror_axes):
        return torch.flip(img, (4, 3))

    if mirror_idx == 4 and (0 in mirror_axes):
        return torch.flip(img, (2,))

    if mirror_idx == 5 and (0 in mirror_axes) and (2 in mirror_axes):
        return torch.flip(img, (4, 2))

    if mirror_idx == 6 and (0 in mirror_axes) and (1 in mirror_axes):
        return torch.flip(img, (3, 2))

    if (
        mirror_idx == 7
        and (0 in mirror_axes)
        and (1 in mirror_axes)
        and (2 in mirror_axes)
    ):
        return torch.flip(img, (4, 3, 2))


def get_training_transforms(
    prob_to_one: bool = False, return_as_list: bool = False
) -> Union[List[Callable], Compose]:
    """
    Returns the data augmentation transforms for training phase.

    The function sets up an augmentation sequence containing a variety of
    transformations such as rotations, zooms, 90 degree rotations,
    Gaussian noise, brightness adjustments, and more.
    If desired, the sequence can be returned as a list.

    Parameters
    ----------
    prob_to_one : bool, optional
        If True, the probability of applying the transformation is set to 1.0 for
            all transformations in the sequence.
        If False, the probability is lower (specified within each transformation).
        Default is False.
    return_as_list : bool, optional
        If True, the sequence of transformations is returned as a list.
        If False, the sequence is returned as a Compose object. Default is False.

    Returns
    -------
    List[Callable] or Compose
        If return_as_list is True, the function returns a list of
            transformation functions.
        If return_as_list is False, the function returns a Compose object
            containing the sequence of transformations.

    """
    aug_sequence = [
        RandRotated(
            keys=("image", "label"),
            range_x=data_aug_params["rotation_x"],
            range_y=data_aug_params["rotation_y"],
            range_z=data_aug_params["rotation_x"],
            prob=(1.0 if prob_to_one else 0.75),
            mode=("bilinear", "nearest"),
        ),
        RandZoomd(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.3),
            min_zoom=0.7,
            max_zoom=1.43,
            mode=("trilinear", "nearest-exact"),
            padding_mode=("constant", "constant"),
        ),  # TODO: Independent scale for each axis?
        RandRotate90d(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.70),
            max_k=3,
            spatial_axes=(0, 1),
        ),
        RandRotate90d(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.70),
            max_k=3,
            spatial_axes=(0, 2),
        ),
        RandRotate90d(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.70),
            max_k=3,
            spatial_axes=(1, 2),
        ),
        AxesShuffle,
        OneOf(
            [
                RandApplyTransform(
                    transform=MedianFilterd(keys=["image"], radius=(2, 8)),
                    prob=(
                        1.0 if prob_to_one else 0.25
                    ),  # Changed range from 8 to 6 and prob to 15%
                    # for efficiency reasons
                ),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.3, 1.5),
                    sigma_y=(0.3, 1.5),
                    sigma_z=(0.3, 1.5),
                    prob=(1.0 if prob_to_one else 0.3),
                ),
            ]
        ),
        RandGaussianNoised(
            keys=["image"], prob=(1.0 if prob_to_one else 0.4), mean=0.0, std=0.7
        ),  # Chaned std from 0.1 to 0.5 --> check visually
        RandomBrightnessTransformd(
            keys=["image"], mu=0.0, sigma=0.5, prob=(1.0 if prob_to_one else 0.30)
        ),
        OneOf(
            [
                RandomContrastTransformd(
                    keys=["image"],
                    contrast_range=(0.5, 2.0),
                    preserve_range=True,
                    prob=(1.0 if prob_to_one else 0.30),
                ),
                RandomContrastTransformd(
                    keys=["image"],
                    contrast_range=(0.5, 2.0),
                    preserve_range=False,
                    prob=(1.0 if prob_to_one else 0.30),
                ),
            ]
        ),
        RandApplyTransform(
            SimulateLowResolutionTransform(
                keys=["image"],
                downscale_factor_range=(0.25, 1.0),
                upscale_mode="trilinear",
                downscale_mode="nearest-exact",
            ),
            prob=(1.0 if prob_to_one else 0.35),
        ),
        RandApplyTransform(
            Compose(
                [
                    RandAdjustContrastWithInversionAndStats(keys=["image"], prob=1.0),
                    RandAdjustContrastWithInversionAndStats(keys=["image"], prob=1.0),
                ]
            ),
            prob=(1.0 if prob_to_one else 0.25),
        ),
        RandAxisFlipd(keys=("image", "label"), prob=(0.5)),
        BlankCuboidTransform(
            keys=["image"],
            prob=(1.0 if prob_to_one else 0.4),
            cuboid_area=(160 // 10, 160 // 3),
            is_3d=True,
            max_cuboids=5,
            replace_with="mean",
        ),  # patch size of 160 hard-coded. Should we make it flexible?
        RandApplyTransform(
            BrightnessGradientAdditiveTransform(
                keys=["image"],
                scale=lambda x, y: np.exp(
                    np.random.uniform(np.log(x[y] // 6), np.log(x[y]))
                ),
                loc=(-0.5, 1.5),
                max_strength=lambda x, y: np.random.uniform(-5, -1)
                if np.random.uniform() < 0.5
                else np.random.uniform(1, 5),
                mean_centered=False,
            ),
            prob=(1.0 if prob_to_one else 0.3),
        ),
        RandApplyTransform(
            LocalGammaTransform(
                keys=["image"],
                scale=lambda x, y: np.exp(
                    np.random.uniform(np.log(x[y] // 6), np.log(x[y]))
                ),
                loc=(-0.5, 1.5),
                gamma=lambda: np.random.uniform(0.01, 0.8)
                if np.random.uniform() < 0.5
                else np.random.uniform(1.5, 4),
            ),
            prob=(1.0 if prob_to_one else 0.3),
        ),
        SharpeningTransformMONAI(
            keys=["image"],
            strength=(0.1, 1),
            same_for_each_channel=False,
            p_per_channel=(1.0 if prob_to_one else 0.2),
        ),
        DownsampleSegForDeepSupervisionTransform(
            keys=["label"], ds_scales=deep_supervision_scales, order="nearest"
        ),
        ToTensord(keys=["image"], dtype=torch.float),
    ]
    if return_as_list:
        return aug_sequence
    return Compose(aug_sequence)


def get_validation_transforms(
    return_as_list: bool = False,
) -> Union[List[Callable], Compose]:
    """
    Returns the data augmentation transforms for the validation phase.

    The function sets up a sequence of transformations including downsampling
    and tensor conversion. If desired, the sequence can be returned as a list.

    Parameters
    ----------
    return_as_list : bool, optional
        If True, the sequence of transformations is returned as a list.
        If False, the sequence is returned as a Compose object. Default is False.

    Returns
    -------
    List[Callable] or Compose
        If return_as_list is True, the function returns a list of
            transformation functions.
        If return_as_list is False, the function returns a Compose object
            containing the sequence of transformations.

    """
    aug_sequence = [
        DownsampleSegForDeepSupervisionTransform(
            keys=["label"], ds_scales=deep_supervision_scales, order="nearest"
        ),
        ToTensord(keys=["image"], dtype=torch.float),
    ]
    if return_as_list:
        return aug_sequence
    return Compose(aug_sequence)


def get_prediction_transforms() -> Compose:
    """
    Returns the data augmentation transforms for the prediction phase.

    The function sets up a Compose object containing a transformation for
    converting data to tensors.

    Returns
    -------
    Compose
        A Compose object containing the sequence of transformations for
        the prediction phase.

    """
    transforms = Compose(
        [
            ToTensor(),
        ]
    )
    return transforms
