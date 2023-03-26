import os
from time import time

import imageio
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
    ToTensord,
)

from membrain_seg.dataloading.data_utils import read_nifti
from membrain_seg.dataloading.transforms import (
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

data_aug_params = {}
data_aug_params["rotation_x"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_y"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_z"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)


def get_training_transforms(prob_to_one=False, return_as_list=False):
    """Returns the data augmentation transforms for training phase."""
    aug_sequence = [
        RandRotated(
            keys=("image", "label"),
            range_x=data_aug_params["rotation_x"],
            range_y=data_aug_params["rotation_y"],
            range_z=data_aug_params["rotation_x"],
            prob=(1.0 if prob_to_one else 0.5),
            mode=("bilinear", "nearest"),
        ),
        RandZoomd(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.3),
            min_zoom=0.7,
            max_zoom=1.43,
            mode=("trilinear", "nearest"),
            padding_mode=("constant", "constant"),
        ),
        RandRotate90d(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.5),
            max_k=3,
            spatial_axes=(0, 1),
        ),
        RandRotate90d(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.5),
            max_k=3,
            spatial_axes=(0, 2),
        ),
        RandRotate90d(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.5),
            max_k=3,
            spatial_axes=(1, 2),
        ),
        AxesShuffle,
        OneOf(
            [
                RandApplyTransform(
                    transform=MedianFilterd(keys=["image"], radius=(2, 6)),
                    prob=(
                        1.0 if prob_to_one else 0.15
                    ),  # Changed range from 8 to 6 and prob to 15%
                    # for efficiency reasons
                ),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.3, 1.5),
                    sigma_y=(0.3, 1.5),
                    sigma_z=(0.3, 1.5),
                    prob=(1.0 if prob_to_one else 0.2),
                ),
            ]
        ),
        RandGaussianNoised(
            keys=["image"], prob=(1.0 if prob_to_one else 0.1), mean=0.0, std=0.5
        ),  # Chaned std from 0.1 to 0.5 --> check visually
        RandomBrightnessTransformd(
            keys=["image"], mu=0.0, sigma=0.5, prob=(1.0 if prob_to_one else 0.1)
        ),
        OneOf(
            [
                RandomContrastTransformd(
                    keys=["image"],
                    contrast_range=(0.5, 2.0),
                    preserve_range=True,
                    prob=(1.0 if prob_to_one else 0.1),
                ),
                RandomContrastTransformd(
                    keys=["image"],
                    contrast_range=(0.5, 2.0),
                    preserve_range=False,
                    prob=(1.0 if prob_to_one else 0.1),
                ),
            ]
        ),
        RandApplyTransform(
            SimulateLowResolutionTransform(
                keys=["image"],
                downscale_factor_range=(0.25, 1.0),
                upscale_mode="trilinear",
                downscale_mode="nearest",
            ),
            prob=(1.0 if prob_to_one else 0.15),
        ),
        RandApplyTransform(
            Compose(
                [
                    RandAdjustContrastWithInversionAndStats(keys=["image"], prob=1.0),
                    RandAdjustContrastWithInversionAndStats(keys=["image"], prob=1.0),
                ]
            ),
            prob=(1.0 if prob_to_one else 0.1),
        ),
        RandAxisFlipd(keys=("image", "label"), prob=(1.0 if prob_to_one else 0.5)),
        BlankCuboidTransform(
            keys=["image"],
            prob=(1.0 if prob_to_one else 0.2),
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
            prob=(1.0 if prob_to_one else 0.15),
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
            prob=(1.0 if prob_to_one else 0.15),
        ),
        SharpeningTransformMONAI(
            keys=["image"],
            strength=(0.1, 1),
            same_for_each_channel=False,
            p_per_channel=(1.0 if prob_to_one else 0.1),
        ),
        DownsampleSegForDeepSupervisionTransform(
            keys=["label"], ds_scales=deep_supervision_scales, order="nearest"
        ),
        ToTensord(keys=["image"], dtype=torch.float),
    ]
    if return_as_list:
        return aug_sequence
    return Compose(aug_sequence)


def get_validation_transforms(return_as_list=False):
    """Returns the data augmentation transforms for training phase."""
    aug_sequence = [
        DownsampleSegForDeepSupervisionTransform(
            keys=["label"], ds_scales=deep_supervision_scales, order="nearest"
        ),
        ToTensord(keys=["image"], dtype=torch.float),
    ]
    if return_as_list:
        return aug_sequence
    return Compose(aug_sequence)


def store_test_images(out_dir, img, label, aug_sequence):
    """
    Stores sample augmented images.

    Phase 1: For each transform, 5 isolated applications are stored.
    Phase 2: 50 examples of full applications of all transforms are stored.
    """
    orig_dict = {"image": img, "label": label}

    def save_images(prefix, out_img, out_label, slice_idx=80):
        imageio.imwrite(
            os.path.join(out_dir, f"{prefix}_dens.png"), out_img[0, :, :, slice_idx]
        )
        if not isinstance(out_label, list):
            imageio.imwrite(
                os.path.join(out_dir, f"{prefix}_label.png"),
                out_label[0, :, :, slice_idx],
            )
        else:
            for i, lbl in enumerate(out_label):
                imageio.imwrite(
                    os.path.join(out_dir, f"{prefix}_label_ds{i+1}.png"),
                    lbl[0, :, :, slice_idx // (2**i)],
                )

    for aug_nr, aug in enumerate(aug_sequence):
        print(aug_nr, type(aug))
        for sub_aug_nr in range(5):
            cur_dict = {
                "image": orig_dict["image"].copy(),
                "label": orig_dict["label"].copy(),
            }
            out_dict = aug(cur_dict)
            out_img = out_dict["image"]
            out_label = out_dict["label"]
            prefix = f"trafo{aug_nr}_{sub_aug_nr}_isolated"
            save_images(prefix, out_img, out_label)

    for combined_nr in range(50):
        cur_dict = {
            "image": orig_dict["image"].copy(),
            "label": orig_dict["label"].copy(),
        }
        for aug in aug_sequence:
            cur_dict = aug(cur_dict)
        out_img = cur_dict["image"]
        out_label = cur_dict["label"]
        prefix = f"combined_{combined_nr}"
        save_images(prefix, out_img, out_label)


def get_augmentation_timing(imgs, labels, aug_sequence):
    """Track the timing of all augmentation methods."""
    start_time = time()
    aug_time_dict = {i: [] for i in range(len(aug_sequence))}

    for i in range(1000):
        for img, label in zip(imgs, labels):
            cur_dict = {"image": img, "label": label}

            for aug_nr, aug in enumerate(aug_sequence):
                aug_start_time = time()
                cur_dict = aug(cur_dict)
                aug_time_dict[aug_nr].append(time() - aug_start_time)

        print(f"{i} / 1000")
        print(f"Current average time: {(time() - start_time) / (i + 1):.2f}")

        if i % 20 == 0:
            for k in range(len(aug_sequence)):
                avg_time = np.mean(aug_time_dict[k])
                print(f"Augmentation {k} avg time: {avg_time:.2f}")


if __name__ == "__main__":
    out_dir = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-seg/\
                membrain-seg/sanity_imgs"
    patch_path1 = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/nnUNet_training/\
                training_dirs/nnUNet_raw_data_base/nnUNet_raw_data/Task527_ChlamyV1/\
                imagesTr/tomo02_patch000_0000.nii.gz"
    label_path1 = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/nnUNet_training/\
                training_dirs/nnUNet_raw_data_base/nnUNet_raw_data/Task527_ChlamyV1/\
                labelsTr/tomo02_patch000.nii.gz"
    img = np.expand_dims(read_nifti(patch_path1), 0)
    label = np.expand_dims(read_nifti(label_path1), 0)

    store_test_images(
        out_dir,
        img,
        label,
        get_training_transforms(prob_to_one=True, return_as_list=True),
    )
    get_augmentation_timing(
        imgs=[img, img],
        labels=[label, label],
        aug_sequence=get_training_transforms(prob_to_one=False, return_as_list=True),
    )
