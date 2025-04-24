import logging
import os
from typing import Dict

import imageio as io
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from membrain_seg.segmentation.dataloading.data_utils import read_nifti
from membrain_seg.segmentation.dataloading.memseg_augmentation import (
    get_training_transforms,
    get_validation_transforms,
)


class CryoETMemSegDataset(Dataset):
    """
    A custom Dataset for Cryo-ET membrane segmentation patches.

    This Dataset loads image-label pairs from a specified directory abd applies
    appropriate transformations for training or validation on the fly.

    Attributes
    ----------
    img_folder : str
        The path to the directory containing the image files.
    label_folder : str
        The path to the directory containing the label files.
    train : bool, default False
        A flag indicating whether the dataset is used for training or not.
    aug_prob_to_one : bool, default False
        A flag indicating whether the probability of augmentation should be
        set to one or not.
    patch_size : int, default 160
        The size of the patches to be extracted from the images.


    Methods
    -------
    __getitem__(idx: int) -> Dict[str, np.ndarray]
        Returns a dictionary containing an image-label pair corresponding to
        the provided index.
    __len__() -> int
        Returns the number of image-label pairs in the dataset.
    load_data() -> None
        Loads image-label pairs into memory from the specified directories.
    initialize_imgs_paths() -> None
        Initializes the list of paths to image-label pairs.
    test(test_folder: str, num_files: int = 20) -> None
        Tests the data loading and augmentation process by generating
            a set of images and their labels. Test images are then stored
            for sanity checks.
    """

    def __init__(
        self,
        img_folder: str,
        label_folder: str,
        train: bool = False,
        aug_prob_to_one: bool = False,
        patch_size: int = 160,
        on_the_fly_loading: bool = False,
    ) -> None:
        """
        Constructs all the necessary attributes for the CryoETMemSegDataset object.

        Parameters
        ----------
        img_folder : str
            The path to the directory containing the image files.
        label_folder : str
            The path to the directory containing the label files.
        train : bool, default False
            A flag indicating whether the dataset is used for training or validation.
        aug_prob_to_one : bool, default False
            A flag indicating whether the probability of augmentation should be set
            to one or not.
        patch_size : int, default 160
            The size of the patches to be extracted from the images.
        on_the_fly_loading : bool, default False
            A flag indicating whether the data should be loaded on the fly or not.
        """
        self.train = train
        self.img_folder, self.label_folder = img_folder, label_folder
        self.patch_size = patch_size
        self.on_the_fly_loading = on_the_fly_loading
        self.initialize_imgs_paths()
        if not self.on_the_fly_loading:
            self.load_data()
        self.transforms = (
            get_training_transforms(prob_to_one=aug_prob_to_one)
            if self.train
            else get_validation_transforms()
        )

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing an image-label pair for the provided index.

        Data augmentations are applied before returning the dictionary.

        Parameters
        ----------
        idx : int
            Index of the sample to be fetched.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing an image and its corresponding label.
        """
        if self.on_the_fly_loading:
            idx_dict = self.load_data_sample(idx)
            idx_dict["image"] = np.expand_dims(idx_dict["image"], 0)
            idx_dict["label"] = np.expand_dims(idx_dict["label"], 0)
            ds_label = idx_dict["dataset"]
        else:
            idx_dict = {
                "image": np.expand_dims(self.imgs[idx], 0),
                "label": np.expand_dims(self.labels[idx], 0),
            }
            ds_label = self.dataset_labels[idx]
        idx_dict = self.get_random_crop(idx_dict)
        idx_dict = self.transforms(idx_dict)
        idx_dict["dataset"] = ds_label  # transforms remove the dataset token
        return idx_dict

    def __len__(self) -> int:
        """
        Returns the number of image-label pairs in the dataset.

        Returns
        -------
        int
            The number of image-label pairs in the dataset.
        """
        return len(self.data_paths)

    def get_random_crop(self, idx_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Returns a random crop from the image-label pair.

        Parameters
        ----------
        idx_dict : Dict[str, np.ndarray]
            A dictionary containing an image and its corresponding label.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing a random crop from the image and its corresponding
            label.
        """
        img, label = idx_dict["image"], idx_dict["label"]
        x, y, z = img.shape[1:]

        if x <= self.patch_size or y <= self.patch_size or z <= self.patch_size:
            # pad with 2s on both sides
            pad_x = max(self.patch_size - x, 0)
            pad_y = max(self.patch_size - y, 0)
            pad_z = max(self.patch_size - z, 0)
            img = np.pad(
                img,
                (
                    (0, 0),
                    (pad_x // 2, pad_x // 2),
                    (pad_y // 2, pad_y // 2),
                    (pad_z // 2, pad_z // 2),
                ),
                mode="constant",
                constant_values=0,
            )
            label = np.pad(
                label,
                (
                    (0, 0),
                    (pad_x // 2, pad_x // 2),
                    (pad_y // 2, pad_y // 2),
                    (pad_z // 2, pad_z // 2),
                ),
                mode="constant",
                constant_values=2,
            )
            # make sure there was no rounding issue
            if (
                img.shape[1] < self.patch_size
                or img.shape[2] < self.patch_size
                or img.shape[3] < self.patch_size
            ):
                img = np.pad(
                    img,
                    (
                        (0, 0),
                        (0, max(self.patch_size - img.shape[1], 0)),
                        (0, max(self.patch_size - img.shape[2], 0)),
                        (0, max(self.patch_size - img.shape[3], 0)),
                    ),
                    mode="constant",
                    constant_values=2,
                )
                label = np.pad(
                    label,
                    (
                        (0, 0),
                        (0, max(self.patch_size - label.shape[1], 0)),
                        (0, max(self.patch_size - label.shape[2], 0)),
                        (0, max(self.patch_size - label.shape[3], 0)),
                    ),
                    mode="constant",
                    constant_values=0,
                )
            assert (
                img.shape[1] == self.patch_size
                and img.shape[2] == self.patch_size
                and img.shape[3] == self.patch_size
            ), f"Image shape is {img.shape} instead of {self.patch_size}"
            return {"image": img, "label": label}

        x_crop, y_crop, z_crop = self.patch_size, self.patch_size, self.patch_size
        x_start = np.random.randint(0, x - x_crop)
        y_start = np.random.randint(0, y - y_crop)
        z_start = np.random.randint(0, z - z_crop)
        img = img[
            :,
            x_start : x_start + x_crop,
            y_start : y_start + y_crop,
            z_start : z_start + z_crop,
        ]
        label = label[
            :,
            x_start : x_start + x_crop,
            y_start : y_start + y_crop,
            z_start : z_start + z_crop,
        ]

        assert (
            img.shape[1] == self.patch_size
            and img.shape[2] == self.patch_size
            and img.shape[3] == self.patch_size
        ), f"Image shape is {img.shape} instead of {self.patch_size}"
        return {"image": img, "label": label}

    def load_data_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Loads a single image-label pair from the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to be loaded.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing an image and its corresponding label.
        """
        label = read_nifti(self.data_paths[idx][1])
        img = read_nifti(self.data_paths[idx][0])
        label = np.transpose(label, (1, 2, 0))
        img = np.transpose(img, (1, 2, 0))
        ds_token = get_dataset_token(self.data_paths[idx][0])
        return {"image": img, "label": label, "dataset": ds_token}

    def load_data(self) -> None:
        """
        Loads image-label pairs into memory from the specified directories.

        Notes
        -----
        This function assumes the image and label files are in NIFTI format.
        """
        logging.info("Loading images into dataset.")
        self.imgs = []
        self.labels = []
        self.dataset_labels = []
        for entry_num in tqdm(range(len(self.data_paths))):
            sample_dict = self.load_data_sample(entry_num)
            self.imgs.append(sample_dict["image"])
            self.labels.append(sample_dict["label"])
            self.dataset_labels.append(sample_dict["dataset"])

    def initialize_imgs_paths(self) -> None:
        """
        Initializes the list of paths to image-label pairs.

        Notes
        -----
        This function assumes the image and label files are in parallel directories
        and have the same file base names.
        """
        self.data_paths = []
        for filename in os.listdir(self.label_folder):
            label_filename = os.path.join(self.label_folder, filename)
            filename = filename[:-7] + "_0000.nii.gz"
            img_filename = os.path.join(self.img_folder, filename)
            self.data_paths.append((img_filename, label_filename))

    def test(self, test_folder: str, num_files: int = 20) -> None:
        """
        Tests the data loading and augmentation process.

        The 2D images and corresponding labels are generated and then
            saved in the specified directory for inspection.

        Parameters
        ----------
        test_folder : str
            The path to the directory where the generated images and labels
            will be saved.
        num_files : int, default 20
            The number of image-label pairs to be generated and saved.
        """
        os.makedirs(test_folder, exist_ok=True)

        for i in range(num_files):
            test_sample = self.__getitem__(i % self.__len__())
            for num_img in range(0, test_sample["image"].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_img{i}_group{num_img}.png"),
                    test_sample["image"][0, :, :, num_img],
                )

            for num_mask in range(0, test_sample["label"][0].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_mask{i}_group{num_mask}.png"),
                    test_sample["label"][0][0, :, :, num_mask],
                )

            for num_mask in range(0, test_sample["label"][1].shape[0], 15):
                io.imsave(
                    os.path.join(test_folder, f"test_mask_ds2_{i}_group{num_mask}.png"),
                    test_sample["label"][1][0, :, :, num_mask],
                )


def get_dataset_token(patch_name):
    """
    Get the dataset token from the patch name.

    Parameters
    ----------
    patch_name : str
        The name of the patch.

    Returns
    -------
    str
        The dataset token.

    """
    basename = os.path.basename(patch_name)
    dataset_token = basename.split("_")[0]
    return dataset_token
