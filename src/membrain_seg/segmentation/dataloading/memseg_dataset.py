import os
from typing import Dict

# from skimage import io
import imageio as io
import numpy as np
from torch.utils.data import Dataset

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
        """
        self.train = train
        self.img_folder, self.label_folder = img_folder, label_folder
        self.initialize_imgs_paths()
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
        idx_dict = {
            "image": np.expand_dims(self.imgs[idx], 0),
            "label": np.expand_dims(self.labels[idx], 0),
        }
        idx_dict = self.transforms(idx_dict)
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

    def load_data(self) -> None:
        """
        Loads image-label pairs into memory from the specified directories.

        Notes
        -----
        This function assumes the image and label files are in NIFTI format.
        """
        print("Loading images into dataset.")
        self.imgs = []
        self.labels = []
        for entry in self.data_paths:
            label = read_nifti(
                entry[1]
            )  # TODO: Change this to be applicable to .mrc images
            img = read_nifti(entry[0])
            label = np.transpose(
                label, (1, 2, 0)
            )  # TODO: Needed? Probably no? z-axis should not matter
            img = np.transpose(img, (1, 2, 0))
            self.imgs.append(img)
            self.labels.append(label)

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
