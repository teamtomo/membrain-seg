import os
from typing import Dict

# from skimage import io
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
        fourier_amplitude_aug: bool = False,
        missing_wedge_aug: bool = False,
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
        fourier_amplitude_aug : bool, default False
            A flag indicating whether Fourier amplitude augmentation should be applied.
        missing_wedge_aug : bool, default False
            A flag indicating whether missing wedge augmentation should be applied.

        """
        self.train = train
        self.img_folder, self.label_folder = img_folder, label_folder
        self.initialize_imgs_paths()
        self.load_data()
        self.transforms = (
            get_training_transforms(
                prob_to_one=aug_prob_to_one,
                fourier_amplitude_aug=fourier_amplitude_aug,
                missing_wedge_aug=missing_wedge_aug,
            )
            if self.train
            else get_validation_transforms()
        )

    def __getitem__(self, idx: int, orig: bool = False) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing an image-label pair for the provided index.

        Data augmentations are applied before returning the dictionary.

        Parameters
        ----------
        idx : int
            Index of the sample to be fetched.
        orig : bool, default False
            A flag indicating whether the original image should be returned or not.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing an image and its corresponding label.
        """
        idx_dict = {
            "image": np.expand_dims(self.imgs[idx], 0),
            "label": np.expand_dims(self.labels[idx], 0),
        }
        if not orig:
            idx_dict = self.transforms(idx_dict)
        idx_dict["dataset"] = self.dataset_labels[idx]
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
        self.dataset_labels = []
        for entry in self.data_paths:
            label = read_nifti(
                entry[1]
            )  # TODO: Change this to be applicable to .mrc images
            img = read_nifti(entry[0])
            label = np.transpose(
                label, (1, 2, 0)
            )  # TODO: Needed? Probably no? z-axis should not matter
            img = np.transpose(img, (1, 2, 0))
            img -= np.mean(img)
            img /= np.std(img)

            self.imgs.append(img)
            self.labels.append(label)
            self.dataset_labels.append(get_dataset_token(entry[0]))

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
        from numpy import fft as fft

        from membrain_seg.segmentation.dataloading.data_utils import store_tomogram

        for i in range(num_files):
            test_sample = self.__getitem__(i % self.__len__(), orig=False)
            test_sample_orig = self.__getitem__(i % self.__len__(), orig=True)
            test_sample_fft = fft.fftshift(fft.fftn(test_sample["image"][0, :, :, :]))
            test_sample_orig_fft = fft.fftshift(
                fft.fftn(test_sample_orig["image"][0, :, :, :])
            )
            print(i)
            store_tomogram(
                os.path.join(test_folder, f"test_img{i}.mrc"),
                test_sample["image"][0, :, :, :],
            )
            store_tomogram(
                os.path.join(test_folder, f"test_img_fft{i}.mrc"),
                test_sample_fft,
            )
            store_tomogram(
                os.path.join(test_folder, f"test_img_orig{i}.mrc"),
                test_sample_orig["image"][0, :, :, :],
            )
            store_tomogram(
                os.path.join(test_folder, f"test_img_orig_fft{i}.mrc"),
                test_sample_orig_fft,
            )
            store_tomogram(
                os.path.join(test_folder, f"test_img{i}_lab.mrc"),
                test_sample["label"][0][0, :, :, :],
            )


def get_dataset_token(patch_name):
    """Returns the dataset token based on the provided patch name."""
    basename = os.path.basename(patch_name)

    if (
        basename.startswith("50_")
        or basename.startswith("633_")
        or basename.startswith("165_")
        or basename.startswith("24_")
        or basename.startswith("38_")
        or basename.startswith("441_")
        or basename.startswith("54_")
        or basename.startswith("8_")
    ):
        if "_raw" not in basename:
            return "Chlamy"
        else:
            return "Chlamy_raw"
    if basename.startswith("HDCR_"):
        return "HDCR"
    if basename.startswith("AntonioSim"):
        return "AntonioSim"
    if basename.startswith("CryoTomoSim"):
        return "CTS"
    if basename.startswith("tomo"):
        if "_raw" not in basename:
            return "Spinach"
        else:
            return "Spinach_raw"
    if basename.startswith("TS_"):
        return "Deepict"
    if basename.startswith("YTC"):
        return "Atty"
    if basename.startswith("Matthias"):
        return "Matthias"
    return "unknown"
    # raise IOError("dataset token not known!!")
