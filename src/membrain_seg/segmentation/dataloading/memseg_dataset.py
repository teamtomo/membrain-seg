import os
from typing import Dict

import imageio as io
import numpy as np
from torch.utils.data import Dataset

from membrain_seg.segmentation.dataloading.data_utils import (
    normalize_tomogram,
    read_nifti,
)
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
        vec_folder: str = None,
        train: bool = False,
        return_normals: bool = False,
        use_fourier_aug: bool = False,
        use_mw_aug: bool = False,
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
        vec_folder : str, default None
            The path to the directory containing the normal vector files.
        train : bool, default False
            A flag indicating whether the dataset is used for training or validation.
        return_normals : bool, default False
            A flag indicating whether the normals should be returned or not.
        use_fourier_aug : bool, default False
            A flag indicating whether the Fourier augmentation should be used or not.
        use_mw_aug : bool, default False
            A flag indicating whether the MW augmentation should be used or not.
        aug_prob_to_one : bool, default False
            A flag indicating whether the probability of augmentation should be set
            to one or not.
        """
        self.train = train
        self.img_folder, self.label_folder, self.vec_folder = (
            img_folder,
            label_folder,
            vec_folder,
        )
        self.return_normals = return_normals
        self.initialize_imgs_paths()
        self.load_data()
        self.transforms = (
            get_training_transforms(
                prob_to_one=aug_prob_to_one,
                use_vectors=return_normals,
                use_fourier_aug=use_fourier_aug,
                use_mw_aug=use_mw_aug,
            )
            if self.train
            else get_validation_transforms(use_vectors=return_normals)
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
        if self.return_normals:
            idx_dict["vectors"] = np.expand_dims(self.normals[idx], 0)
        idx_dict = self.transforms(idx_dict)
        idx_dict["dataset"] = self.dataset_labels[idx]
        if self.return_normals:
            idx_dict["vectors"] = [
                np.transpose(entry.squeeze(), axes=(3, 0, 1, 2))
                for entry in idx_dict["vectors"]
            ]
        return idx_dict

    def __len__(self) -> int:
        """
        Returns the number of image-label pairs in the dataset.

        Returns
        -------
        int
            The number of image-label pairs in the dataset.
        """
        return len(self.imgs)

    def load_data(self) -> None:
        """
        Loads image-label pairs into memory from the specified directories.

        In addition to the image-label pairs, the normals are also loaded
        if the return_normals flag is set to True.

        Notes
        -----
        This function assumes the image and label files are in NIFTI format.
        """
        print("Loading images into dataset.")
        self.imgs = []
        self.labels = []
        self.normals = []
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
            img = normalize_tomogram(img, cut_extreme_values=True)

            self.imgs.append(img)
            self.labels.append(label)
            self.dataset_labels.append(get_dataset_token(entry[0]))

            if self.return_normals:
                norm = np.stack(
                    [
                        np.transpose(read_nifti(entry[2][i]), (1, 2, 0))
                        for i in range(3)
                    ],
                    axis=-1,
                )
                assert not np.isnan(norm).any(), f"NaNs detected in normals for {entry}"
                assert not np.isinf(norm).any(), f"Infs detected in normals for {entry}"
                self.normals.append(norm)

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
            filetoken = filename[:-7]
            filename = filename[:-7] + "_0000.nii.gz"
            img_filename = os.path.join(self.img_folder, filename)
            self.data_paths.append((img_filename, label_filename))

            if self.return_normals:
                vec_filenames = [
                    os.path.join(
                        self.vec_folder, filetoken + "_norm" + str(2) + ".nii.gz"
                    ),
                    os.path.join(
                        self.vec_folder, filetoken + "_norm" + str(1) + ".nii.gz"
                    ),
                    os.path.join(
                        self.vec_folder, filetoken + "_norm" + str(3) + ".nii.gz"
                    ),
                ]
                self.data_paths[-1] += (vec_filenames,)

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

    def test_with_normals(self, test_folder, num_files=20):
        """
        Testing of dataloading and augmentations.

        To test data loading and augmentations, 2D images are
        stored for inspection.
        """
        os.makedirs(test_folder, exist_ok=True)

        for i in range(num_files):
            print("sample", i + 1)
            idx = np.random.randint(0, len(self))
            test_sample = self.__getitem__(idx)

            test_sample["image"] = np.array(test_sample["image"])
            test_sample["label"][0] = np.array(test_sample["label"][0])
            test_sample["vectors"][0] = np.array(test_sample["vectors"][0])

            for num_img in range(0, test_sample["image"].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_img{i}_group{num_img}.png"),
                    (test_sample["image"] * 10 + 128).astype(np.uint8)[
                        0, :, :, num_img
                    ],
                )

            for num_mask in range(0, test_sample["label"][0].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_mask{i}_group{num_mask}.png"),
                    (test_sample["label"][0] * 64).astype(np.uint8)[0, :, :, num_mask],
                )

            for num_mask in range(0, test_sample["vectors"][0].shape[-2], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_vecs{i}_group{num_mask}.png"),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask
                    ],
                )
                io.imsave(
                    os.path.join(
                        test_folder, f"test_vecs{i}_group{num_mask}_comp1.png"
                    ),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask, 0
                    ],
                )
                io.imsave(
                    os.path.join(
                        test_folder, f"test_vecs{i}_group{num_mask}_comp2.png"
                    ),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask, 1
                    ],
                )
                io.imsave(
                    os.path.join(
                        test_folder, f"test_vecs{i}_group{num_mask}_comp3.png"
                    ),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask, 2
                    ],
                )

            for num_mask in range(0, test_sample["label"][1].shape[0], 15):
                io.imsave(
                    os.path.join(test_folder, f"test_mask_ds2_{i}_group{num_mask}.png"),
                    (test_sample["label"][1] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask
                    ],
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
