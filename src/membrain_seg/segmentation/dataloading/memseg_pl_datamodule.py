import os
from typing import Optional

import pytorch_lightning as pl
from monai.data import DataLoader

from membrain_seg.segmentation.dataloading.memseg_dataset import CryoETMemSegDataset


class MemBrainSegDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning datamodule for membrane segmentation.

    Parameters
    ----------
    data_dir : str
        The directory where the data resides. The directory should have a
        specific structure.
    batch_size : int
        The batch size for the data loaders.
    num_workers : int
        The number of workers to use in the data loaders.
    aug_prob_to_one : bool, default False
        Whether to apply data augmentation.

    Attributes
    ----------
    train_img_dir : str
        The path to the directory containing training images.
    train_lab_dir : str
        The path to the directory containing training labels.
    val_img_dir : str
        The path to the directory containing validation images.
    val_lab_dir : str
        The path to the directory containing validation labels.
    train_dataset : CryoETMemSegDataset
        The training dataset.
    val_dataset : CryoETMemSegDataset
        The validation dataset.
    test_dataset : CryoETMemSegDataset
        The test dataset.
    """

    def __init__(self, data_dir, batch_size, num_workers, aug_prob_to_one=False, missing_wedge_aug=False, fourier_amplitude_aug=False):
        """Initialization of data paths and data loaders.

        The data_dir should have the following structure:
        data_dir/
        ├── imagesTr/       # Directory containing training images
        │   ├── img1.nii.gz    # Image file (currently requires nii.gz format)
        │   ├── img2.nii.gz    # Image file
        │   └── ...
        ├── imagesVal/      # Directory containing validation images
        │   ├── img1.nii.gz    # Image file
        │   ├── img2.nii.gz    # Image file
        │   └── ...
        ├── labelsTr/       # Directory containing training labels
        │   ├── label1.nii.gz  # Label file (currently requires nii.gz format)
        │   ├── label2.nii.gz  # Label file
        │   └── ...
        └── labelsVal/      # Directory containing validation labels
            ├── label1.nii.gz  # Label file
            ├── label2.nii.gz  # Label file
            └── ...
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_img_dir = os.path.join(self.data_dir, "imagesTr")
        self.train_lab_dir = os.path.join(self.data_dir, "labelsTr")
        self.val_img_dir = os.path.join(self.data_dir, "imagesVal")
        self.val_lab_dir = os.path.join(self.data_dir, "labelsVal")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_prob_to_one = aug_prob_to_one
        self.fourier_amplitude_aug = fourier_amplitude_aug
        self.missing_wedge_aug = missing_wedge_aug

    def setup(self, stage: Optional[str] = None):
        """
        Setups the datasets for different stages of the training process.

        If stage is None, the datasets for both the fit and test stages are setup.

        Parameters
        ----------
        stage : str, optional
            The stage of the training process.
            One of None, "fit" or "test".
        """
        if stage in (None, "fit"):
            self.train_dataset = CryoETMemSegDataset(
                img_folder=self.train_img_dir,
                label_folder=self.train_lab_dir,
                train=True,
                aug_prob_to_one=self.aug_prob_to_one,#
                fourier_amplitude_aug=self.fourier_amplitude_aug,
                missing_wedge_aug=self.missing_wedge_aug

            )
            self.val_dataset = CryoETMemSegDataset(
                img_folder=self.val_img_dir, label_folder=self.val_lab_dir, train=False
            )

        if stage in (None, "test"):
            self.test_dataset = CryoETMemSegDataset(
                self.data_dir, test=True, transform=self.transform
            )  # TODO: How to do prediction?

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns
        -------
        DataLoader
            The test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
