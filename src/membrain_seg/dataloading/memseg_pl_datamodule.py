import os

import pytorch_lightning as pl
from monai.data import DataLoader

from membrain_seg.dataloading.memseg_dataset import CryoETMemSegDataset


class MemBrainSegDataModule(pl.LightningDataModule):
    """Pytorch Lightning datamodule for membrane segmentation."""

    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.train_img_dir = os.path.join(self.data_dir, "imagesTr")
        self.train_lab_dir = os.path.join(self.data_dir, "labelsTr")
        self.val_img_dir = os.path.join(self.data_dir, "imagesVal")
        self.val_lab_dir = os.path.join(self.data_dir, "labelsVal")
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Load datasets."""
        if stage in (None, "fit"):
            self.train_dataset = CryoETMemSegDataset(
                img_folder=self.train_img_dir,
                label_folder=self.train_lab_dir,
                train=True,
            )
            self.val_dataset = CryoETMemSegDataset(
                img_folder=self.val_img_dir, label_folder=self.val_lab_dir, train=False
            )

        if stage in (None, "test"):
            self.test_dataset = CryoETMemSegDataset(
                self.data_dir, test=True, transform=self.transform
            )  # TODO: How to do prediction?

    def train_dataloader(self):
        """Define training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Define validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Define test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
