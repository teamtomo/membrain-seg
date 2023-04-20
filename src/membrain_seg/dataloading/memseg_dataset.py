import os

# from skimage import io
import imageio as io
import numpy as np
from torch.utils.data import Dataset

from membrain_seg.dataloading.data_utils import read_nifti
from membrain_seg.dataloading.memseg_augmentation import (
    get_training_transforms,
    get_validation_transforms,
)


class CryoETMemSegDataset(Dataset):
    """Dataset of Cryo-ET membrane segmentation patches."""

    def __init__(self, img_folder, label_folder, train=False, aug_prob_to_one=False):
        self.train = train
        self.img_folder, self.label_folder = img_folder, label_folder
        self.initialize_imgs_paths()
        self.load_data()
        self.transforms = (
            get_training_transforms(prob_to_one=aug_prob_to_one)
            if self.train
            else get_validation_transforms()
        )

    def __getitem__(self, idx):
        """Returns image & labels of sample with index idx."""
        idx_dict = {
            "image": np.expand_dims(self.imgs[idx], 0),
            "label": np.expand_dims(self.labels[idx], 0),
        }
        idx_dict = self.transforms(idx_dict)
        return idx_dict

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_paths)

    def load_data(self):
        """
        Loading of samples.

        Load data given data paths. This should be adjusted to our new
        dataloading pipeline once it's set up.
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

    def initialize_imgs_paths(self):
        """
        Initialization of data paths.

        Initialize paths to images and labels given the training directory.
        This should be adjusted once the new dataloading pipeline is set up.
        """
        self.data_paths = []
        for filename in os.listdir(self.label_folder):
            label_filename = os.path.join(self.label_folder, filename)
            filename = filename[:-7] + "_0000.nii.gz"
            img_filename = os.path.join(self.img_folder, filename)
            self.data_paths.append((img_filename, label_filename))

    def test(self, test_folder, num_files=20):
        """
        Testing of dataloading and augmentations.

        To test data loading and augmentations, 2D images are
        stored for inspection.
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
