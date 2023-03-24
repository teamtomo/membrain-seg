import os

# from skimage import io
import imageio as io
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset


def read_nifti(nifti_file):
    """Read nifti files. This will be redundant once we move to mrc files I guess?."""
    a = np.array(sitk.GetArrayFromImage(sitk.ReadImage(nifti_file)), dtype=float)
    return a


class CryoETMemSegDataset(Dataset):
    """Dataset of Cryo-ET membrane segmentation patches."""

    def __init__(self, img_folder, label_folder):
        self.img_folder, self.label_folder = img_folder, label_folder
        self.initialize_imgs_paths()
        self.load_data()

    def __getitem__(self, idx):
        """Returns image & labels of sample with index idx."""
        return np.expand_dims(self.imgs[idx], 0), np.expand_dims(self.labels[idx], 0)

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_paths)

    def load_data(self):
        """
        Loading of samples.

        Load data given data paths. This should be adjusted to our new
        dataloading pipeline once it's set up.
        """
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

    def test(self, test_folder="", num_files=20):
        """
        Testing of dataloading and augmentations.

        To test data loading and augmentations, 2D images are
        stored for inspection.
        """
        os.makedirs(test_folder, exist_ok=True)
        for i in range(num_files):
            test_sample = self.__getitem__(i % self.__len__())

            for num_img in range(test_sample[0].shape[0]):
                io.imsave(
                    os.path.join(test_folder, f"test_img{i}_group{num_img}.tif"),
                    test_sample[0][num_img, ...],
                    check_contrast=False,
                )

            for num_mask in range(test_sample[1].shape[0]):
                io.imsave(
                    os.path.join(test_folder, f"test_mask{i}_group{num_mask}.tif"),
                    test_sample[1][num_mask, ...],
                    check_contrast=False,
                )


def collate_tr_val(augmentations):
    """
    Batch augmentation collate function.

    This collate function can perform data augmentations using
    batchgenerators package for the entire batch.
    Will be redundant once all augmentations are performed on single samples.
    """

    def cur_collate(batch):
        data_stack = np.stack([sample[0] for sample in batch], axis=0)
        label_stack = np.stack([sample[1] for sample in batch], axis=0)
        data_dict = {"data": data_stack, "seg": label_stack}
        for trafo in augmentations:
            data_dict = trafo(**data_dict)
        return data_dict["data"], data_dict["target"]

    return cur_collate
