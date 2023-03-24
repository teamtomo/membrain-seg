import numpy as np
import SimpleITK as sitk


def read_nifti(nifti_file):
    """Read nifti files. This will be redundant once we move to mrc files I guess?."""
    a = np.array(sitk.GetArrayFromImage(sitk.ReadImage(nifti_file)), dtype=float)
    return a
