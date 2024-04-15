# ---------------------------------------------------------------------------------
# DISCLAIMER: This code is adapted from the MATLAB and C++ implementations provided
# in the paper titled "Robust membrane detection based on tensor voting for electron
# tomography" by Antonio Martinez-Sanchez, Inmaculada Garcia, Shoh Asano, Vladan Lucic,
# and Jose-Jesus Fernandez, published in the Journal of Structural Biology,
# Volume 186, Issue 1, 2014, Pages 49-61. The original work can be accessed via
# https://doi.org/10.1016/j.jsb.2014.02.015 and is used under conditions that adhere
# to the original licensing agreements. For details on the original license, refer to
# the publication: https://www.sciencedirect.com/science/article/pii/S1047847714000495.
# ---------------------------------------------------------------------------------
import numpy as np
import scipy.ndimage as ndimage

from membrain_seg.segmentation.skeletonization.angauss import angauss
from membrain_seg.segmentation.skeletonization.diff3d import diff3d
from membrain_seg.segmentation.skeletonization.eig3d import eig3d
from membrain_seg.segmentation.skeletonization.load_save_file import (
    load_tomogram,
    read_nifti,
    write_nifti,
)
from membrain_seg.segmentation.skeletonization.nonmaxsup import nonmaxsup


# This function should only take label path as input.
# The generated skeleton will be saved in the same folder.
def skeletonization(label_path: str):
    """
    Perform skeletonization on a tomogram segmentation.

    This function reads a segmentation file (label_path) which can be in .nii, .nii.gz,
    or .mrc format. It performs skeletonization on the segmentation where the label '1'
    represents the structures of interest. The resultant skeleton is saved in the same
    directory with '_skeleton' appended before the file extension.

    Parameters
    ----------
    label_path : str
        The path to the input file.
        This file should be a tomogram segmentation file in
        either .nii, .nii.gz, or .mrc format.
        The function will perform checks to confirm the file format.

    Returns
    -------
    None
        The function does not return any value.
        It saves the skeletonized image in the same directory as the input file,
        replacing the original extension with '_skeleton.nii'.

    Raises
    ------
    ValueError
        If the input file format is not supported (.nii, .nii.gz, or .mrc),
        a ValueError is raised.

    Notes
    -----
    The skeletonization is based on the computation of the distance transform
    of the regions abeled as '1' (foreground), followed by an eigenvalue analysis
    of the Hessian matrix of the distance transform to identify ridge-like
    structures corresponding to the centerlines of the segmented objects.

    Examples
    --------
    >>> skeletonization("/path/to/your/datafile.nii")
    Skeleton saved to: /path/to/your/datafile_skeleton.nii
    >>> membrain skeletonize --label-path /path/to/your/datafile.nii
    This command runs the skeletonization process from the command line.
    """
    # labels can be .nii files or mrc files
    if label_path.endswith(".nii") or label_path.endswith(".nii.gz"):
        # Load NIfTI files (.nii or .nii.gz)
        segmentation = read_nifti(label_path)
        save_path = label_path.replace(".nii", "_skeleton.nii")
    elif label_path.endswith(".mrc"):
        # Load MRC files
        segmentation = load_tomogram(label_path)
        segmentation = segmentation.data
        save_path = label_path.replace(".mrc", "_skeleton.nii")
    else:
        # Error handling for unsupported file formats
        print(
            "Error: Label file format not supported. Please use .nii, .nii.gz, or .mrc."
        )

    # Segmentation consists of 0, 1 and 2 respectively
    # for background, mask and labels to be ignored.
    labels = (segmentation == 1) * 1.0

    print("Distance transform on original labels.")
    labels_dt = ndimage.distance_transform_edt(labels) * (-1)

    # Calculates partial derivative along 3 dimensions.
    print("Computing partial derivative.")
    gradX = diff3d(labels_dt, 0)
    gradY = diff3d(labels_dt, 1)
    gradZ = diff3d(labels_dt, 2)

    # Calculates Hessian tensor
    print("Computing Hessian tensor.")
    hessianXX = diff3d(gradX, 0)
    hessianYY = diff3d(gradY, 1)
    hessianZZ = diff3d(gradZ, 2)
    hessianXY = diff3d(gradX, 1)
    hessianXZ = diff3d(gradX, 2)
    hessianYZ = diff3d(gradY, 2)

    # Smoothing
    print("Gaussian filtering.")
    std = 0.75  # Gaussian standard deviation
    hessianXX = angauss(hessianXX, std, 1)
    hessianYY = angauss(hessianYY, std, 1)
    hessianZZ = angauss(hessianZZ, std, 1)
    hessianXY = angauss(hessianXY, std, 1)
    hessianXZ = angauss(hessianXZ, std, 1)
    hessianYZ = angauss(hessianYZ, std, 1)

    # Solve Eigen problem
    print("Computing Eigenvalues and Eigenvectors, this step can take a few minutes.")
    print(
        "In case the execution of the program is terminated unexpectedly, "
        "attempt to rerun it using smaller data segments or patches."
    )
    first_eigenvalue, first_eigenvector = eig3d(
        hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ
    )

    # Non-maximum suppression
    print("Genration of skeleton based on non-maximum suppression algorithm.")
    first_eigenvalue = ndimage.gaussian_filter(first_eigenvalue, sigma=1)
    first_eigenvalue = np.abs(first_eigenvalue)
    Skeleton = nonmaxsup(
        first_eigenvalue,
        first_eigenvector[:, :, :, 0],
        first_eigenvector[:, :, :, 1],
        first_eigenvector[:, :, :, 2],
        labels,
    )

    # Save the skeleton
    write_nifti(save_path, Skeleton)
    print("Skeleton saved to: ", save_path)
