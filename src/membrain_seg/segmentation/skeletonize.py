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

from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)
from membrain_seg.segmentation.skeletonization.diff3d import (
    compute_gradients,
    compute_hessian,
)
from membrain_seg.segmentation.skeletonization.eig3d import eig3d
from membrain_seg.segmentation.skeletonization.nonmaxsup import nonmaxsup
from membrain_seg.segmentation.skeletonization.smoothing import process_hessian_tensors


# This function should only take label path as input.
# The generated skeleton will be saved in the same folder.
def skeletonization(label_path: str):
    """
    Perform skeletonization on a tomogram segmentation.

    This function reads a segmentation file (label_path). It performs skeletonization on
    the segmentation where the non-zero labels represent the structures of interest.
    The resultan skeleton is saved in the same directory with 'skeleton' appended
    after the filename.

    Parameters
    ----------
    label_path : str
        The path to the input file.
        This file should be a tomogram segmentation file.

    Returns
    -------
    None
        The function does not return any value.
        It saves the skeletonized image in the same directory as the input file.
        The skeletonized image is saved with the original filename followed
        by '.skeleton.mrc'.


    Notes
    -----
    The skeletonization is based on the computation of the distance transform
    of the non-zero regions (foreground), followed by an eigenvalue analysis
    of the Hessian matrix of the distance transform to identify ridge-like
    structures corresponding to the centerlines of the segmented objects.

    Examples
    --------
    >>> skeletonization("/path/to/your/datafile.mrc")
    Skeleton saved to: /path/to/your/datafile.skeleton.mrc
    >>> membrain skeletonize --label-path /path/to/your/datafile.mrc
    This command runs the skeletonization process from the command line.
    """
    # Read original segmentation
    segmentation = load_tomogram(label_path)
    segmentation = segmentation.data
    save_path = label_path + ".skeleton.mrc"

    # Convert non-zero segmentation values to 1.0
    labels = (segmentation > 0) * 1.0

    print("Distance transform on original labels.")
    labels_dt = ndimage.distance_transform_edt(labels) * (-1)

    # Calculates partial derivative along 3 dimensions.
    print("Computing partial derivative.")
    gradients = compute_gradients(labels_dt)
    gradX, gradY, gradZ = gradients

    # Calculates Hessian tensor
    print("Computing Hessian tensor.")
    hessians = compute_hessian(gradX, gradY, gradZ)
    hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ = hessians

    # Apply Gaussian filter with the same sigma value for all dimensions
    print("Applying Gaussian filtering.")
    hessian_components = [
        hessianXX,
        hessianYY,
        hessianZZ,
        hessianXY,
        hessianXZ,
        hessianYZ,
    ]
    filtered_hessian = process_hessian_tensors(hessian_components)
    (
        filtered_hessianXX,
        filtered_hessianYY,
        filtered_hessianZZ,
        filtered_hessianXY,
        filtered_hessianXZ,
        filtered_hessianYZ,
    ) = filtered_hessian

    # Solve Eigen problem
    print("Computing Eigenvalues and Eigenvectors, this step can take a few minutes.")
    print(
        "In case the execution of the program is terminated unexpectedly, "
        "attempt to rerun it using smaller data segments or patches."
    )
    first_eigenvalue, first_eigenvector = eig3d(
        filtered_hessianXX,
        filtered_hessianYY,
        filtered_hessianZZ,
        filtered_hessianXY,
        filtered_hessianXZ,
        filtered_hessianYZ,
    )

    # Non-maximum suppression
    print("Genration of skeleton based on non-maximum suppression algorithm.")
    first_eigenvalue = ndimage.gaussian_filter(first_eigenvalue, sigma=1)
    first_eigenvalue = np.abs(first_eigenvalue)
    skeleton = nonmaxsup(
        first_eigenvalue,
        first_eigenvector[:, :, :, 0],
        first_eigenvector[:, :, :, 1],
        first_eigenvector[:, :, :, 2],
        labels,
    )

    # Save the skeleton
    store_tomogram(save_path, skeleton)
    print("Skeleton saved to: ", save_path)
