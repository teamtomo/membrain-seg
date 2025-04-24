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
import logging

import numpy as np
import scipy.ndimage as ndimage
import torch

from membrain_seg.segmentation.skeletonization.diff3d import (
    compute_gradients,
    compute_hessian,
)
from membrain_seg.segmentation.skeletonization.eig3d import (
    batch_mask_eigendecomposition_3d,
)
from membrain_seg.segmentation.skeletonization.nonmaxsup import nonmaxsup
from membrain_seg.segmentation.training.surface_dice import apply_gaussian_filter


def skeletonization(segmentation: np.ndarray, batch_size: int, device: str=None) -> np.ndarray:
    """
    Perform skeletonization on a tomogram segmentation.

    This function skeletonizes the input segmentation where the non-zero labels
    represent the structures of interest. The resultan skeleton is saved with
    '_skel' appended after the filename.

    Parameters
    ----------
    segmentation : ndarray
        Tomogram segmentation as a numpy array, where non-zero values represent
        the structures of interest.

    batch_size : int
        The number of elements to process in one batch during eigen decomposition.
        Useful for managing memory usage.

    Returns
    -------
    ndarray
        Returns the skeletonized image as a numpy array.

    Notes
    -----
    The skeletonization is based on the computation of the distance transform
    of the non-zero regions (foreground), followed by an eigenvalue analysis
    of the Hessian matrix of the distance transform to identify ridge-like
    structures corresponding to the centerlines of the segmented objects.

    Examples
    --------
    >>> membrain skeletonize --label-path <path> --out-folder <output-directory>
        --batch-size 1000000
    This command runs the skeletonization process from the command line.
    """
    # Convert non-zero segmentation values to 1.0
    labels = (segmentation > 0) * 1.0

    labels_dt = ndimage.distance_transform_edt(labels) * (-1)

    # Calculates partial derivative along 3 dimensions.
    gradX, gradY, gradZ = compute_gradients(labels_dt)

    # Calculates Hessian tensor
    hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ = compute_hessian(
        gradX, gradY, gradZ
    )
    hessians = [hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ]
    del gradX, gradY, gradZ

    # Apply Gaussian filter with the same sigma value for all dimensions
    # Load hessian tensors on GPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        device = torch.device(device)

    filtered_hessian = [
        apply_gaussian_filter(
            torch.from_numpy(comp).float().to(device).unsqueeze(0).unsqueeze(0),
            kernel_size=9,
            sigma=1.0,
        )
        .squeeze()
        .to("cpu")
        for comp in hessians
    ]

    # Solve Eigen problem
    logging.info("Computing Eigenvalues and Eigenvectors.")
    logging.info(
        "In case the execution of the program is terminated unexpectedly, "
        "attempt to rerun it using smaller segmentation patches"
        "or give a specified batch size as input, e.g. batch_size=1000000."
    )
    first_eigenvalue, first_eigenvector = batch_mask_eigendecomposition_3d(
        filtered_hessian, batch_size, labels
    )

    # Non-maximum suppression
    first_eigenvalue = ndimage.gaussian_filter(first_eigenvalue, sigma=1)
    skeleton = nonmaxsup(
        first_eigenvalue,
        first_eigenvector[:, :, :, 0],
        first_eigenvector[:, :, :, 1],
        first_eigenvector[:, :, :, 2],
        labels,
    )
    return skeleton
