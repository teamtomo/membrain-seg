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


def surfaceness(I, s, labels):
    """The skeletonization algorithm is based on nonmax-suppression."""
    # Get Tensor
    Ix = diff3d(I, 0)
    Iy = diff3d(I, 1)
    Iz = diff3d(I, 2)

    print("Computing Hessian tensor")
    Ixx = diff3d(Ix, 0)
    Iyy = diff3d(Iy, 1)
    Izz = diff3d(Iz, 2)
    Ixy = diff3d(Ix, 1)
    Ixz = diff3d(Ix, 2)
    Iyz = diff3d(Iy, 2)

    # Smoothing
    Ixx = angauss(Ixx, s, 1)
    Iyy = angauss(Iyy, s, 1)
    Izz = angauss(Izz, s, 1)
    Ixy = angauss(Ixy, s, 1)
    Ixz = angauss(Ixz, s, 1)
    Iyz = angauss(Iyz, s, 1)

    # Solve Eigen problem
    print("Computing Eigenvalues and Eigenvectors, this step can take a few minutes")
    L1, V1 = eig3d(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

    # Non-maximum suppression
    print("Non-maximum suppression")
    L1 = ndimage.gaussian_filter(L1, sigma=1)
    L1 = np.abs(L1)
    P = nonmaxsup(L1, V1[:, :, :, 0], V1[:, :, :, 1], V1[:, :, :, 2], labels)
    return P


# this function should only take label path as input
def skeletonization(label_path):
    """
    The function takes tomogram segmentation (labels) as input.
    Skeleton is generated for original segmentation.
    """
    # labels can be .nii files or mrc files
    if label_path.endswith(".nii") or label_path.endswith(".nii.gz"):
        # Load NIfTI files (.nii or .nii.gz)
        seg = read_nifti(label_path)
        save_path = label_path.replace(".nii", "_skeleton.nii")
    elif label_path.endswith(".mrc"):
        # Load MRC files
        seg = load_tomogram(label_path)
        seg = seg.data
        save_path = label_path.replace(".mrc", "_skeleton.nii")
    else:
        # Error handling for unsupported file formats
        print(
            "Error: Label file format not supported. Please use .nii, .nii.gz, or .mrc."
        )

    # read labels
    labels = (seg == 1) * 1.0

    print("Distance transform")
    labels_dt = ndimage.distance_transform_edt(labels) * (-1)

    # skeletonization
    B = surfaceness(I=labels_dt, s=0.75, labels=labels)

    # Use original labels to filter noise
    Ske = B * labels

    write_nifti(save_path, Ske)
    print("Skeleton saved to: ", save_path)
