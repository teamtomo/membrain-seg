import numpy as np
from membrain_seg.segmentation.skeletonization.nonmaxsup_kernel import nonmaxsup_kernel


def nonmaxsup(I, Vx, Vy, Vz, labels):
    """
    NONMAXSUP  Ridge centreline detection by non-maximum suppresion criteria.
    INPUT:
    I: Input tomogram, input data must be single or double
    Vi: The i coordinate of the major eigenvector
    labels: Original labels for mask
    OUTPUT:
    B: binary output
    """
    # cut the boundary
    Nx, Ny, Nz = I.shape
    H = np.zeros((Nx, Ny, Nz))
    b = 1
    H[b : Nx - b, b : Ny - b, b : Nz - b] = 1
    labels = labels * H

    # Use mask here, only consider non-background voxels
    non_zero_coordinates = np.where(labels == 1)
    M = list(
        zip(non_zero_coordinates[0], non_zero_coordinates[1], non_zero_coordinates[2])
    )

    inter_factor = 0.71
    B = nonmaxsup_kernel(I, Vx, Vy, Vz, M, inter_factor)
    return B
