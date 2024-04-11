import numpy as np
from membrain_seg.segmentation.skeletonization.eigendecomposition import (
    eigendecomposition,
)


def eig3d(Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
    """Get the first eigenvalues and eigenvectors."""
    # Preparing input data
    Nx, Ny, Nz = Ixx.shape
    N = Nx * Ny * Nz

    L1 = Ixx.reshape(N)
    L2 = Iyy.reshape(N)
    L3 = Izz.reshape(N)
    V1h = Ixy.reshape(N)
    V2h = Ixz.reshape(N)
    V3h = Iyz.reshape(N)

    L1, L2, L3, V1xm, V1ym, V1zm, V2xm, V2ym, V2zm, V3xm, V3ym, V3zm = (
        eigendecomposition(L1, L2, L3, V1h, V2h, V3h)
    )

    # Preparing output data
    L1 = L1.reshape((Nx, Ny, Nz))

    V1 = np.zeros((Nx, Ny, Nz, 3), dtype=complex)
    V1[:, :, :, 0] = V1xm.reshape((Nx, Ny, Nz))
    V1[:, :, :, 1] = V1ym.reshape((Nx, Ny, Nz))
    V1[:, :, :, 2] = V1zm.reshape((Nx, Ny, Nz))

    return L1, V1
