import numpy as np
from scipy.linalg import eigh


def eigendecomposition(
    hessianXX: np.ndarray,
    hessianYY: np.ndarray,
    hessianZZ: np.ndarray,
    hessianXY: np.ndarray,
    hessianXZ: np.ndarray,
    hessianYZ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the eigenproblem for a set of 3x3 symmetric matrices,
    representing Hessian matrices at each voxel.

    This function computes the largest eigenvalue and corresponding
    eigenvector for each matrix, which are used for further analysis
    such as in structure analysis in image processing.

    Parameters
    ----------
    hessianXX, hessianYY, hessianZZ : np.ndarray
        Diagonal components of the Hessian matrices.
    hessianXY, hessianXZ, hessianYZ : np.ndarray
        Off-diagonal components of the Hessian matrices.

    Returns
    -------
    np.ndarray
        The largest eigenvalues of the Hessian matrices.
    np.ndarray, np.ndarray, np.ndarray
        The components of the eigenvectors corresponding to the largest eigenvalues.

    Notes
    -----
    The function is designed to process a large number of small
    matrices (3x3) typically found in voxel-wise computations
    in 3D imaging studies.
    The eigenvalues and eigenvectors are sorted by the eigenvalues' magnitudes.
    """
    m = len(hessianXX)
    Qo = np.zeros((m, 3, 3), dtype=complex)
    w = np.zeros((m, 3), dtype=complex)

    for i in range(m):
        A = np.array(
            [
                [hessianXX[i], hessianXY[i], hessianXZ[i]],
                [hessianXY[i], hessianYY[i], hessianYZ[i]],
                [hessianXZ[i], hessianYZ[i], hessianZZ[i]],
            ]
        )

        # Use Python package scipy.linalg
        # to compute the eigenvalues and eigenvectors of the symmetric matrix A
        w_i, Qo_i = eigh(A)
        w[i] = w_i[::-1]  # Reversing to get the largest eigenvalue first
        Qo_i[:, [0, 2]] = Qo_i[
            :, [2, 0]
        ]  # Swapping to correct the order of eigenvectors
        Qo[i] = Qo_i

    return w[:, 0], Qo[:, 0, 0], Qo[:, 1, 0], Qo[:, 2, 0]


def eig3d(
    hessianXX: np.ndarray,
    hessianYY: np.ndarray,
    hessianZZ: np.ndarray,
    hessianXY: np.ndarray,
    hessianXZ: np.ndarray,
    hessianYZ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the first eigenvalue and corresponding eigenvector
    of the Hessian matrix at each voxel.

    This function takes the components of the Hessian matrix at each point in a 3D grid
    and computes the eigenvalue and eigenvector corresponding to the largest absolute
    eigenvalue at each point.

    Parameters
    ----------
    hessianXX, hessianYY, hessianZZ : np.ndarray
        Diagonal components of the Hessian matrix.
    hessianXY, hessianXZ, hessianYZ : np.ndarray
        Off-diagonal components of the Hessian matrix.

    Returns
    -------
    first_eigenvalue : np.ndarray
        The first (largest) eigenvalue at each point in the 3D grid.
    first_eigenvector : np.ndarray
        The corresponding eigenvector of the first eigenvalue at each point
        in the 3D grid.
        This is returned as a 4D array where the last dimension has size 3,
        representing the vector components.

    Notes
    -----
    The eigenvalue and eigenvector are computed using an eigendecomposition function
    tailored to symmetric 3x3 matrices, typical for Hessian matrices derived from
    image data. The computation is vectorized over the entire 3D grid for efficiency.
    """
    # Get the size of input
    Nx, Ny, Nz = hessianXX.shape
    # Flatten the input matrices for bulk processing
    first_eigenvalue, first_eigen_x, first_eigen_y, first_eigen_z = eigendecomposition(
        hessianXX.flatten(),
        hessianYY.flatten(),
        hessianZZ.flatten(),
        hessianXY.flatten(),
        hessianXZ.flatten(),
        hessianYZ.flatten(),
    )

    # Reshape the first eigenvalue to 3D
    first_eigenvalue = first_eigenvalue.reshape((Nx, Ny, Nz))

    # Prepare the first eigenvector array
    first_eigenvector = np.zeros((Nx, Ny, Nz, 3), dtype=complex)
    first_eigenvector[:, :, :, 0] = first_eigen_x.reshape((Nx, Ny, Nz))
    first_eigenvector[:, :, :, 1] = first_eigen_y.reshape((Nx, Ny, Nz))
    first_eigenvector[:, :, :, 2] = first_eigen_z.reshape((Nx, Ny, Nz))

    return first_eigenvalue, first_eigenvector
