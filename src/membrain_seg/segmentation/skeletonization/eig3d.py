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
from typing import List, Tuple

import numpy as np
import torch


def batch_mask_eigendecomposition_3d(
    filtered_hessian: List[torch.Tensor], batch_size: int, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform batch eigendecomposition on a 3D Hessian matrix using a binary mask to
    select voxels.

    This function processes only those voxels where the label is set to 1,
    computing the largest eigenvalue and its corresponding eigenvector for
    each selected voxel. It handles large 3D datasets efficiently by performing
    computations in batches and leveraging GPU acceleration.

    Parameters
    ----------
    filtered_hessian : List[torch.Tensor]
        A list of six torch.Tensors representing the Hessian matrix components:
        [hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ]
    batch_size : int
        The number of matrices to include in each batch for processing.
    labels : np.ndarray
        A 3D numpy array representing the binary mask where 1 indicates a voxel
        to be processed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - first_eigenvalues:
          A 3D array with the largest eigenvalues for the processed voxels.
        - first_eigenvectors:
          A 4D array with the corresponding eigenvectors for these eigenvalues.
    """
    hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ = filtered_hessian
    del filtered_hessian
    Nx, Ny, Nz = hessianXX.shape

    # Set the batch size to the total number of voxels
    # if no specified batch size is given
    if batch_size is None:
        batch_size = Nx * Ny * Nz
    print("batch_size=", batch_size)

    # Identify coordinates where computation is needed
    active_voxel_coords = np.where(labels == 1)
    x_indices, y_indices, z_indices = active_voxel_coords
    num_active_voxels = x_indices.shape[0]

    # Prepare a tensor stack for the selected Hessian matrix components
    hessian_components = torch.stack(
        [
            hessianXX[x_indices, y_indices, z_indices],
            hessianXY[x_indices, y_indices, z_indices],
            hessianXZ[x_indices, y_indices, z_indices],
            hessianXY[x_indices, y_indices, z_indices],
            hessianYY[x_indices, y_indices, z_indices],
            hessianYZ[x_indices, y_indices, z_indices],
            hessianXZ[x_indices, y_indices, z_indices],
            hessianYZ[x_indices, y_indices, z_indices],
            hessianZZ[x_indices, y_indices, z_indices],
        ],
        dim=-1,
    ).view(-1, 3, 3)
    del hessianXX, hessianYY, hessianZZ, hessianXY, hessianXZ, hessianYZ
    print("Hessian component matrix shape:", hessian_components.shape)

    # Initialize output arrays
    first_eigenvalues = np.zeros((Nx, Ny, Nz), dtype=np.complex64)
    first_eigenvectors = np.zeros((Nx, Ny, Nz, 3), dtype=np.complex64)

    # Process in batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(0, num_active_voxels, batch_size):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # print('i=', i)
        # print(f"Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        # print(f"Cached:    {torch.cuda.memory_reserved(0)/1e9:.2f} GB")

        i_end = min(i + batch_size, num_active_voxels)
        batch_matrix = hessian_components[i:i_end, :, :]

        # Compute eigenvalues and eigenvectors for this batch
        eigenvalues, eigenvectors = torch.linalg.eig(batch_matrix.to(device))
        max_eigenvalue_idx = torch.argmax(torch.abs(eigenvalues), dim=-1)
        batch_first_eigenvalues = eigenvalues[
            torch.arange(len(max_eigenvalue_idx)), max_eigenvalue_idx
        ]
        batch_first_eigenvectors = eigenvectors[
            torch.arange(len(max_eigenvalue_idx)), :, max_eigenvalue_idx
        ]

        # Store results back to CPU to save cuda memory
        first_eigenvalues[
            x_indices[i:i_end], y_indices[i:i_end], z_indices[i:i_end]
        ] = batch_first_eigenvalues.cpu().numpy()
        first_eigenvectors[
            x_indices[i:i_end], y_indices[i:i_end], z_indices[i:i_end], :
        ] = (batch_first_eigenvectors.view(-1, 3).cpu().numpy())

    return first_eigenvalues, first_eigenvectors
