from typing import List, Tuple

import numpy as np


def nonmaxsup_kernel(
    image: np.ndarray,
    vector_x: np.ndarray,
    vector_y: np.ndarray,
    vector_z: np.ndarray,
    mask_coords: List[Tuple[int, int, int]],
    interpolation_factor: float,
) -> np.ndarray:
    """
    Apply non-maximum suppression based on trilinear interpolation to
    enhance ridge structures in a 3D image.

    This function adjusts the influence of eigenvectors at each voxel
    using a given interpolation factor, and marks local maxima
    in the output array indicating significant ridge features.

    Parameters
    ----------
    image : np.ndarray
        The 3D image data array from which ridges are to be enhanced.
        Dimension: (Nx, Ny, Nz).
    vector_x : np.ndarray
        The x-component of the eigenvector associated with the largest
        eigenvalue at each voxel.
    vector_y : np.ndarray
        The y-component of the eigenvector.
    vector_z : np.ndarray
        The z-component of the eigenvector.
    mask_coords : List[Tuple[int, int, int]]
        A list of coordinates where the non-maximum suppression is to
        be applied.
    interpolation_factor : float
        A factor used in the trilinear interpolation for calculating
        the adjacent values.

    Returns
    -------
    np.ndarray
        A 3D binary array the same size as `image`, containing 1s at voxels
        identified as local maxima and 0s elsewhere.
    """
    # Initialize the output suppression matrix
    result = np.zeros_like(image)

    # Convert mask_coords to a structured NumPy array for efficient indexing
    coords = np.array(mask_coords, dtype=[("x", int), ("y", int), ("z", int)])
    x, y, z = coords["x"], coords["y"], coords["z"]

    # Compute normalized interpolation coefficients based on
    # the eigenvector components and interpolation factor
    dx = np.abs(vector_x[x, y, z] * interpolation_factor)
    dy = np.abs(vector_y[x, y, z] * interpolation_factor)
    dz = np.abs(vector_z[x, y, z] * interpolation_factor)

    # Calculate indices for forward and backward interpolation
    # based on the directionality of the eigenvector components
    next_x = x + np.sign(dx).astype(int)
    next_y = y + np.sign(dy).astype(int)
    next_z = z + np.sign(dz).astype(int)
    prev_x = x - np.sign(dx).astype(int)
    prev_y = y - np.sign(dy).astype(int)
    prev_z = z - np.sign(dz).astype(int)

    # Calculate trilinear interpolated values
    # for forward (+ve) and backward (-ve) directions
    interpolated_values_forward = (
        image[x, y, z] * (1 - dx) * (1 - dy) * (1 - dz)
        + image[next_x, y, z] * dx * (1 - dy) * (1 - dz)
        + image[x, next_y, z] * (1 - dx) * dy * (1 - dz)
        + image[x, y, next_z] * (1 - dx) * (1 - dy) * dz
        + image[next_x, next_y, z] * dx * dy * (1 - dz)
        + image[next_x, y, next_z] * dx * (1 - dy) * dz
        + image[x, next_y, next_z] * (1 - dx) * dy * dz
        + image[next_x, next_y, next_z] * dx * dy * dz
    )

    interpolated_values_backward = (
        image[x, y, z] * (1 - dx) * (1 - dy) * (1 - dz)
        + image[prev_x, y, z] * dx * (1 - dy) * (1 - dz)
        + image[x, prev_y, z] * (1 - dx) * dy * (1 - dz)
        + image[x, y, prev_z] * (1 - dx) * (1 - dy) * dz
        + image[prev_x, prev_y, z] * dx * dy * (1 - dz)
        + image[prev_x, y, prev_z] * dx * (1 - dy) * dz
        + image[x, prev_y, prev_z] * (1 - dx) * dy * dz
        + image[prev_x, prev_y, prev_z] * dx * dy * dz
    )

    # Local values from image at specified coordinates
    local_values = image[x, y, z]

    # Determine local maxima by comparing local values to interpolated values
    result[x, y, z] = (local_values > interpolated_values_forward) & (
        local_values > interpolated_values_backward
    )
    return result
