import numpy as np
from membrain_seg.segmentation.skeletonization.nonmaxsup_kernel import nonmaxsup_kernel

def nonmaxsup(eigenvalues: np.ndarray, eigenvector_x: np.ndarray, eigenvector_y: np.ndarray, eigenvector_z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Perform non-maximum suppression on the given tomogram to detect ridge centrelines.

    This function applies a non-maximum suppression algorithm to identify and enhance
    ridge-like structures in volumetric data based on eigenvalues and the major eigenvector's
    components. The process involves masking with the input labels to focus only on regions
    of interest and suppressing non-ridge areas.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues of the Hessian matrix, used to identify potential ridge points.
    eigenvector_x : np.ndarray
        X-component of the principal eigenvector associated with the largest eigenvalue.
    eigenvector_y : np.ndarray
        Y-component of the principal eigenvector.
    eigenvector_z : np.ndarray
        Z-component of the principal eigenvector.
    labels : np.ndarray
        A binary mask where 1 indicates regions of interest and 0 indicates background.

    Returns
    -------
    np.ndarray
        A binary array where 1 indicates detected ridge centreline and 0 indicates background.

    Notes
    -----
    The non-maximum suppression is focused within the regions specified by the labels.
    The algorithm leverages an interpolation factor to adjust the suppression sensitivity
    and is implemented through a specific kernel function for efficient processing.
    """
    # Define the boundary for processing to avoid edge effects
    Nx, Ny, Nz = eigenvalues.shape
    margin = 1
    mask = np.zeros((Nx, Ny, Nz))
    mask[margin : Nx - margin, margin : Ny - margin, margin : Nz - margin] = 1
    masked_labels = labels * mask

    # Filter coordinates where suppression is applicable
    relevant_coords = np.where(masked_labels == 1)
    coordinates_list = list(zip(relevant_coords[0], relevant_coords[1], relevant_coords[2]))

    # Define interpolation factor for kernel processing
    interpolation_factor = 0.71
    binary_output = nonmaxsup_kernel(eigenvalues, eigenvector_x, eigenvector_y, eigenvector_z, coordinates_list, interpolation_factor)
    
    return binary_output
