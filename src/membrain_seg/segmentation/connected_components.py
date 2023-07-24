import numpy as np
from scipy import ndimage


def connected_components(binary_seg: np.ndarray, size_thres: int = None):
    """
    Compute connected components from a 3D membrane segmentation.

    The function uses 3D connectivity for label assignments. If a size
    threshold is specified, components having size smaller than the
    threshold will be removed.


    Parameters
    ----------
    binary_seg : np.ndarray
        Input 3D binary image with shape (X, Y, Z).
        Non-zero elements are considered as the foreground.

    size_thres : int, optional
        The size threshold for removing small connected components.
        Components having size (number of voxels) smaller than this
        value will be removed.
        If None (default), no components are removed.

    Returns
    -------
    labeled_array : np.ndarray
        A 3D image where each voxel in a connected component has a
        unique label. The labels are in the range 1 to N, where N is
        the number of connected components found. All background voxels
        are zero.
    """
    print("Computing connected components.")
    # Get 3D connected components
    structure = np.ones((3, 3, 3))
    labeled_array, num_features = ndimage.label(binary_seg, structure=structure)

    # remove small clusters
    if size_thres is not None:
        print("Removing components smaller than", size_thres, "voxels.")
        sizes = ndimage.sum(binary_seg, labeled_array, range(1, num_features + 1))
        too_small = np.nonzero(sizes < size_thres)[0] + 1  # features labeled from 1
        for feat_nr in too_small[::-1]:  # iterate in reverse order
            labeled_array[labeled_array == feat_nr] = 0
            labeled_array[labeled_array > feat_nr] -= 1
    return labeled_array
