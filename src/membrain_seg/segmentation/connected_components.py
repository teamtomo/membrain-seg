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
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled_array, num_features = ndimage.label(binary_seg, structure=structure)
    print(f"Found {num_features} connected components.")
    if size_thres is not None and size_thres > 1:
        sizes = np.bincount(labeled_array.ravel())
        print(f"Removing components smaller than {size_thres} voxels.")
        # keep only components above threshold
        keep_mask = np.isin(labeled_array, np.where(sizes >= size_thres)[0])
        labeled_array = labeled_array * keep_mask
        # relabel to make them contiguous 1..N
        labeled_array, _, _ = relabel_sequential(labeled_array)

    return labeled_array


def relabel_sequential(labeled_array):
    """Make labels contiguous 1..N."""
    unique = np.unique(labeled_array)
    unique = unique[unique != 0]  # skip background
    mapping = {old: new for new, old in enumerate(unique, 1)}
    relabeled = np.zeros_like(labeled_array)
    print(f"Relabeled to {len(unique)} connected components.")
    for old, new in mapping.items():
        relabeled[labeled_array == old] = new
    return relabeled, mapping, len(unique)
