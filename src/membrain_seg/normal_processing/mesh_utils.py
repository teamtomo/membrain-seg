import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from skimage import measure

from membrain_seg.segmentation.dataloading.data_utils import (
    read_nifti,
)


def convert_seg_to_mesh(seg: np.ndarray, smoothing: int) -> pv.PolyData:
    """
    Convert a segmentation array to a mesh using marching cubes.

    Parameters
    ----------
    seg : np.ndarray
        The segmentation array.
    smoothing : int
        The number of smoothing iterations to apply to the mesh.


    Returns
    -------
    pv.PolyData
        The resulting mesh.
    """
    verts, faces, _, _ = measure.marching_cubes(
        seg, 0.5, step_size=1.5, method="lewiner"
    )
    all_col = np.ones((faces.shape[0], 1), dtype=int) * 3  # Prepend 3 for vtk format
    faces = np.concatenate((all_col, faces), axis=1)
    surf = pv.PolyData(verts, faces)
    return surf.smooth(n_iter=smoothing)


def find_nearest_normals(
    coords: np.ndarray, normals: np.ndarray, points: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Find nearest normals for each point in a set, under a distance threshold.

    Parameters
    ----------
    coords : np.ndarray
        The array of coordinates.
    normals : np.ndarray
        The array of normals associated with coordinates.
    points : np.ndarray
        The array of points to find nearest normals for.
    threshold : float
        The distance threshold for considering nearest normals.

    Returns
    -------
    np.ndarray
        The array of normals corresponding to the nearest points under
        the threshold.
    """
    # Create a cKDTree for fast nearest neighbor queries
    tree = cKDTree(coords)
    # Query the tree for nearest neighbors within the threshold
    distances, indices = tree.query(points, k=1)
    # Select the normals for the nearest neighbors
    nearest_normals = normals[indices]
    # Mask out invalid indices and distances beyond the threshold
    mask = (indices != len(coords)) & (distances < threshold)
    normal_labels = np.zeros(points.shape)
    normal_labels[mask] = nearest_normals[mask]
    return normal_labels


def compute_normals_for_mesh(surf: pv.PolyData) -> np.ndarray:
    """
    Compute and return the normal vectors for each cell in the mesh.

    Parameters
    ----------
    surf : pv.PolyData
        The input mesh for which normals are computed.

    Returns
    -------
    np.ndarray
        Array of center coordinates concatenated with normal vectors.
    """
    surf_normals = surf.compute_normals(flip_normals=True)
    normals = surf_normals.get_array("Normals")
    centers = surf.cell_centers().points.copy()
    return np.concatenate((centers, normals), axis=1)


def convert_file_to_mesh(
    seg_file: str,
    out_file: str = None,
    smoothing: int = 2000,
    decimation_degree: float = 0.90,
) -> tuple:
    """
    Convert an .nii.gz segmentation file to a mesh file.

    Parameters
    ----------
    seg_file : str
        The path to the input segmentation file.
    out_file : str, optional
        The path where the output mesh file will be saved.
        If not provided, the file is not saved.
    smoothing : int, optional
        The number of smoothing iterations to apply to the resulting mesh.
        Default is 2000.
    decimation_degree : float, optional
        The degree of decimation to reduce the mesh size. Value should
        be between 0 and 1. Default is 0.90.

    Returns
    -------
    tuple
        A tuple containing the mesh (pyvista.PolyData) as the first element.
        If return_normal_array is True, the second element will be a numpy
        array of center coordinates concatenated with normal vectors.


    Raises
    ------
    ValueError
        If no membrane (value 1.0) is found in the segmentation file.
    """
    try:
        seg = read_nifti(seg_file)  # Assuming read_nifti is defined elsewhere
        seg = np.transpose(seg, (1, 2, 0))
        seg = (seg == 1.0) * 1.0  # Binary segmentation for membrane

        if 1.0 not in np.unique(seg):
            raise ValueError(f"No membrane found in {seg_file}.")

    except Exception as e:
        print(f"Error processing segmentation file: {e}")
        return False

    surf = convert_seg_to_mesh(seg, smoothing)

    if decimation_degree is not None:
        surf = surf.decimate(decimation_degree)

    if out_file is not None:
        surf.save(out_file)

    return surf
