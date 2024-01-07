import os
from typing import Tuple

import numpy as np

from membrain_seg.normal_processing.mesh_utils import (
    compute_normals_for_mesh,
    convert_file_to_mesh,
    find_nearest_normals,
)
from membrain_seg.segmentation.dataloading.data_utils import (
    get_csv_data,
    load_tomogram,
    read_nifti,
    store_array_in_csv,
    store_point_and_vectors_in_vtp,
    write_nifti,
)

from .euler_utils import compute_Euler_angles_for_normals


def compute_meshgrid(shape: Tuple[int]) -> np.ndarray:
    """
    Compute meshgrid arrays for given shape.

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the volume for which to compute the meshgrid.

    Returns
    -------
    np.ndarray
        The concatenated meshgrid arrays.
    """
    coords = [np.arange(s) for s in shape]
    grids = np.meshgrid(*coords)
    grids = [np.expand_dims(grid, -1) for grid in grids]
    return np.concatenate(grids, axis=-1).reshape(-1, 3)


def get_in_input_dir(task_dir: str, train_val_token: str) -> Tuple[str, str]:
    """
    Get the input directories for training and validation data.

    Parameters
    ----------
    task_dir : str
        The base directory of the task containing label directories.
    train_val_token : str
        A token indicating whether to get directories for "Tr" (Training) or
        "Val" (Validation) data.

    Returns
    -------
    Tuple[str, str]
        A tuple containing paths to the ground truth directory and the vector
        directory.
    """
    if train_val_token == "Tr":
        gt_dir_nifti = os.path.join(task_dir, "labelsTr")
        vecs_dir = os.path.join(task_dir, "labelsTr_vecs")
    elif train_val_token == "Val":
        gt_dir_nifti = os.path.join(task_dir, "labelsVal")
        vecs_dir = os.path.join(task_dir, "labelsVal_vecs")
    os.makedirs(vecs_dir, exist_ok=True)
    return gt_dir_nifti, vecs_dir


def match_coords_to_membrane_normals(
    coords_file: str,
    out_coords_file: str,
    membrane_seg_path: str = None,
    membrane_normals_path: str = None,
    euler_conversion: str = "zxz",
    min_dist_thres: float = 3.0,
    smoothing: int = 2000,
    decimation_degree: float = 0.8,
    coords_unit: str = "voxels",  # or "angstroms"
) -> None:
    """
    Matches coordinates from a CSV file to membrane normals and stores results.

    Parameters
    ----------
    coords_file : str
        Path to the input CSV file containing coordinates.
    out_coords_file : str
        Path to output CSV file where the matched coordinates and normals will
        be stored.
    membrane_seg_path : str, optional
        Path to the membrane segmentation file, which will be converted to a
        mesh to compute normals.
    membrane_normals_path : str, optional
        Path to the precomputed membrane normals file.
    euler_conversion : str, optional
        The Euler angle conversion convention to use ('zxz' or 'zyz').
        Default is 'zxz'.
    min_dist_thres : float, optional
        The maximum distance threshold for considering nearest normals.
    smoothing : int, optional
        The number of smoothing iterations to apply to the resulting mesh.
        Default is 2000.
    decimation_degree : float, optional
        The degree of decimation to reduce the mesh size. Value should be
        between 0 and 1. Default is 0.8.
    coords_unit : str, optional
        The unit of the coordinates in the input CSV file. Default is 'voxels'.


    Notes
    -----
    - At least one of membrane_seg_path or membrane_normals_path must be
        provided.
    - Normals can be computed from a mesh generated from the segmentation
        or loaded from a precomputed file (e.g. MemBrain-seg output).
    - If euler_conversion is specified, Euler angles for the normals are
        computed and appended to the output.

    Raises
    ------
    AssertionError
        If neither membrane_seg_path nor membrane_normals_path is provided.
    """
    assert membrane_seg_path is not None or membrane_normals_path is not None
    if euler_conversion is not None and euler_conversion not in ["zxz", "zyz"]:
        raise OSError("Convention not known.")

    points = get_csv_data(coords_file)

    if membrane_normals_path is None:
        print("Computing normals from segmentation file." " This may take a while...")
        surf = convert_file_to_mesh(
            membrane_seg_path,
            smoothing=smoothing,
            decimation_degree=decimation_degree,
            angstrom_verts=coords_unit == "angstroms",
        )
        if not surf:
            raise ValueError("No membrane found in", membrane_seg_path, "Aborting.")

        print("Computing normals.")
        centers_with_norms = compute_normals_for_mesh(surf)
        coords, normals = centers_with_norms[:, :3], centers_with_norms[:, 3:]
    else:
        print("Loading membrane segmentation from", membrane_seg_path)
        seg = load_tomogram(membrane_seg_path)
        seg = seg.data
        membrane_normals_paths = [
            membrane_normals_path[:-5] + f"{i}.mrc" for i in range(3)
        ]
        print("Loading normals from", *membrane_normals_paths)
        normals_arrays = [
            load_tomogram(membrane_normals_path).data
            for membrane_normals_path in membrane_normals_paths
        ]
        normals_array = np.stack(normals_arrays, axis=-1)
        coords = np.argwhere(seg != 0)

        normals = normals_array[coords[:, 0], coords[:, 1], coords[:, 2]]
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    print("Matching coordinates to nearest normals.")
    norms = find_nearest_normals(
        coords=coords,
        normals=normals,
        points=points,
        threshold=min_dist_thres,
        remove_masked=False,
    )
    centers_with_norms = np.concatenate((points, norms), axis=1)

    if euler_conversion is not None:
        print("Computing Euler angles.")
        euler_angles = compute_Euler_angles_for_normals(
            points=centers_with_norms[:, :3],
            normals=centers_with_norms[:, 3:],
            convention=euler_conversion,
        )
        centers_with_norms = np.concatenate((centers_with_norms, euler_angles), axis=1)
    store_array_in_csv(out_coords_file, centers_with_norms)
    centers_with_norms = np.array(centers_with_norms, dtype=float)
    store_point_and_vectors_in_vtp(
        out_path=out_coords_file[:-3] + "vtp",
        in_points=centers_with_norms[:, :3],
        in_vectors=centers_with_norms[:, 3:6],
    )


def extract_normals_GT(
    task_dir: str, min_dist_thres: float = 3.0, decimation_degree: float = 0.5
) -> None:
    """
    Extract normal vectors from segmentation files in a given task directory.

    Processes each .nii.gz file in the task's training and validation directories,
    computes normals, and saves them in a specified output directory.

    Parameters
    ----------
    task_dir : str
        The base directory of the task containing segmentation files.
    min_dist_thres : float, optional
        The minimum distance threshold for considering nearest normals.
        Default is 3.0.
    decimation_degree : float, optional
        The degree of decimation to reduce the mesh size.
        Value should be between 0 and 1. Default is 0.5.

    Returns
    -------
    None
    """
    for train_val_token in ["Tr", "Val"]:
        gt_dir_nifti, vecs_dir = get_in_input_dir(task_dir, train_val_token)

        for filetoken in os.listdir(gt_dir_nifti):
            # Assume that the filename ends with .nii.gz
            filetoken = filetoken[:-7]
            print("Processing", filetoken)
            seg_file = os.path.join(gt_dir_nifti, filetoken + ".nii.gz")
            out_vec_file = os.path.join(vecs_dir, filetoken + "_norms.nii.gz")
            seg = read_nifti(seg_file)
            surf = convert_file_to_mesh(seg_file, decimation_degree=decimation_degree)

            if surf is False:
                print(
                    "Did not find any membrane in", seg_file, ". Setting normals to 0."
                )
                normal_labels = np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3))
            else:
                centers_with_norms = compute_normals_for_mesh(surf)

                # Split centers and normals
                coords, normals = centers_with_norms[:, :3], centers_with_norms[:, 3:]

                # Read NIfTI file and create ones mask
                meshgrid_points = compute_meshgrid(seg.shape)

                # Find and reshape nearest normals
                normal_labels = find_nearest_normals(
                    coords, normals, meshgrid_points, min_dist_thres
                )
                normal_labels = normal_labels.reshape(
                    seg.shape[0], seg.shape[1], seg.shape[2], 3
                )
            for k in range(3):
                normal_labels[:, :, :, k] = np.transpose(
                    normal_labels[:, :, :, k], (2, 1, 0)
                )
            write_nifti(out_vec_file, normal_labels)
