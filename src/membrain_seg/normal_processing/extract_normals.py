import os
from typing import Tuple

import numpy as np

from membrain_seg.normal_processing.mesh_utils import (
    compute_normals_for_mesh,
    convert_file_to_mesh,
    find_nearest_normals,
)
from membrain_seg.segmentation.dataloading.data_utils import (
    read_nifti,
    write_nifti,
)

task_dir = (
    "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-seg-normalAugs/"
    "Task143_cryoET7"
)
# task_dir = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBra\
#     in-seg-normalAugs/Task529_ChlamyV3_HDCR"


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


def extract_normals(
    task_dir: str, min_dist_thres: float = 3.0, decimation_degree: float = 0.5
) -> None:
    """
    Extract normal vectors from segmentation files in a given task directory.

    Processes each .nii.gz file in the task's training and validation directories,
    computes normals, and saves them in a specified output directory. Assumes the
    presence of `read_nifti`, `convert_file_to_mesh`, `compute_normals_for_mesh`,
    `compute_meshgrid`, `find_nearest_normals`, and `write_nifti` functions.

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


if __name__ == "__main__":
    extract_normals(task_dir)
