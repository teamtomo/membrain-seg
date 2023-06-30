import os

import numpy as np
from membrain_seg.dataloading.data_utils import (
    load_tomogram,
    make_directory_if_not_exists,
    write_nifti,
)


class InvalidCoordinatesError(Exception):
    """Exception raised of coordinates exceed tomogram boundaries."""

    pass


def pad_labels(patch, padding):
    """
    Pads labels of a 3D array, typically for the boundaries of a 3D patch.

    Parameters
    ----------
    patch : numpy.ndarray
        The input 3D array that needs to be padded.
    padding : tuple
        The tuple containing padding dimensions for the 3D array.
        It should contain three elements: (pad_depth, pad_height, pad_width).

    Returns
    -------
    patch : numpy.ndarray
        The padded 3D array.

    Notes
    -----
    This function pads the borders of the 3D array with the value of 2.0,
    typically used to ignore labels at the boundaries during a subsequent analysis.
    """
    patch[: padding[0], :, :] = 2.0
    patch[-padding[0] :, :, :] = 2.0
    patch[:, : padding[1], :] = 2.0
    patch[:, -padding[1] :, :] = 2.0
    patch[:, :, : padding[2]] = 2.0
    patch[:, :, -padding[2] :] = 2.0
    return patch


def get_out_files_and_patch_number(
    token, out_folder_raw, out_folder_lab, patch_nr, idx_add
):
    """
    Create filenames and corrected patch numbers.

    Generates unique file names for raw and labeled data patches by incrementing patch
    number until non-existing file names are found in the specified directories.
    #Also returns the final patch number used to generate these file names.

    Parameters
    ----------
    token : str
        The unique identifier used as a part of the filename.
    out_folder_raw : str
        The directory path where raw data patches are stored.
    out_folder_lab : str
        The directory path where labeled data patches are stored.
    patch_nr : int
        The initial patch number to be used for generating file names.
    idx_add : int
        The number to be added to the initial patch number to generate unique
        file names.

    Returns
    -------
    int
        The final patch number used to generate unique file names.
    str
        The full file path for the raw data patch.
    str
        The full file path for the labeled data patch.

    Notes
    -----
    The function generates filenames in the format <token>_patch<number>_raw.nii.gz
    for raw data patches and <token>_patch<number>_labels.nii.gz for labeled data
    patches.
    If a file with the same name already exists in the specified directories,
    the function increments the patch number until it finds a unique filename.
    The final patch number is obtained by adding the initial patch number and
    `idx_add` and then incrementing it further if needed.

    """
    patch_nr += idx_add
    out_file_patch = os.path.join(
        out_folder_raw, token + "_patch" + str(patch_nr) + "_raw.nii.gz"
    )
    out_file_patch_label = os.path.join(
        out_folder_lab, token + "_patch" + str(patch_nr) + "_labels.nii.gz"
    )
    exist_add = 0
    while os.path.isfile(out_file_patch):
        exist_add += 1
        out_file_patch = os.path.join(
            out_folder_raw,
            token + "_patch" + str(patch_nr + exist_add) + "_raw.nii.gz",
        )
        out_file_patch_label = os.path.join(
            out_folder_lab,
            token + "_patch" + str(patch_nr + exist_add) + "_labels.nii.gz",
        )
    return patch_nr + exist_add, out_file_patch, out_file_patch_label


def extract_patches(tomo_path, seg_path, coords, out_dir, idx_add=0, token=None):
    """
    Extracts 3D patches from a given tomogram and corresponding segmentation.

    The patches are then saved to the specified output directory.

    Parameters
    ----------
    tomo_path : str
        Path to the input tomogram file from which patches will be extracted.
    seg_path : str
        Path to the corresponding segmentation file.
    coords : list
        List of tuples where each tuple represents the 3D coordinates of a patch center.
    out_dir : str
        The output directory where the extracted patches will be saved.
    idx_add : int, optional
        The index addition for patch numbering, default is 0.
    token : str, optional
        Token to uniquely identify the tomogram, default is None. If None,
        the base name of the tomogram file path is used.

    Returns
    -------
    None

    Notes
    -----
    Patches are saved in the 'imagesCorr' and 'labelsCorr' subdirectories of the
    output directory.
    The patch from the tomogram and the corresponding patch from the segmentation
    are saved with the same name for easy correspondence.

    Exceptions
    ----------
    Raises InvalidCoordinatesError if a patch cannot be extracted due to the provided
    coordinates being too close to the border of the tomogram.
    """
    padding = (16, 16, 16)
    if token is None:
        token = os.path.splitext(os.path.basename(tomo_path))[0]
    out_folder_raw = os.path.join(out_dir, "imagesCorr")
    out_folder_lab = os.path.join(out_dir, "labelsCorr")
    make_directory_if_not_exists(out_folder_raw)
    make_directory_if_not_exists(out_folder_lab)

    tomo = load_tomogram(tomo_path)
    labels = load_tomogram(seg_path)

    for patch_nr, cur_coords in enumerate(coords):
        patch_nr, out_file_patch, out_file_patch_label = get_out_files_and_patch_number(
            token, out_folder_raw, out_folder_lab, patch_nr, idx_add
        )
        print("Extracting patch nr", patch_nr, "from tomo", token)
        try:
            min_coords = np.array(cur_coords) - 80
            if not (
                np.all(min_coords) >= 0
                and np.all(min_coords < (np.array(tomo.shape) - 160))
            ):
                raise InvalidCoordinatesError(
                    "Invalid coordinates: Some values are outside the allowed range.\
                Your selected center points should be at least 80 voxels away from \
                the tomogram borders."
                )
            cur_patch = tomo[
                min_coords[0] : min_coords[0] + 160,
                min_coords[1] : min_coords[1] + 160,
                min_coords[2] : min_coords[2] + 160,
            ]
            cur_patch_labels = labels[
                min_coords[0] : min_coords[0] + 160,
                min_coords[1] : min_coords[1] + 160,
                min_coords[2] : min_coords[2] + 160,
            ]
            cur_patch_labels = pad_labels(cur_patch_labels, padding)

            cur_patch = np.transpose(cur_patch, (2, 1, 0))
            cur_patch_labels = np.transpose(cur_patch_labels, (2, 1, 0))
            write_nifti(out_file_patch, cur_patch)
            write_nifti(out_file_patch_label, cur_patch_labels)

        except InvalidCoordinatesError as e:
            print("Error:", str(e))
