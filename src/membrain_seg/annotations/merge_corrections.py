import os

import numpy as np
import SimpleITK as sitk

from membrain_seg.segmentation.dataloading.data_utils import write_nifti

from .extract_patches import pad_labels


def get_corrections_from_folder(folder_name, orig_pred_file):
    """
    Gather all corrections into one array.

    Extract correction data from a specified folder and applies these corrections to
    a given prediction.

    Parameters
    ----------
    folder_name : str
        Path to the folder containing correction files.
    orig_pred_file : str
        Path to the original prediction file that the corrections will be applied to.

    Returns
    -------
    merged : numpy.ndarray
        The corrected prediction with 'Add', 'Remove' and 'Ignore' corrections applied.

    Notes
    -----
    This function expects the correction files to be named with prefixes
    'Add', 'Remove', or 'Ignore'.
    """
    orig_pred = sitk.GetArrayFromImage(sitk.ReadImage(orig_pred_file))
    add_patch = np.zeros(orig_pred.shape)
    remove_patch = np.zeros(orig_pred.shape)
    ignore_patch = np.zeros(orig_pred.shape)
    correction_count = 0
    for filename in os.listdir(folder_name):
        if not (
            filename.startswith("Add")
            or filename.startswith("add")
            or filename.startswith("Remove")
            or filename.startswith("remove")
            or filename.startswith("Ignore")
            or filename.startswith("ignore")
        ):
            print(
                "File does not fit into Add/Remove/Ignore naming! " "Not processing",
                filename,
            )
            continue
        readdata = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(folder_name, filename))
        )
        print("Adding file", filename)

        if filename.startswith("Add") or filename.startswith("add"):
            add_patch += readdata
        if filename.startswith("Remove") or filename.startswith("remove"):
            remove_patch += readdata
        if filename.startswith("Ignore") or filename.startswith("ignore"):
            ignore_patch += readdata
        correction_count += 1

    if correction_count == 0:
        print("WARNING: We have not found any corrections in folder", folder_name)
        print(
            "We have still created a merged correction, though. Please check carefully!"
        )
    add_patch = (add_patch > 0.0) * 1
    remove_patch = (remove_patch > 0.0) * 1
    ignore_patch = (ignore_patch > 0.0) * 1

    merged = orig_pred
    merged = np.where(remove_patch.clip(min=0, max=1) == 0, merged, 0)
    merged = np.where(add_patch.clip(min=0, max=1) == 0, merged, 1)
    merged = np.where(ignore_patch.clip(min=0, max=1) == 0, merged, 2)
    merged = pad_labels(merged, padding=16)
    return merged


def convert_single_nrrd_files(labels_dir, corrections_dir, out_dir):
    """
    Merge labels from .nrrd files.

    Process all label files in a given directory, find corresponding correction folders,
    apply corrections and write corrected files to an output directory.

    Parameters
    ----------
    labels_dir : str
        Path to the directory containing label files.
    corrections_dir : str
        Path to the directory containing correction folders. Each correction folder
        should have the
        same name as the corresponding label file (without the file extension).
    out_dir : str
        Path to the directory where corrected label files will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function expects each label file to have a corresponding folder in
    corrections_dir.
    If no such folder is found, a warning will be printed.
    """
    for label_file in os.listdir(labels_dir):
        if not os.path.isfile(os.path.join(labels_dir, label_file)):
            continue
        print("")
        print("Finding correction files for", label_file)
        # token = os.path.splitext(label_file)[0]
        if label_file.endswith(".nii.gz"):
            token = label_file[:-7]
        elif label_file.endswith(".nrrd"):
            token = label_file[:-5]
        elif label_file.endswith(".mrc"):
            token = label_file[:-4]
        found_flag = 0
        for filename in os.listdir(corrections_dir):
            if not os.path.isdir(os.path.join(corrections_dir, filename)):
                continue
            if filename == token:
                cur_patch_corrections_folder = os.path.join(corrections_dir, filename)
                merged_corrections = get_corrections_from_folder(
                    cur_patch_corrections_folder, os.path.join(labels_dir, label_file)
                )
                out_file = os.path.join(out_dir, token + ".nii.gz")
                print("Storing corrections in", out_file)
                write_nifti(out_file, merged_corrections)
                found_flag = 1
        if found_flag == 0:
            print("No corrections folder found for patch", token)
