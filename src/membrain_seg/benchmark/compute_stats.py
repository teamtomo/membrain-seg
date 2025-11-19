"""
Compute statistics for segmentations on the entire dataset.

Workflow:
1. Load predictions and ground truth segmentations from two separate directories.  
    Filenames (except for extension) must match. (predictions can be .nii.gz or .mrc)
2. Skeletonize all segmentations.
3. Compute both dice and surface-dice scores for each pair of predictions and ground 
    truth segmentations.
4. Compute also global dice and surface-dice by aggregating all segmentations / 
    skeletons.
"""

import csv
import os

import numpy as np
from tqdm import tqdm

from membrain_seg.benchmark.metrics import masked_dice, masked_surface_dice
from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    read_nifti,
)
from membrain_seg.segmentation.skeletonize import skeletonization, skeletonize_skimage

ds_dict = {}


def reset_ds_dict():
    """Reset the dataset dictionary."""
    global ds_dict
    ds_dict = {}


def get_filepaths(dir_gt: str, dir_pred: str):
    """
    Get filepaths for all ground truth segmentations and predictions.

    Parameters
    ----------
    dir_gt : str
        Directory containing ground truth segmentations.
    dir_pred : str
        Directory containing predictions.

    Returns
    -------
    gt_files : list
        List of ground truth segmentation filepaths.
    pred_files : list
        List of prediction filepaths.
    """
    # Load all segmentations and skeletons
    gt_files = os.listdir(dir_gt)
    # take all file with .nii.gz extension
    gt_files = [f for f in gt_files if f.endswith(".nii.gz")]

    # check whether predictions are in .mrc or .nii.gz format
    is_mrc = False
    pred_files = os.listdir(dir_pred)
    pred_files_mrc = [f for f in pred_files if f.endswith(".mrc")]
    pred_files_nii = [f for f in pred_files if f.endswith(".nii.gz")]
    if len(pred_files_mrc) > 0:
        pred_files = pred_files_mrc
        is_mrc = True
    elif len(pred_files_nii) > 0:
        pred_files = pred_files_nii
    else:
        raise ValueError("No predictions found in .mrc or .nii.gz format.")

    # check whether the number of predictions and ground truth segmentations match
    if len(gt_files) != len(pred_files):
        raise ValueError(
            "Number of ground truth segmentations and predictions do not match."
        )

    # sort all files alphabetically
    gt_files.sort()
    pred_files.sort()

    # for each gt file, find the corresponding pred file
    filtered_gt_files = []
    filtered_pred_files = []
    for i in range(len(gt_files)):
        gt_token = gt_files[i][:-7]  # remove .nii.gz
        if is_mrc:
            pred_file = gt_token + ".mrc"
        else:
            pred_file = gt_token + ".nii.gz"
        if pred_file not in pred_files:
            # only do a warning here
            print(f"Warning: No prediction found for {gt_files[i]}. Skipping.")
            continue
        filtered_gt_files.append(gt_files[i])
        filtered_pred_files.append(pred_file)

    print(len(filtered_gt_files), "matching files found.")
    print(
        len(gt_files) - len(filtered_gt_files),
        "files skipped due to missing predictions.",
    )

    # make sure gt_files and pred_files are the same before the extension
    for gt, pred in zip(filtered_gt_files, filtered_pred_files):
        if is_mrc:
            assert gt[:-7] == pred[:-4]
        else:
            assert gt[:-7] == pred[:-7]
    return filtered_gt_files, filtered_pred_files


def read_nifti_or_mrc(file_path: str):
    """Read a nifti or mrc file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    np.ndarray
        The data.
    """
    if file_path.endswith(".mrc"):
        return load_tomogram(file_path).data
    else:
        return read_nifti(file_path)


def get_ds_token(filename):
    """Get the dataset token from the filename.

    Parameters
    ----------
    filename : str
        The filename of the patch.

    Returns
    -------
    str
        The dataset token.
    """
    token = filename.split("_")[0]
    if token in ["atty", "benedikt", "rory", "virly"]:
        return "collaborators"
    elif token in ["cts", "polnet"]:
        return "synthetic"
    else:
        return token


def initialize_ds_dict_entry(ds_token):
    """Initialize a dataset dictionary entry.

    Parameters
    ----------
    ds_token : str
        The dataset token.
    """
    if ds_token not in ds_dict:
        ds_dict[ds_token] = {
            "surf_dice": [],
            "tp_pred_sdice": 0,
            "tp_gt_sdice": 0,
            "all_pred_sdice": 0,
            "all_gt_sdice": 0,
            "dice": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }


def update_ds_dict_entry(ds_token, surf_dice, confusion_dict, dice, dice_dict):
    """Update the dataset dictionary entry.

    Parameters
    ----------
    ds_token : str
        The dataset token.
    surf_dice : float
        Surface dice score.
    confusion_dict : dict
        Dictionary containing the following
        keys:
        - tp_pred: True positives in the prediction.
        - tp_gt: True positives in the ground truth.
        - all_pred: All positives in the prediction.
        - all_gt: All positives in the ground truth.
    dice : float
        Dice score.
    dice_dict : dict
        Dictionary containing the following
        keys:
        - tp: True positives.
        - fp: False positives.
        - fn: False negatives.

    """
    ds_dict[ds_token]["surf_dice"].append(surf_dice)
    ds_dict[ds_token]["tp_pred_sdice"] += confusion_dict["tp_pred"]
    ds_dict[ds_token]["tp_gt_sdice"] += confusion_dict["tp_gt"]
    ds_dict[ds_token]["all_pred_sdice"] += confusion_dict["all_pred"]
    ds_dict[ds_token]["all_gt_sdice"] += confusion_dict["all_gt"]
    ds_dict[ds_token]["dice"].append(dice)
    ds_dict[ds_token]["tp"] += dice_dict["tp"]
    ds_dict[ds_token]["fp"] += dice_dict["fp"]
    ds_dict[ds_token]["fn"] += dice_dict["fn"]


def print_ds_dict():
    """Print the dataset dictionary."""
    print("")
    print("Dataset statistics:")
    for ds_token in ds_dict:
        print(f"Dataset: {ds_token}")
        print(f"Surface dice: {np.mean(ds_dict[ds_token]['surf_dice'])}")
        print(f"Global surface dice: {get_global_stats(ds_token, s_dice=True)}")
        print(f"Dice: {np.mean(ds_dict[ds_token]['dice'])}")
        print(f"Global dice: {get_global_stats(ds_token, s_dice=False)}")
        print("")


def get_global_stats(
    ds_token,
    s_dice: bool,
):
    """Aggregates global statistics for a dataset.

    Parameters
    ----------
    ds_token : str
        The dataset token.
    s_dice : bool
        Whether to compute surface dice or dice.

    Returns
    -------
    float
        The global statistic.
    """
    if s_dice:
        global_dice = (
            2.0
            * (
                ds_dict[ds_token]["tp_pred_sdice"]
                / (ds_dict[ds_token]["all_pred_sdice"] + 1e-6)
            )
            * (
                ds_dict[ds_token]["tp_gt_sdice"]
                / (ds_dict[ds_token]["all_gt_sdice"] + 1e-6)
            )
            / (
                ds_dict[ds_token]["tp_pred_sdice"]
                / (ds_dict[ds_token]["all_pred_sdice"] + 1e-6)
                + ds_dict[ds_token]["tp_gt_sdice"]
                / (ds_dict[ds_token]["all_gt_sdice"] + 1e-6)
                + 1e-6
            )
        )
    else:
        global_dice = (
            2.0
            * (
                ds_dict[ds_token]["tp"]
                / (ds_dict[ds_token]["tp"] + ds_dict[ds_token]["fp"] + 1e-6)
            )
            * (
                ds_dict[ds_token]["tp"]
                / (ds_dict[ds_token]["tp"] + ds_dict[ds_token]["fn"] + 1e-6)
            )
            / (
                ds_dict[ds_token]["tp"]
                / (ds_dict[ds_token]["tp"] + ds_dict[ds_token]["fp"] + 1e-6)
                + ds_dict[ds_token]["tp"]
                / (ds_dict[ds_token]["tp"] + ds_dict[ds_token]["fn"] + 1e-6)
                + 1e-6
            )
        )
    return global_dice


def store_stats(out_file):
    """Store the dataset dictionary in a csv file.

    Parameters
    ----------
    out_file : str
        Path to the output file.
    """
    # store ds_dict in a csv file
    header = [
        "Dataset",
        "Surface Dice",
        "Global Surface Dice",
        "Dice",
        "Global Dice",
    ]
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ds_token in ds_dict:
            row = [
                ds_token,
                np.mean(ds_dict[ds_token]["surf_dice"]),
                get_global_stats(ds_token, s_dice=True),
                np.mean(ds_dict[ds_token]["dice"]),
                get_global_stats(ds_token, s_dice=False),
            ]
            writer.writerow(row)


def compute_stats(
    dir_gt: str,
    dir_pred: str,
    out_dir: str,
    out_file_token: str = "stats",
    skeletonization_method: str = "3D-NMS",
):
    """
    Compute statistics for segmentations on the entire dataset.

    Parameters
    ----------
    dir_gt : str
        Directory containing ground truth segmentations.
    dir_pred : str
        Directory containing predictions.
    out_dir : str
        Directory to save the results.
    out_file_token : str
        Token to append to the output file.
    skeletonization_method : str
        Skeletonization method to use. Supported: "3D-NMS", "2D-skimage".
    """
    reset_ds_dict()
    gt_files, pred_files = get_filepaths(dir_gt, dir_pred)

    length = len(gt_files)
    for i in tqdm(range(length)):
        gt_file = gt_files[i]
        pred_file = pred_files[i]

        ds_token = get_ds_token(gt_file)
        initialize_ds_dict_entry(ds_token)

        # Load ground truth and prediction
        gt = read_nifti_or_mrc(os.path.join(dir_gt, gt_file))
        pred = read_nifti_or_mrc(os.path.join(dir_pred, pred_file))

        # Skeletonize both segmentations
        if skeletonization_method == "3D-NMS":
            gt_skeleton = skeletonization(gt == 1, batch_size=100000)
            pred_skeleton = skeletonization(pred, batch_size=100000)
        elif skeletonization_method == "2D-skimage":
            gt_skeleton = skeletonize_skimage(gt == 1)
            pred_skeleton = skeletonize_skimage(pred)
        else:
            raise ValueError(
                f"Unsupported skeletonization method: {skeletonization_method}"
            )
        mask = gt != 2

        # Compute surface dice
        surf_dice, confusion_dict = masked_surface_dice(
            pred_skeleton, gt_skeleton, pred, gt, mask
        )
        dice, dice_dict = masked_dice(pred, gt, mask)
        print(
            "Iteration:",
            i,
            "Surface dice:",
            surf_dice,
            "Dice:",
            dice,
            os.path.basename(gt_file),
        )
        update_ds_dict_entry(ds_token, surf_dice, confusion_dict, dice, dice_dict)
    print_ds_dict()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{out_file_token}.csv")
    store_stats(out_file)
    return out_file
