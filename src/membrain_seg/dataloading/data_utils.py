import os
from typing import Any, Callable, Dict, Tuple, Union

import mrcfile
import numpy as np
import SimpleITK as sitk
from skimage.util import img_as_float32
from torch import Tensor, device


def make_directory_if_not_exists(path: str):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str
        Path to the directory to be created.

    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_data_for_inference(data_path: str, transforms: Callable, device: device):
    """
    Load tomogram for inference.

    This function loads the tomogram, normalizes it, and performs defined
    transforms on it (most likely just conversion to Torch.Tensor).
    Additionally moves tomogram to GPU if available.

    Parameters
    ----------
    data_path : str
        Path to the tomogram to be loaded.
    transforms : callable
        A function or transform that takes in an ndarray and returns a transformed
        version.
    device : torch.device
        The device to which the data should be transferred.

    Returns
    -------
    new_data : torch.Tensor
        The transformed data, ready for inference. It has an extra batch
        dimension added, and is moved to the appropriate device.

    """
    tomogram = load_tomogram(data_path, normalize_data=True)
    tomogram = np.expand_dims(tomogram, 0)

    new_data = transforms(tomogram)
    new_data = new_data.unsqueeze(0)  # Add batch dimension
    new_data = new_data.to(device)
    return new_data


def store_segmented_tomograms(
    network_output: Tensor,
    out_folder: str,
    orig_data_path: str,
    ckpt_token: str,
    store_probabilities: bool = False,
) -> None:
    """
    Helper function for storing output segmentations.

    Stores segmentation into
    os.path.join(out_folder, os.path.basename(orig_data_path))
    If specified, also logits are stored before thresholding.

    Parameters
    ----------
    network_output : torch.Tensor
        The output from the network.
    out_folder : str
        Directory path to store the output segmentation.
    orig_data_path : str
        Original data path.
    ckpt_token : str
        Checkpoint token.
    store_probabilities : bool, optional
        If True, probabilities are stored before thresholding.
    """
    # Create out directory if it doesn't exist yet
    make_directory_if_not_exists(out_folder)

    predictions = network_output[0]
    predictions_np = predictions.squeeze(0).squeeze(0).cpu().numpy()
    out_folder = out_folder
    if store_probabilities:
        out_file = os.path.join(
            out_folder, os.path.basename(orig_data_path)[:-4] + "_scores.mrc"
        )
        store_tomogram(out_file, predictions_np)
    predictions_np_thres = predictions.squeeze(0).squeeze(0).cpu().numpy() > 0.0
    out_file_thres = os.path.join(
        out_folder,
        os.path.basename(orig_data_path)[:-4] + "_" + ckpt_token + "_segmented.mrc",
    )
    store_tomogram(out_file_thres, predictions_np_thres)
    print("MemBrain has finished segmenting your tomogram.")


def read_nifti(nifti_file: str) -> np.ndarray:
    """
    Read nifti files.

    Parameters
    ----------
    nifti_file : str
        Path to the nifti file.

    Returns
    -------
    a : np.ndarray
        Numpy array representation of the nifti file.

    """
    a = np.array(sitk.GetArrayFromImage(sitk.ReadImage(nifti_file)), dtype=float)
    return a


def load_tomogram(
    filename: str,
    return_pixel_size: bool = False,
    return_header: bool = False,
    normalize_data: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """
    Loads tomogram and transposes s.t. we have data in the form x,y,z.

    If specified, also normalizes the tomogram.

    Parameters
    ----------
    filename : str
        File name of the tomogram to load.
    return_pixel_size : bool, optional
        If True, return pixel size.
    return_header : bool, optional
        If True, return header.
    normalize_data : bool, optional
        If True, normalize data.

    Returns
    -------
    data : np.ndarray
        Numpy array of the loaded data.

    """
    with mrcfile.open(filename, permissive=True) as mrc:
        data = np.array(mrc.data)
        data = np.transpose(data, (2, 1, 0))
        cella = mrc.header.cella
        cellb = mrc.header.cellb
        origin = mrc.header.origin
        pixel_spacing = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        header_dict = {
            "cella": cella,
            "cellb": cellb,
            "origin": origin,
            "pixel_spacing": pixel_spacing,
        }
        if normalize_data:
            data = img_as_float32(data)
            data -= np.mean(data)
            data /= np.std(data)
        if return_pixel_size:
            return data, mrc.voxel_size
        if return_header:
            return data, header_dict
    return data


def store_tomogram(
    filename: str, tomogram: np.ndarray, header_dict: Dict[str, Any] = None
) -> None:
    """
    Store tomogram in specified path.

    Parameters
    ----------
    filename : str
        Name of the file to store the tomogram.
    tomogram : np.ndarray
        The tomogram data.
    header_dict : Dict[str, Any], optional
        Header dictionary to use.

    """
    if tomogram.dtype != np.int8:
        tomogram = np.array(tomogram, dtype=np.float32)
    tomogram = np.transpose(tomogram, (2, 1, 0))
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(tomogram)
        if header_dict is not None:
            mrc.header.cella = header_dict["cella"]
            mrc.header.cellb = header_dict["cellb"]
            mrc.header.origin = header_dict["origin"]


def normalize_tomogram(tomogram: np.ndarray) -> np.ndarray:
    """
    Normalize a tomogram to zero mean and unit standard deviation.

    Parameters
    ----------
    tomogram : np.ndarray
        Input tomogram to normalize.

    Returns
    -------
    np.ndarray
        Normalized tomogram with zero mean and unit standard deviation.
    """
    tomogram = img_as_float32(tomogram)
    tomogram -= np.mean(tomogram)
    tomogram /= np.std(tomogram)
    return tomogram
