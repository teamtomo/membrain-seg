import csv
import os
from typing import Any, Callable, Tuple, Union

import numpy as np
import SimpleITK as sitk
from skimage.util import img_as_float32
from torch import Tensor, device

from .tomo_io import ImageIO


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


def get_csv_data(csv_path, delimiter=",", with_header=False, return_header=False):
    """
    Load data from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    delimiter : str, optional
        Character used to separate fields. Default is ','.
    with_header : bool, optional
        If True, the function expects the CSV file to contain a header and will
        exclude it from the output data.
        Default is False.
    return_header : bool, optional
        If True, the function returns the header along with the data.
        Default is False.

    Returns
    -------
    out_array : numpy.ndarray
        Numpy array of data from the CSV file. If with_header or return_header is True,
        the first row (header) will be excluded from the array.
        If the CSV file is empty, a numpy array of shape (0, 13) will be returned.
    header : numpy.ndarray, optional
        Only returned if return_header is True. Numpy array containing the CSV
        file's header.

    Raises
    ------
    Exception
        If the data can't be converted to float numpy array, a numpy array with
        original type will be returned.

    """
    rows = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            rows.append(row)
    assert len(rows) != 0
    out_array = np.stack(rows)
    if return_header:
        try:
            out_array = np.array(out_array[1:, :], dtype=np.float), out_array[0, :]
        finally:
            out_array = np.array(out_array[1:, :]), out_array[0, :]
        return out_array
    if with_header:
        try:
            out_array = np.array(out_array[1:, :], dtype=np.float)
        finally:
            out_array = np.array(out_array[1:, :])
        return out_array
    try:
        out_array = np.array(out_array, dtype=np.float)
    except Exception:
        out_array = np.array(out_array)
    return out_array


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
    tomogram, mrc = load_tomogram(
        data_path, normalize_data=True, return_mrc_object=True
    )
    tomogram = np.expand_dims(tomogram, 0)

    new_data = transforms(tomogram)
    new_data = new_data.unsqueeze(0)  # Add batch dimension
    new_data = new_data.to(device)
    return new_data, mrc


def store_segmented_tomograms(
    network_output: Tensor,
    out_folder: str,
    orig_data_path: str,
    ckpt_token: str,
    store_probabilities: bool = False,
    mrc_object: ImageIO = None,
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
    mrc_object: ImageIO, optional
        If given, the mrc_object will be used to retain header information
        from another tomogram. This way, pixel sizes and other header
        information is not lost.
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
        store_tomogram(out_file, predictions_np, mrc=mrc_object)
    predictions_np_thres = predictions.squeeze(0).squeeze(0).cpu().numpy() > 0.0
    out_file_thres = os.path.join(
        out_folder,
        os.path.basename(orig_data_path)[:-4] + "_" + ckpt_token + "_segmented.mrc",
    )
    store_tomogram(out_file_thres, predictions_np_thres, mrc=mrc_object)
    print("MemBrain has finished segmenting your tomogram.")


def read_nifti(nifti_file: str) -> np.ndarray:
    """
    Read nifti file.

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


def write_nifti(out_file: str, image: np.ndarray) -> None:
    """
    Write nifti file.

    Parameters
    ----------
    out_file : str
        Path to the nifti file. (Where should it be stored?)
    image: np.ndarray
        3D tomogram that should be stored in the given file.

    Returns
    -------
    None

    """
    out_image = sitk.GetImageFromArray(image)
    sitk.WriteImage(out_image, out_file)


def load_tomogram(
    filename: str,
    return_mrc_object: bool = False,
    normalize_data: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """
    Loads tomogram and transposes s.t. we have data in the form x,y,z.

    If specified, also normalizes the tomogram.

    Parameters
    ----------
    filename : str
        File name of the tomogram to load.
    return_mrc_object : bool, optional
        If True, returns mrc object that contains more header information.
    normalize_data : bool, optional
        If True, normalize data.

    Returns
    -------
    data : np.ndarray
        Numpy array of the loaded data.

    """
    mrc = ImageIO(filename)
    mrc.read()
    data = mrc.data
    if normalize_data:
        data = img_as_float32(data)
        data -= np.mean(data)
        data /= np.std(data)
    if return_mrc_object:
        return data, mrc
    return data


def store_tomogram(filename: str, tomogram: np.ndarray, mrc: ImageIO = None) -> None:
    """
    Store tomogram in specified path.

    Parameters
    ----------
    filename : str
        Name of the file to store the tomogram.
    tomogram : np.ndarray
        The tomogram data.
    mrc : ImageIO, optional
        Mrc object containing header information.

    """
    if mrc is None:
        mrc = ImageIO()
    if tomogram.dtype == bool:
        tomogram = tomogram.astype("ubyte")
    mrc.write(filename, tomogram)


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
