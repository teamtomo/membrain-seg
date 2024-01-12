import csv
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import mrcfile
import numpy as np
import SimpleITK as sitk
import vtk
from skimage.util import img_as_float32
from torch import Tensor, device

from membrain_seg.segmentation.connected_components import connected_components


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


def store_array_in_csv(out_file: str, data: np.ndarray, out_del: str = ",") -> None:
    """
    Store tabular data in a CSV file.

    Parameters
    ----------
    out_file : str
        The output file path where the CSV will be stored.
    data : np.ndarray
        The data array to store in the CSV file.
    out_del : str, optional
        The delimiter to use in the CSV file, by default ",".
    """
    with open(out_file, "w") as out_csv:
        csv_writer = csv.writer(out_csv, delimiter=out_del)
        for i in range(data.shape[0]):
            row = data[i]
            csv_writer.writerow(row)


def store_point_and_vectors_in_vtp(
    out_path: str,
    in_points: np.ndarray,
    in_vectors: np.ndarray = None,
):
    """
    Store points and, optionally, their associated vectors into a VTP file.

    This function takes an array of points and an optional array of vectors
    corresponding to each point and stores them in a VTK PolyData format,
    which is then written to a VTP file.

    Parameters
    ----------
    out_path : str
        The path to the desired output VTP file.
    in_points : np.ndarray
        A Numpy array of points where each point is represented as [x, y, z].
        Shape should be (n_points, 3).
    in_vectors : np.ndarray, optional
        A Numpy array of vectors associated with each point, typically representing
        normals or other vector data. Shape should be (n_points, 3). If not provided,
        only point data is written to the VTP file.

    Returns
    -------
    None
        This function does not return a value. It writes directly to the specified file.

    Raises
    ------
    IOError
        If there is an error writing the file, an error message is printed.
    """
    points = vtk.vtkPoints()
    for point in in_points:
        points.InsertNextPoint(point[0], point[1], point[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    if in_vectors is not None:
        vectors = vtk.vtkDoubleArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("Normal")
        for vector in in_vectors:
            vectors.InsertNextTuple(vector)
        polydata.GetPointData().AddArray(vectors)
        polydata.GetPointData().SetActiveVectors(vectors.GetName())

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        error_msg = "Error writing the file"
        print(error_msg)


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
    new_data = np.expand_dims(tomogram.data, 0)

    new_data = transforms(new_data)
    new_data = new_data.unsqueeze(0)  # Add batch dimension
    new_data = new_data.to(device)
    return new_data, tomogram.header, tomogram.voxel_size


def store_segmented_tomograms(
    network_output: Tensor,
    out_folder: str,
    orig_data_path: str,
    ckpt_token: str,
    normals_output: Tensor = None,
    store_normals: bool = False,
    uncertainty_output: Tensor = None,
    store_uncertainty: bool = False,
    store_probabilities: bool = False,
    store_connected_components: bool = False,
    connected_component_thres: int = None,
    mrc_header: np.recarray = None,
    voxel_size: float = None,
    segmentation_threshold: float = 0.0,
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
    normals_output : torch.Tensor, optional
        The output from the network for the normals.
    store_normals : bool, optional
        If True, normals are stored along with the segmentations.
    uncertainty_output : torch.Tensor, optional
        The output from the network for the uncertainty.
    store_uncertainty : bool, optional
        If True, uncertainty is stored along with the segmentations.
    store_probabilities : bool, optional
        If True, probabilities are stored before thresholding.
    store_connected_components : bool, optional
        If True, connected components of the segmentations are computed.
    connected_component_thres : int, optional
        If specified, all connected components smaller than this threshold
        are removed from the segmentation.
    mrc_header: np.recarray, optional
        If given, the mrc header will be used to retain header information
        from another tomogram. This way, pixel sizes and other header
        information is not lost.
    voxel_size: float, optional
        If given, this will be the voxel size stored in the header of the
        output segmentation.
    segmentation_threshold : float, optional
        Threshold for the segmentation. Default is 0.0.
    """
    # Create out directory if it doesn't exist yet
    make_directory_if_not_exists(out_folder)

    predictions = network_output[0]
    predictions_np = predictions.squeeze(0).squeeze(0).cpu().numpy()
    out_folder = out_folder
    if store_probabilities:
        out_file = os.path.join(
            out_folder,
            os.path.splitext(os.path.basename(orig_data_path))[0] + "_scores.mrc",
        )
        out_tomo = Tomogram(
            data=predictions_np, header=mrc_header, voxel_size=voxel_size
        )
        store_tomogram(out_file, out_tomo)
    predictions_np_thres = (
        predictions.squeeze(0).squeeze(0).cpu().numpy() > segmentation_threshold
    )
    out_file_thres = os.path.join(
        out_folder,
        os.path.splitext(os.path.basename(orig_data_path))[0]
        + "_"
        + ckpt_token
        + "_segmented.mrc",
    )
    if store_connected_components:
        predictions_np_thres = connected_components(
            predictions_np_thres, size_thres=connected_component_thres
        )
    out_tomo = Tomogram(
        data=predictions_np_thres, header=mrc_header, voxel_size=voxel_size
    )
    store_tomogram(out_file_thres, out_tomo)
    if store_normals:
        assert normals_output is not None
        normals = normals_output[0]  # shape [3, x, y, z]
        normals_np = normals.cpu().numpy()
        out_files_normals = [
            os.path.join(
                out_folder,
                os.path.splitext(os.path.basename(orig_data_path))[0]
                + "_"
                + ckpt_token
                + "_normals"
                + str(i)
                + ".mrc",
            )
            for i in range(3)
        ]
        for i, out_file_normals in enumerate(out_files_normals):
            out_tomo = Tomogram(
                data=normals_np[i], header=mrc_header, voxel_size=voxel_size
            )
            store_tomogram(out_file_normals, out_tomo)
    if store_uncertainty:
        assert uncertainty_output is not None
        out_file_uncertainty = os.path.join(
            out_folder,
            os.path.splitext(os.path.basename(orig_data_path))[0]
            + "_"
            + ckpt_token
            + "_uncertainty.mrc",
        )
        out_tomo = Tomogram(
            data=uncertainty_output, header=mrc_header, voxel_size=voxel_size
        )
        store_tomogram(out_file_uncertainty, out_tomo)

    print("MemBrain has finished segmenting your tomogram.")
    return out_file_thres


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


@dataclass
class Tomogram:
    """
    A class used to represent a Tomogram.

    Attributes
    ----------
    data : np.ndarray
        The 3D array data representing the tomogram.
    header : Any
        The header information from the tomogram file.
    voxel_size : Any, optional
        The voxel size of the tomogram.
    """

    data: np.ndarray
    header: Any
    voxel_size: Optional[Any] = None


def load_tomogram(
    filename: str,
    normalize_data: bool = False,
) -> Tomogram:
    """
    Loads tomogram and transposes s.t. we have data in the form x,y,z.

    If specified, also normalizes the tomogram.

    Parameters
    ----------
    filename : str
        File name of the tomogram to load.
    normalize_data : bool, optional
        If True, normalize data.

    Returns
    -------
    tomogram : Tomogram
        A Tomogram dataclass containing the loaded data, header
        and voxel size.

    """
    with mrcfile.open(filename, permissive=True) as tomogram:
        data = tomogram.data.copy()
        data = np.transpose(data, (2, 1, 0))
        header = tomogram.header
        voxel_size = tomogram.voxel_size
    if normalize_data:
        data = img_as_float32(data)
        data -= np.mean(data)
        data /= np.std(data)
    return Tomogram(data=data, header=header, voxel_size=voxel_size)


_dtype_to_mode = {
    np.dtype("float16"): 12,
    np.dtype("float32"): 2,
    np.dtype("int8"): 0,
    np.dtype("int16"): 1,
    np.dtype("uint8"): 6,
    np.dtype("uint16"): 6,
    np.dtype("complex64"): 4,
}


def convert_dtype(tomogram: np.ndarray) -> np.ndarray:
    """
    Convert tomogram data to a less memory-intensive dtype if possible.

    Parameters
    ----------
    tomogram : np.ndarray
        Input tomogram data.

    Returns
    -------
    np.ndarray
        Tomogram data in a possibly more memory-efficient dtype.

    Raises
    ------
    ValueError
        If the dtype of the tomogram is not in _dtype_to_mode and can't be converted
        to a more memory-efficient dtype.
    """
    dtype = tomogram.dtype
    # Check if data can be represented as int or uint
    if np.allclose(tomogram, tomogram.astype(int)):
        if (
            tomogram.min() >= np.iinfo("int8").min
            and tomogram.max() <= np.iinfo("int8").max
        ):
            return tomogram.astype("int8")
        elif (
            tomogram.min() >= np.iinfo("int16").min
            and tomogram.max() <= np.iinfo("int16").max
        ):
            return tomogram.astype("int16")
        elif np.all(tomogram >= 0):
            if tomogram.max() <= np.iinfo("uint8").max:
                return tomogram.astype("uint8")
            elif tomogram.max() <= np.iinfo("uint16").max:
                return tomogram.astype("uint16")
    # Check if data can be represented as float
    if (
        tomogram.min() >= np.finfo("float16").min
        and tomogram.max() <= np.finfo("float16").max
    ) and np.allclose(tomogram, tomogram.astype("float16")):
        return tomogram.astype("float16")
    elif (
        tomogram.min() >= np.finfo("float32").min
        and tomogram.max() <= np.finfo("float32").max
    ):
        return tomogram.astype("float32")
    # If none of the above, and dtype is in _dtype_to_mode, keep original dtype
    if dtype in _dtype_to_mode:
        return tomogram
    # Otherwise, raise an error
    raise ValueError(f"Cannot convert tomogram of dtype {dtype}")


def store_tomogram(
    filename: str, tomogram: Union[Tomogram, np.ndarray], voxel_size=None
) -> None:
    """
    Store tomogram in specified path.

    Parameters
    ----------
    filename : str
        Name of the file to store the tomogram.
    tomogram : Tomogram or np.ndarray
        The tomogram data if given as np.ndarray. If given as a Tomogram,
        both data and header are used for storing.
    voxel_size: float, optional
        If specified, this voxel size will be stored into the tomogram header.
    """
    with mrcfile.new(filename, overwrite=True) as out_mrc:
        if isinstance(tomogram, Tomogram):
            data = tomogram.data
            header = tomogram.header
            if voxel_size is None:
                voxel_size = tomogram.voxel_size
        else:
            data = tomogram
            header = None

        if header is not None:
            attributes = header.dtype.names
            for attr in attributes:
                if attr not in ["nlabl", "label"]:
                    continue
                setattr(out_mrc.header, attr, getattr(header, attr))

        data = convert_dtype(data)
        data = np.transpose(data, (2, 1, 0))
        out_mrc.set_data(data)
        if voxel_size is not None:
            out_mrc.voxel_size = voxel_size


def normalize_tomogram(
    tomogram: np.ndarray, cut_extreme_values: bool = False
) -> np.ndarray:
    """
    Normalize a tomogram to zero mean and unit standard deviation.

    Parameters
    ----------
    tomogram : np.ndarray
        Input tomogram to normalize.
    cut_extreme_values : bool, optional
        If True, values outside of the range [-5*std, 5*std] are cut to these values.
        Default is False.

    Returns
    -------
    np.ndarray
        Normalized tomogram with zero mean and unit standard deviation.
    """
    tomogram = img_as_float32(tomogram)
    if cut_extreme_values:
        temp_std = np.std(tomogram)
        tomogram[tomogram < -5 * temp_std] = -5 * temp_std
        tomogram[tomogram > 5 * temp_std] = 5 * temp_std
    tomogram -= np.mean(tomogram)
    tomogram /= np.std(tomogram)
    return tomogram
