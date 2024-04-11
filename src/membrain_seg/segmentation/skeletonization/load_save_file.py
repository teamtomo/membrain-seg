from dataclasses import dataclass
from typing import Any, Optional, Union

import mrcfile
import numpy as np
import SimpleITK as sitk
from skimage.util import img_as_float32


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


def load_tomogram(filename: str, normalize_data: bool = False) -> Tomogram:
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
        A Tomogram dataclass containing the loaded data, header, and voxel size.
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
        If the dtype of the tomogram is not in _dtype_to_mode and can not be converted
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

    # If none of the above, and dtype is in _dtype_to_mode, keep the original dtype
    if dtype in _dtype_to_mode:
        return tomogram

    # Otherwise, raise an error
    raise ValueError(f"Cannot convert tomogram of dtype {dtype}")


def store_tomogram(
    filename: str, tomogram: Union[Tomogram, np.ndarray], voxel_size=None
) -> None:
    """
    Store tomogram in the specified path.

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
