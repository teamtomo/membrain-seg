import os

import mrcfile
import numpy as np
import SimpleITK as sitk
from skimage.utils import img_as_float32


def load_data_for_inference(data_path, transforms, device):
    """Load tomogram for inference.

    This function loads the tomogram, normalizes it, and performs defined
    transforms on it (most likely just conversion to Torch.Tensor).
    Additionally moves tomogram to GPU if available.
    """
    tomogram = load_tomogram(data_path, normalize_data=True)
    tomogram = np.expand_dims(tomogram, 0)

    new_data = transforms(tomogram)
    new_data = new_data.unsqueeze(0)  # Add batch dimension
    new_data = new_data.to(device)
    return new_data


def store_segmented_tomograms(
    network_output, out_folder, orig_data_path, ckpt_token, store_probabilities=False
):
    """Helper function for storing output segmentations.

    Stores segmentation into
    os.path.join(out_folder, os.path.basename(orig_data_path))
    If specified, also logits are stored before thresholding.
    """
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


def read_nifti(nifti_file):
    """Read nifti files. This will be redundant once we move to mrc files I guess?."""
    a = np.array(sitk.GetArrayFromImage(sitk.ReadImage(nifti_file)), dtype=float)
    return a


def load_tomogram(
    filename, return_pixel_size=False, return_header=False, normalize_data=False
):
    """
    Loads data and transposes s.t. we have data in the form x,y,z.

    If specified, tomogram values are normalized to zero mean and unit std.
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


def store_tomogram(filename, tomogram, header_dict=None):
    """Store tomogram in specified path."""
    if tomogram.dtype != np.int8:
        tomogram = np.array(tomogram, dtype=np.float32)
    tomogram = np.transpose(tomogram, (2, 1, 0))
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(tomogram)
        if header_dict is not None:
            mrc.header.cella = header_dict["cella"]
            mrc.header.cellb = header_dict["cellb"]
            mrc.header.origin = header_dict["origin"]


def normalize_tomogram(tomogram):
    """Normalize tomogram to zero mean and unit standard deviation."""
    tomogram -= np.mean(tomogram)
    tomogram /= np.std(tomogram)
    return tomogram
