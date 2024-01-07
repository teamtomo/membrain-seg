import os

import torch
from monai.inferers import SlidingWindowInferer

from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet

from .dataloading.data_utils import (
    load_data_for_inference,
    store_segmented_tomograms,
)
from .dataloading.memseg_augmentation import (
    adjust_mirrored_vectors,
    get_mirrored_img,
    get_prediction_transforms,
)


def segment(
    tomogram_path,
    ckpt_path,
    out_folder,
    store_normals=False,
    store_probabilities=False,
    sw_roi_size=160,
    store_connected_components=False,
    connected_component_thres=None,
    test_time_augmentation=True,
    segmentation_threshold=0.0,
):
    """
    Segment tomograms using a trained model.

    This function takes a path to a tomogram file, a path to a trained
    model checkpoint file, and a path to an output folder. It loads the
    trained model, and performs sliding window inference with 8-fold test-time
    augmentation on the new data, and then stores the resulting segmentations
    in the output folder.

    Parameters
    ----------
    tomogram_path : str
        Path to the tomogram file to be segmented.
    ckpt_path : str
        Path to the trained model checkpoint file.
    out_folder : str
        Path to the folder where the output segmentations should be stored.
    store_normals : bool, optional
        If True, store the predicted normals along with the segmentations
    store_probabilities : bool, optional
        If True, store the predicted probabilities along with the segmentations
        (default is False).
    sw_roi_size: int, optional
        Sliding window size used for inference. Smaller values than 160 consume less
        GPU, but also lead to worse segmentation results!
        Must be a multiple of 32.
    store_connected_components: bool, optional
        If True, connected components are computed and stored instead of the raw
        segmentation.
    connected_component_thres: int, optional
        If specified, all connected components smaller than this threshold
        are removed from the segmentation.
    test_time_augmentation: bool, optional
        If True, test-time augmentation is performed, i.e. data is rotated
        into eight different orientations and predictions are averaged.
    segmentation_threshold: float, optional
        Threshold for the membrane segmentation. Only voxels with a membrane
        score higher than this threshold will be segmented. (default: 0.0)

    Returns
    -------
    segmentation_file: str
        Path to the segmented tomogram.

    Raises
    ------
    FileNotFoundError
        If `tomogram_path` or `ckpt_path` does not point to a file.
    """
    model_checkpoint = ckpt_path
    ckpt_token = os.path.basename(model_checkpoint).split("-val_loss")[
        0
    ]  # TODO: Probably better to not keep this with custom checkpoint names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load trained weights from checkpoint
    try:
        # Attempt to load the model with compute_normal_vectors=False
        # (i.e. 1 output channel)
        pl_model = SemanticSegmentationUnet.load_from_checkpoint(
            model_checkpoint,
            map_location=device,
            strict=False,
            compute_normal_vectors=False,
        )
    except RuntimeError:
        # If an error occurs, try loading the model with compute_normal_vectors=True
        # (i.e. 4 output channels)
        pl_model = SemanticSegmentationUnet.load_from_checkpoint(
            model_checkpoint,
            map_location=device,
            strict=False,
            compute_normal_vectors=True,
        )
    pl_model.to(device)

    # Preprocess the new data
    new_data_path = tomogram_path
    transforms = get_prediction_transforms()
    new_data, mrc_header, voxel_size = load_data_for_inference(
        new_data_path, transforms, device=torch.device("cpu")
    )
    new_data = new_data.to(torch.float32)

    # Put the model into evaluation mode
    pl_model.eval()

    # Perform sliding window inference on the new data
    if sw_roi_size % 32 != 0:
        raise OSError("Sliding window size must be multiple of 32°!")
    roi_size = (sw_roi_size, sw_roi_size, sw_roi_size)
    sw_batch_size = 1
    inferer = SlidingWindowInferer(
        roi_size,
        sw_batch_size,
        overlap=0.5,
        progress=True,
        mode="gaussian",
        device=torch.device("cpu"),
    )

    # Perform test time augmentation (8-fold mirroring)
    predictions = torch.zeros_like(new_data)
    normal_predictions = torch.zeros([new_data.shape[0], 3, *new_data.shape[2:]])
    print("Performing 8-fold test-time augmentation.")
    for m in range(8 if test_time_augmentation else 1):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                mirrored_output = (
                    inferer(get_mirrored_img(new_data.clone(), m).to(device), pl_model)[
                        0
                    ]
                    .detach()
                    .cpu()
                )
                predictions += get_mirrored_img(mirrored_output[:, :1, ...], m)
                if store_normals:
                    assert (
                        mirrored_output.shape[1] == 4
                    ), "Model does not output normal vectors!"
                    normals_output_mirrored_back = get_mirrored_img(
                        mirrored_output[:, 1:, ...], m
                    )
                    normals_output_mirrored_back = adjust_mirrored_vectors(
                        normals_output_mirrored_back, m
                    )
                    normal_predictions += normals_output_mirrored_back

    if test_time_augmentation:
        predictions /= 8.0

    # Extract segmentations and store them in an output file.
    segmentation_file = store_segmented_tomograms(
        predictions,
        out_folder=out_folder,
        orig_data_path=new_data_path,
        ckpt_token=ckpt_token,
        normals_output=normal_predictions,
        store_normals=store_normals,
        store_probabilities=store_probabilities,
        store_connected_components=store_connected_components,
        connected_component_thres=connected_component_thres,
        mrc_header=mrc_header,
        voxel_size=voxel_size,
        segmentation_threshold=segmentation_threshold,
    )
    return segmentation_file
