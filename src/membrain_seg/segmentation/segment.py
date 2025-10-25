import logging
import os

import torch
from monai.inferers import SlidingWindowInferer

from membrain_seg.segmentation.networks.inference_unet import (
    PreprocessedSemanticSegmentationUnet,
)
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (
    determine_output_shape,
)

from .dataloading.data_utils import (
    load_data_for_inference,
    store_segmented_tomograms,
)
from .dataloading.memseg_augmentation import get_mirrored_img, get_prediction_transforms


def segment(
    tomogram_path,
    ckpt_path,
    out_folder,
    rescale_patches=False,
    in_pixel_size=None,
    out_pixel_size=10.0,
    store_probabilities=False,
    sw_roi_size=160,
    store_connected_components=False,
    connected_component_thres=None,
    test_time_augmentation=True,
    store_uncertainty_map=False,
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
    rescale_patches : bool, optional
        If True, rescale the patches to the output pixel size (default is False).
    in_pixel_size : float, optional
        Pixel size of the input tomogram in Angstrom (default is None).
    out_pixel_size : float, optional
        Pixel size of the output segmentation in Angstrom (default is 10.0).
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
    store_uncertainty_map: bool, optional
        If True, store an uncertainty map based on the voxel-wise variance
        across the TTA predictions. Requires test_time_augmentation=True.
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
    # Load the trained PyTorch Lightning model
    model_checkpoint = ckpt_path
    ckpt_token = os.path.basename(model_checkpoint).split("-val_loss")[
        0
    ]  # TODO: Probably better to not keep this with custom checkpoint names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load trained weights from checkpoint
    pl_model = PreprocessedSemanticSegmentationUnet.load_from_checkpoint(
        model_checkpoint, map_location=device, strict=False
    )
    pl_model.to(device)
    if sw_roi_size % 32 != 0:
        raise OSError("Sliding window size must be multiple of 32°!")
    pl_model.target_shape = (sw_roi_size, sw_roi_size, sw_roi_size)

    # Preprocess the new data
    new_data_path = tomogram_path
    transforms = get_prediction_transforms()
    new_data, mrc_header, voxel_size = load_data_for_inference(
        new_data_path, transforms, device=torch.device("cpu")
    )
    new_data = new_data.to(torch.float32)

    if rescale_patches:
        # Rescale patches if necessary
        if in_pixel_size is None:
            in_pixel_size = voxel_size.x
        if in_pixel_size == 0.0:
            raise ValueError(
                "Input pixel size is 0.0. Please specify the pixel size manually."
            )
        if in_pixel_size == 1.0:
            logging.warning(
                "WARNING: Input pixel size is 1.0. Looks like a corrupt header.",
                "Please specify the pixel size manually.",
            )
        pl_model.rescale_patches = in_pixel_size != out_pixel_size

        # Determine the sliding window size according to the input and output pixel size
        sw_roi_size = determine_output_shape(
            # switch in and out pixel size to get SW shape
            pixel_size_in=out_pixel_size,
            pixel_size_out=in_pixel_size,
            orig_shape=(sw_roi_size, sw_roi_size, sw_roi_size),
        )
        sw_roi_size = sw_roi_size[0]

    # Put the model into evaluation mode
    pl_model.eval()

    # Perform sliding window inference on the new data
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
    if store_uncertainty_map:
        # make an assert statement to make sure that test_time_augmentation=True
        assert test_time_augmentation, (
            "To store uncertainty maps, test_time_augmentation must be True."
            "Otherwise, variance cannot be computed."
        )
        all_tta_predictions = torch.zeros((8,) + predictions.shape)
    if test_time_augmentation:
        logging.info(
            "Performing 8-fold test-time augmentation. "
            + "I.e. the following bar will run 8 times.",
        )
    for m in range(8 if test_time_augmentation else 1):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                mirrored_input = get_mirrored_img(new_data.clone(), m).to(device)
                mirrored_pred = inferer(mirrored_input, pl_model)
                if not (
                    isinstance(mirrored_pred, list) or isinstance(mirrored_pred, tuple)
                ):
                    mirrored_pred = [mirrored_pred]
                correct_pred = get_mirrored_img(mirrored_pred[0], m)
                predictions += correct_pred.detach().cpu()
                # After finishing prediction for this TTA variant, 
                # store its probability map if uncertainty maps are enabled
                if store_uncertainty_map:
                    all_tta_predictions[m] = correct_pred.detach().cpu()
    if test_time_augmentation:
        predictions /= 8.0
        all_tta_predictions = torch.sigmoid(all_tta_predictions)
        uncertainty_map = torch.var(all_tta_predictions, dim=0)

    # Extract segmentations and store them in an output file.
    segmentation_file = store_segmented_tomograms(
        predictions,
        out_folder=out_folder,
        orig_data_path=new_data_path,
        ckpt_token=ckpt_token,
        store_probabilities=store_probabilities,
        store_connected_components=store_connected_components,
        connected_component_thres=connected_component_thres,
        mrc_header=mrc_header,
        voxel_size=voxel_size,
        segmentation_threshold=segmentation_threshold,
        store_uncertainty_map=store_uncertainty_map,
        uncertainty_map=uncertainty_map if store_uncertainty_map else None,
    )
    return segmentation_file
