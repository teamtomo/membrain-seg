import os

import torch
from monai.inferers import SlidingWindowInferer

from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet

from .dataloading.data_utils import (
    load_data_for_inference,
    store_segmented_tomograms,
)
from .dataloading.memseg_augmentation import get_mirrored_img, get_prediction_transforms


def segment(
    tomogram_path, ckpt_path, out_folder, store_probabilities=False, sw_roi_size=160
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
    store_probabilities : bool, optional
        If True, store the predicted probabilities along with the segmentations
        (default is False).
    sw_roi_size: int, optional
        Sliding window size used for inference. Smaller values than 160 consume less
        GPU, but also lead to worse segmentation results!
        Must be a multiple of 32.

    Returns
    -------
    None

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
    pl_model = SemanticSegmentationUnet()
    pl_model = pl_model.load_from_checkpoint(model_checkpoint, map_location=device)
    pl_model.to(device)

    # Preprocess the new data
    new_data_path = tomogram_path
    transforms = get_prediction_transforms()
    new_data, mrc_header = load_data_for_inference(
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
    print("Performing 8-fold test-time augmentation.")
    for m in range(8):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                predictions += (
                    get_mirrored_img(
                        inferer(
                            get_mirrored_img(new_data.clone(), m).to(device), pl_model
                        )[0],
                        m,
                    )
                    .detach()
                    .cpu()
                )

    predictions /= 8.0

    # Extract segmentations and store them in an output file.
    store_segmented_tomograms(
        predictions,
        out_folder=out_folder,
        orig_data_path=new_data_path,
        ckpt_token=ckpt_token,
        store_probabilities=store_probabilities,
        mrc_header=mrc_header,
    )
