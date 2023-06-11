import os

import torch
from monai.inferers import SlidingWindowInferer

from membrain_seg.networks.unet import SemanticSegmentationUnet

from .dataloading.data_utils import (
    load_data_for_inference,
    store_segmented_tomograms,
)
from .dataloading.memseg_augmentation import get_mirrored_img, get_prediction_transforms


def segment(tomogram_path, ckpt_path, out_folder, store_probabilities=False):
    """Segment you tomograms using a trained model."""
    # Load the trained PyTorch Lightning model
    model_checkpoint = ckpt_path
    ckpt_token = os.path.basename(model_checkpoint).split("-val_loss")[
        0
    ]  # TODO: Probably better to not keep this with custom checkpoint names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load trained weights from checkpoint
    pl_model = SemanticSegmentationUnet()
    pl_model = pl_model.load_from_checkpoint(model_checkpoint)
    pl_model.to(device)

    # Preprocess the new data
    new_data_path = tomogram_path
    transforms = get_prediction_transforms()
    new_data = load_data_for_inference(new_data_path, transforms, device)
    new_data = new_data.to(torch.float32)

    # Put the model into evaluation mode
    pl_model.eval()

    # Perform sliding window inference on the new data
    roi_size = (160, 160, 160)
    sw_batch_size = 2
    inferer = SlidingWindowInferer(
        roi_size, sw_batch_size, overlap=0.5, progress=True, mode="gaussian"
    )

    # Perform test time augmentation (8-fold mirroring)
    predictions = torch.zeros_like(new_data)
    for m in range(8):
        with torch.no_grad():
            predictions += get_mirrored_img(
                inferer(get_mirrored_img(new_data.clone(), m), pl_model)[0], m
            )
    predictions /= 8.0

    # Extract segmentations and store them in an output file.
    store_segmented_tomograms(
        predictions,
        out_folder=out_folder,
        orig_data_path=new_data_path,
        ckpt_token=ckpt_token,
        store_probabilities=store_probabilities,
    )
