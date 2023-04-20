import argparse
import os

import torch
from dataloading.data_utils import (
    load_data_for_inference,
    store_segmented_tomograms,
)
from dataloading.memseg_augmentation import get_mirrored_img, get_prediction_transforms
from monai.inferers import SlidingWindowInferer

from membrain_seg.networks.unet import SemanticSegmentationUnet

# Argument parser for the tomogram path
parser = argparse.ArgumentParser(
    description="Segment membranes in a tomogram using a trained MemBrain model"
)
parser.add_argument(
    "--tomogram_path", type=str, help="Path to the tomogram to be segmented", default=""
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="Path to the model checkpoint that should be used",
    default="./checkpoints/membrain_v7_528_merged_DA_DS-epoch=1999-val_loss=0.59.ckpt",
)
parser.add_argument(
    "--out_folder",
    type=str,
    help="Path to the folder where segmentations should be stored",
    default="./predictions",
)
parser.add_argument(
    "--store_probabilities",
    type=bool,
    help="Should probability maps be output in addition to segmentations?",
    default=False,
)
args = parser.parse_args()

# Load the trained PyTorch Lightning model
model_checkpoint = args.ckpt_path
ckpt_token = os.path.basename(model_checkpoint).split("-val_loss")[
    0
]  # TODO: Probably better to not keep this with custom checkpoint names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load trained weights from checkpoint
pl_model = SemanticSegmentationUnet()
pl_model = pl_model.load_from_checkpoint(model_checkpoint)
pl_model.to(device)

# Preprocess the new data
new_data_path = args.tomogram_path
transforms = get_prediction_transforms()
new_data = load_data_for_inference(new_data_path, transforms, device)

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
    out_folder=args.out_folder,
    orig_data_path=new_data_path,
    ckpt_token=ckpt_token,
    store_probabilities=args.store_probabilities,
)
