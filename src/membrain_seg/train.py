import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from membrain_seg.dataloading.memseg_pl_datamodule import MemBrainSegDataModule
from membrain_seg.networks.unet import SemanticSegmentationUnet
from membrain_seg.parse_utils import str2bool

warnings.filterwarnings("ignore", category=UserWarning, module="torch._tensor")
warnings.filterwarnings("ignore", category=UserWarning, module="monai.data")


def main(args):
    """Start training the model."""
    # Set up the data module
    data_module = MemBrainSegDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_prob_to_one=args.aug_prob_to_one,
    )

    # Set up the model
    model = SemanticSegmentationUnet(
        max_epochs=args.max_epochs, use_deep_supervision=args.use_deep_supervision
    )

    project_name = args.project_name
    checkpointing_name = project_name + "_" + args.sub_name
    # Set up logging
    wandb_logger = pl_loggers.WandbLogger(
        project=project_name, log_model=False, save_code=True
    )
    csv_logger = pl_loggers.CSVLogger(args.log_dir)

    # Set up model checkpointing
    checkpoint_callback_val_loss = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpointing_name + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=100,
        dirpath="checkpoints/",
        filename=checkpointing_name
        + "-{epoch}-{val_loss:.2f}",  # Customize the filename of saved checkpoints
        verbose=True,  # Print a message when a checkpoint is saved
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    class PrintLearningRate(Callback):
        def on_epoch_start(self, trainer, pl_module):
            current_lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(f"Epoch {trainer.current_epoch}: Learning Rate = {current_lr}")

    print_lr_cb = PrintLearningRate()
    # Set up the trainer
    trainer = pl.Trainer(
        logger=[csv_logger, wandb_logger],
        callbacks=[
            checkpoint_callback_val_loss,
            checkpoint_callback_regular,
            lr_monitor,
            print_lr_cb,
        ],
        max_epochs=args.max_epochs,
    )

    # Start the training process
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # This seems to be causing problems with PL version 2.0.0

    # TODO: Adjust parser to be useful!
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scicore/home/engel0006/GROUP/pool-engel/\
Lorenz/MemBrain-seg/data",
    )
    """
    The data_dir should have the following structure:
        data_dir/
        ├── imagesTr/       # Directory containing training images
        │   ├── img1.nii.gz    # Image file (currently requires nii.gz format)
        │   ├── img2.nii.gz    # Image file
        │   └── ...
        ├── imagesVal/      # Directory containing validation images
        │   ├── img3.nii.gz    # Image file
        │   ├── img4.nii.gz    # Image file
        │   └── ...
        ├── labelsTr/       # Directory containing training labels
        │   ├── img1.nii.gz  # Label file (currently requires nii.gz format)
        │   ├── img2.nii.gz  # Label file
        │   └── ...
        └── labelsVal/      # Directory containing validation labels
            ├── img3.nii.gz  # Label file
            ├── img4.nii.gz  # Label file
            └── ...
    """
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument(
        "--aug_prob_to_one",
        type=str2bool,
        default=False,
        help='Pass "True" or "False".',
    )
    parser.add_argument(
        "--use_deep_supervision",
        type=str2bool,
        default=False,
        help='Pass "True" or "False".',
    )
    parser.add_argument("--project_name", type=str, default="membrain-seg_v0")
    parser.add_argument("--sub_name", type=str, default="1")

    args = parser.parse_args()

    main(args)
