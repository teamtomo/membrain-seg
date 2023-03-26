import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from membrain_seg.dataloading.memseg_pl_datamodule import MemBrainSegDataModule
from membrain_seg.training.unet import SemanticSegmentationUnet

warnings.filterwarnings("ignore", category=UserWarning, module="torch._tensor")
warnings.filterwarnings("ignore", category=UserWarning, module="monai.data")


def main(args):
    """Start training the model."""
    # Set up the data module
    data_module = MemBrainSegDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Set up the model
    model = SemanticSegmentationUnet()

    # Set up logging
    wandb_logger = pl_loggers.WandbLogger(
        project="membrain_seg", log_model=True, save_code=True
    )
    csv_logger = pl_loggers.CSVLogger(args.log_dir)

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="membrain_seg-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )

    # Set up the trainer
    trainer = pl.Trainer(
        logger=[wandb_logger, csv_logger],
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
    )

    # Start the training process
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # This seems to be causing problems with PL version 2.0.0

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scicore/home/engel0006/GROUP/pool-engel/\
    Lorenz/MemBrain-seg/data",
    )
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)

    args = parser.parse_args()

    main(args)
