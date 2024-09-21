import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from membrain_seg.segmentation.dataloading.memseg_pl_datamodule import (
    MemBrainSegDataModule,
)
from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from membrain_seg.segmentation.training.optim_utils import PrintLearningRate
from membrain_seg.segmentation.training.training_param_summary import (
    print_training_parameters,
)

os.environ["WANDB_MODE"] = "offline"

warnings.filterwarnings("ignore", category=UserWarning, module="torch._tensor")
warnings.filterwarnings("ignore", category=UserWarning, module="monai.data")


def train(
    data_dir: str = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-seg/data",
    log_dir: str = "logs/",
    batch_size: int = 2,
    num_workers: int = 8,
    on_the_fly_dataloading: bool = False,
    max_epochs: int = 1000,
    aug_prob_to_one: bool = False,
    use_deep_supervision: bool = False,
    project_name: str = "membrain-seg_v0",
    sub_name: str = "1",
    use_BCE_dice: bool = True,
    use_surf_dice: bool = False,
    surf_dice_weight: float = 1.0,
    exclude_deepict_from_dice: bool = False,
    no_sdice_for_no_deepict: bool = False,
    cosine_annealing_interval: int = None,
    surf_dice_tokens: list = None,
    missing_wedge_aug: bool = False,
    fourier_amplitude_aug: bool = False,
):
    """
    Train the model on the specified data.

    The function sets up a data module and a model, configures logging,
    model checkpointing and learning rate monitoring,
    and starts the training process.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing training data.
    log_dir : str, optional
        Path to the directory where logs should be stored.
    batch_size : int, optional
        Number of samples per batch of input data.
    num_workers : int, optional
        Number of subprocesses to use for data loading.
    on_the_fly_dataloading : bool, optional
        If True, data is loaded on the fly.
    max_epochs : int, optional
        Maximum number of epochs to train for.
    aug_prob_to_one : bool, optional
        If True, all augmentation probabilities are set to 1.
    use_deep_supervision : bool, optional
        If True, enables deep supervision in the U-Net model.
    project_name : str, optional
        Name of the project for logging purposes.
    sub_name : str, optional
        Sub-name of the project for logging purposes.
    use_BCE_dice : bool, optional
        If True, uses a combination of Dice and binary cross entropy loss.
    use_surf_dice : bool, optional
        If True, uses Surface Dice loss.
    surf_dice_weight : float, optional
        Surface Dice weight compared to BCE Dice loss.
    missing_wedge_aug : bool, optional
        If True, uses additional, artificial missing wedges for data augmentation.
    fourier_amplitude_aug : bool, optional
        If True, uses Fourier amplitude matching as data augmentation.
    exclude_deepict_from_dice : bool, optional
        If True, computes only Surface Dice for Deepict data.
    no_sdice_for_no_deepict : bool, optional
        If True, computes only normal Dice for non-Deepict samples.
    cosine_annealing_interval : int, optional
        If an integer is specified, cosine annealing with warm restarts is
        performed in the given interval times.
    use_surf_dice : bool, optional
        If True, enables Surface-Dice loss.
    surf_dice_weight : float, optional
        Weight for the Surface-Dice loss.
    surf_dice_tokens : list, optional
        List of tokens to use for the Surface-Dice loss.

    Returns
    -------
    None
    """
    print_training_parameters(
        data_dir=data_dir,
        log_dir=log_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        on_the_fly_dataloading=on_the_fly_dataloading,
        max_epochs=max_epochs,
        aug_prob_to_one=aug_prob_to_one,
        use_deep_supervision=use_deep_supervision,
        project_name=project_name,
        sub_name=sub_name,
        use_surf_dice=use_surf_dice,
        surf_dice_weight=surf_dice_weight,
        surf_dice_tokens=surf_dice_tokens,
    )
    # Set up the data module
    data_module = MemBrainSegDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        on_the_fly_dataloading=on_the_fly_dataloading,
        aug_prob_to_one=aug_prob_to_one,
        fourier_amplitude_aug=fourier_amplitude_aug,
        missing_wedge_aug=missing_wedge_aug,
    )

    # Set up the model
    model = SemanticSegmentationUnet(
        max_epochs=max_epochs,
        use_deep_supervision=use_deep_supervision,
        use_BCE_dice=use_BCE_dice,
        use_surf_dice=use_surf_dice,
        surf_dice_weight=surf_dice_weight,
        surf_dice_tokens=surf_dice_tokens,
    )

    project_name = project_name
    checkpointing_name = project_name + "_" + sub_name
    # Set up logging
    pl_loggers.WandbLogger(
        project=project_name, log_model=False, save_code=True, name=sub_name
    )
    csv_logger = pl_loggers.CSVLogger(log_dir)

    # Set up model checkpointing
    ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpointing_name + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    checkpoint_callback_last = ModelCheckpoint(
        save_top_k=1,  # Save only the best model
        monitor="epoch",  # Monitor 'epoch' to save the last epoch model
        mode="max",  # Set mode to 'max' to save the model with maximum 'epoch'
        dirpath="checkpoints/",
        filename=checkpointing_name + "-{epoch}-{val_loss:.2f}",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    print_lr_cb = PrintLearningRate()

    # Set up the trainer
    trainer = pl.Trainer(
        precision="16-mixed",
        logger=[csv_logger],
        callbacks=[
            # checkpoint_callback_val_loss,
            checkpoint_callback_last,
            lr_monitor,
            print_lr_cb,
        ],
        max_epochs=max_epochs,
    )

    # Start the training process
    trainer.fit(model, data_module)
