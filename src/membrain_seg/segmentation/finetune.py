import logging

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from membrain_seg.segmentation.dataloading.memseg_pl_datamodule import (
    MemBrainSegDataModule,
)
from membrain_seg.segmentation.networks.unet import SemanticSegmentationUnet
from membrain_seg.segmentation.training.optim_utils import (
    ToleranceCallback,
)
from membrain_seg.segmentation.training.training_param_summary import (
    print_training_parameters,
)


def fine_tune(
    pretrained_checkpoint_path: str,
    finetune_data_dir: str,
    finetune_learning_rate: float = 1e-5,
    log_dir: str = "logs_finetune/",
    batch_size: int = 2,
    num_workers: int = 8,
    max_epochs: int = 100,
    early_stop_threshold: float = 0.05,
    aug_prob_to_one: bool = False,
    use_deep_supervision: bool = False,
    project_name: str = "membrain-seg_finetune",
    sub_name: str = "1",
    use_surf_dice: bool = False,
    surf_dice_weight: float = 1.0,
    surf_dice_tokens: list = None,
) -> None:
    """
    Fine-tune a pre-trained U-Net model on new datasets.

    This function finetunes a pre-trained U-Net model on new data provided by the user.
    The `finetune_data_dir` should contain the following directories:
    - `imagesTr` and `labelsTr` for the user's own new training data.
    - `imagesVal` and `labelsVal` for the old data, which will be used
      for validation to ensure that the fine-tuned model's performance
      is not significantly worse on the original training data than the
      pre-trained model.

    Callbacks used during the fine-tuning process
    ---------
    - ModelCheckpoint: Saves the model checkpoints based on training loss
      and at regular intervals.
    - ToleranceCallback: Stops training if the validation loss deviates significantly
      from the baseline value set after the first epoch.
    - LearningRateMonitor: Monitors and logs the learning rate during training.
    - PrintLearningRate: Prints the current learning rate at the start of each epoch.

    Parameters
    ----------
    pretrained_checkpoint_path : str
        Path to the checkpoint of the pre-trained model.
    finetune_data_dir : str
        Path to the directory containing the new data for fine-tuning
        and old data for validation.
    finetune_learning_rate : float, optional
        Learning rate for fine-tuning, by default 1e-5.
    log_dir : str, optional
        Path to the directory where logs should be stored.
    batch_size : int, optional
        Number of samples per batch of input data.
    num_workers : int, optional
        Number of subprocesses to use for data loading.
    max_epochs : int, optional
        Maximum number of epochs to finetune, by default 100.
    early_stop_threshold : float, optional
        Threshold for early stopping based on validation loss deviation,
        by default 0.05.
    aug_prob_to_one : bool, optional
        If True, all augmentation probabilities are set to 1.
    use_deep_supervision : bool, optional
        If True, enables deep supervision in the U-Net model.
    project_name : str, optional
        Name of the project for logging purposes.
    sub_name : str, optional
        Sub-name of the project for logging purposes.
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
    # Print training parameters for verification
    print_training_parameters(
        data_dir=finetune_data_dir,
        log_dir=log_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        aug_prob_to_one=aug_prob_to_one,
        use_deep_supervision=use_deep_supervision,
        project_name=project_name,
        sub_name=sub_name,
        use_surf_dice=use_surf_dice,
        surf_dice_weight=surf_dice_weight,
        surf_dice_tokens=surf_dice_tokens,
    )
    logging.info("————————————————————————————————————————————————————————")
    logging.info(
        f"Pretrained Checkpoint:\n"
        f"   '{pretrained_checkpoint_path}' \n"
        f"   Path to the pretrained model checkpoint."
    )
    logging.info("\n")

    # Initialize the data module with fine-tuning datasets
    # New data for finetuning and old data for validation
    finetune_data_module = MemBrainSegDataModule(
        data_dir=finetune_data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        aug_prob_to_one=aug_prob_to_one,
    )

    # Load the pre-trained model with updated learning rate
    pretrained_model = SemanticSegmentationUnet.load_from_checkpoint(
        pretrained_checkpoint_path, learning_rate=finetune_learning_rate
    )

    checkpointing_name = project_name + "_" + sub_name

    # Set up logging
    csv_logger = pl_loggers.CSVLogger(log_dir)

    # Set up model checkpointing based on training loss
    checkpoint_callback_train_loss = ModelCheckpoint(
        dirpath="finetuned_checkpoints/",
        filename=checkpointing_name + "-{epoch:02d}-{train_loss:.2f}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
    )

    # Set up regular checkpointing every 5 epochs
    checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=5,
        dirpath="finetuned_checkpoints/",
        filename=checkpointing_name + "-{epoch}-{train_loss:.2f}",
        verbose=True,  # Print a message when a checkpoint is saved
    )

    # Set up ToleranceCallback by monitoring validation loss
    early_stop_metric = "val_loss"
    tolerance_callback = ToleranceCallback(early_stop_metric, early_stop_threshold)

    # Monitor learning rate changes
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    # Initialize the trainer with specified precision, logger, and callbacks
    trainer = pl.Trainer(
        precision="16-mixed",
        logger=[csv_logger],
        callbacks=[
            checkpoint_callback_train_loss,
            checkpoint_callback_regular,
            lr_monitor,
            tolerance_callback,
        ],
        max_epochs=max_epochs,
    )

    # Start the fine-tuning process
    trainer.fit(pretrained_model, finetune_data_module)
