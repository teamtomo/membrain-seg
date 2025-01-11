import logging
from typing import List, Optional

from typer import Option
from typing_extensions import Annotated

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli

logging.basicConfig(level=logging.INFO)


@cli.command(name="finetune", no_args_is_help=True)
def finetune(
    pretrained_checkpoint_path: str = Option(  # noqa: B008
        ...,
        help="Path to the checkpoint of the pre-trained model.",
        **PKWARGS,
    ),
    finetune_data_dir: str = Option(  # noqa: B008
        ...,
        help='Path to the directory containing the new data for fine-tuning. \
            Following the same required structure as the train function. \
            To learn more about the required\
            data structure, type "membrain data_structure_help"',
        **PKWARGS,
    ),
):
    """
    CLI for fine-tuning a pre-trained model.

    Initiates fine-tuning of a pre-trained model on new datasets
    and validation on original datasets.

    This function fine-tunes a pre-trained model on new datasets provided by the user.
    The directory specified by `finetune_data_dir` should be structured according to the
    requirements for the training function.
    For more details, use "membrain data_structure_help".

    Parameters
    ----------
    pretrained_checkpoint_path : str
        Path to the checkpoint file of the pre-trained model.
    finetune_data_dir : str
        Directory containing the new dataset for fine-tuning,
        structured as per the MemBrain's requirement.
        Use "membrain data_structure_help" for detailed information
        on the required data structure.

    Note

    ----

    This command configures and executes a fine-tuning session
    using the provided model checkpoint.
    The actual fine-tuning logic resides in the function '_fine_tune'.
    """
    from ..finetune import fine_tune as _fine_tune

    finetune_learning_rate = 1e-5
    log_dir = "logs_finetune/"
    batch_size = 2
    num_workers = 8
    max_epochs = 100
    early_stop_threshold = 0.05
    aug_prob_to_one = True
    use_deep_supervision = True
    project_name = "membrain-seg_finetune"
    sub_name = "1"

    _fine_tune(
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        finetune_data_dir=finetune_data_dir,
        finetune_learning_rate=finetune_learning_rate,
        log_dir=log_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        early_stop_threshold=early_stop_threshold,
        aug_prob_to_one=aug_prob_to_one,
        use_deep_supervision=use_deep_supervision,
        project_name=project_name,
        sub_name=sub_name,
    )


@cli.command(name="finetune_advanced", no_args_is_help=True)
def finetune_advanced(
    pretrained_checkpoint_path: str = Option(  # noqa: B008
        ...,
        help="Path to the checkpoint of the pre-trained model.",
        **PKWARGS,
    ),
    finetune_data_dir: str = Option(  # noqa: B008
        ...,
        help='Path to the directory containing the new data for fine-tuning. \
            Following the same required structure as the train function. \
            To learn more about the required\
            data structure, type "membrain data_structure_help"',
        **PKWARGS,
    ),
    finetune_learning_rate: float = Option(  # noqa: B008
        1e-5,
        help="Learning rate for fine-tuning the model. This parameter controls the \
          step size at each iteration while moving toward a minimum loss. \
          A smaller learning rate can lead to a more precise convergence but may \
          require more epochs. Adjust based on your dataset size and complexity.",
    ),
    log_dir: str = Option(  # noqa: B008
        "logs_fine_tune/",
        help="Log directory path. Finetuning logs will be stored here.",
    ),
    batch_size: int = Option(  # noqa: B008
        2,
        help="Batch size for training.",
    ),
    num_workers: int = Option(  # noqa: B008
        8,
        help="Number of worker threads for data loading.",
    ),
    max_epochs: int = Option(  # noqa: B008
        100,
        help="Maximum number of epochs for fine-tuning.",
    ),
    early_stop_threshold: float = Option(  # noqa: B008
        0.05,
        help="Threshold for early stopping based on validation loss deviation.",
    ),
    aug_prob_to_one: bool = Option(  # noqa: B008
        True,
        help='Whether to augment with a probability of one. This helps with the \
            model\'s generalization,\
            but also severely increases training time.\
                Pass "True" or "False".',
    ),
    use_surface_dice: bool = Option(  # noqa: B008
        False, help='Whether to use Surface-Dice as a loss. Pass "True" or "False".'
    ),
    surface_dice_weight: float = Option(  # noqa: B008
        1.0, help="Scaling factor for the Surface-Dice loss. "
    ),
    surface_dice_tokens: Annotated[
        Optional[List[str]],
        Option(
            help='List of tokens to \
            use for the Surface-Dice loss. \
            Pass tokens separately:\
            For example, train_advanced --surface_dice_tokens "ds1" \
            --surface_dice_tokens "ds2"'
        ),
    ] = None,
    use_deep_supervision: bool = Option(  # noqa: B008
        True, help='Whether to use deep supervision. Pass "True" or "False".'
    ),
    project_name: str = Option(  # noqa: B008
        "membrain-seg_v0_finetune",
        help="Project name. This helps to find your model again.",
    ),
    sub_name: str = Option(  # noqa: B008
        "1",
        help="Subproject name. For multiple runs in the same project,\
            please specify sub_names.",
    ),
):
    """
    CLI for fine-tuning a pre-trained model with advanced options.

    Initiates fine-tuning of a pre-trained model on new datasets
    and validation on original datasets with more advanced options.

    This function finetunes a pre-trained U-Net model on new data provided by the user.
    The `finetune_data_dir` should contain the following directories:
    - `imagesTr` and `labelsTr` for the user's own new training data.
    - `imagesVal` and `labelsVal` for the old data, which will be used
      for validation to ensure that the fine-tuned model's performance
      is not significantly worse on the original training data than the
      pre-trained model.

    Parameters
    ----------
    pretrained_checkpoint_path : str
        Path to the checkpoint file of the pre-trained model.
    finetune_data_dir : str
        Directory containing the new dataset for fine-tuning,
        structured as per the MemBrain's requirement.
        Use "membrain data_structure_help" for detailed information
        on the required data structure.
    finetune_learning_rate : float
        Learning rate for fine-tuning the model. This parameter controls the step size
        at each iteration while moving toward a minimum loss. A smaller learning rate
        can lead to a more precise convergence but may require more epochs.
        Adjust based on your dataset size and complexity.
    log_dir : str
        Path to the directory where logs will be stored, by default 'logs_fine_tune/'.
    batch_size : int
        Number of samples per batch, by default 2.
    num_workers : int
        Number of worker threads for data loading, by default 8.
    max_epochs : int
        Maximum number of fine-tuning epochs, by default 100.
    early_stop_threshold : float
        Threshold for early stopping based on validation loss deviation,
        by default 0.05.
    aug_prob_to_one : bool
        Determines whether to apply very strong data augmentation, by default True.
        If set to False, data augmentation still happens, but not as frequently.
        More data augmentation can lead to better performance, but also increases the
        training time substantially.
    use_surface_dice : bool
        Determines whether to use Surface-Dice loss, by default False.
    surface_dice_weight : float
        Scaling factor for the Surface-Dice loss, by default 1.0.
    surface_dice_tokens : list
        List of tokens to use for the Surface-Dice loss.
    use_deep_supervision : bool
        Determines whether to use deep supervision, by default True.
    project_name : str
        Name of the project for logging purposes, by default 'membrain-seg_v0_finetune'.
    sub_name : str
        Sub-name for the project, by default '1'.

    Note
    ----
    This command configures and executes a fine-tuning session
    using the provided model checkpoint.
    The actual fine-tuning logic resides in the function '_fine_tune'.
    """
    from ..finetune import fine_tune as _fine_tune

    _fine_tune(
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        finetune_data_dir=finetune_data_dir,
        finetune_learning_rate=finetune_learning_rate,
        log_dir=log_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        early_stop_threshold=early_stop_threshold,
        aug_prob_to_one=aug_prob_to_one,
        use_deep_supervision=use_deep_supervision,
        use_surf_dice=use_surface_dice,
        surf_dice_weight=surface_dice_weight,
        surf_dice_tokens=surface_dice_tokens,
        project_name=project_name,
        sub_name=sub_name,
    )
