from typing import List, Optional

from typer import Option
from typing_extensions import Annotated

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


@cli.command(name="train", no_args_is_help=True)
def train(
    data_dir: str = Option(  # noqa: B008
        ...,
        help='Data directory path, following the required structure. To learn more \
            about the required data structure, type "membrain data_structure_help"',
        **PKWARGS,
    ),
):
    """
    Initiates the MemBrain training routine.

    Parameters
    ----------
    data_dir : str
        Path to the data directory, structured as per the MemBrain-seg's requirement.
        Type "membrain data_structure_help" for detailed information on the required
        data structure.

    Note

    ----

    The actual training logic resides in the function '_train'.
    """
    from ..train import train as _train

    log_dir = "./logs"
    batch_size = 2
    num_workers = 8
    on_the_fly_dataloading = False
    max_epochs = 1000
    aug_prob_to_one = True
    use_deep_supervision = True
    project_name = "membrain-seg_v0"
    sub_name = "1"

    _train(
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
    )


@cli.command(name="train_advanced", no_args_is_help=True)
def train_advanced(
    data_dir: str = Option(  # noqa: B008
        ...,
        help='Data directory path, following the required structure. \
            To learn more about the required\
            data structure, type "membrain data_structure_help"',
        **PKWARGS,
    ),
    log_dir: str = Option(  # noqa: B008
        "logs/",
        help="Log directory path. Training logs will be stored here.",
    ),
    batch_size: int = Option(  # noqa: B008
        2,
        help="Batch size for training.",
    ),
    num_workers: int = Option(  # noqa: B008
        8,
        help="Number of worker threads for loading data",
    ),
    on_the_fly_dataloading: bool = Option(  # noqa: B008
        False,
        help="Whether to load data on the fly. This is useful for large datasets.",
    ),
    max_epochs: int = Option(  # noqa: B008
        1000,
        help="Maximum number of epochs for training",
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
        "membrain-seg_v0",
        help="Project name. This helps to find your model again.",
    ),
    sub_name: str = Option(  # noqa: B008
        "1",
        help="Subproject name. For multiple runs in the same project,\
            please specify sub_names.",
    ),
    use_BCE_dice: bool = Option(  # noqa: B008
        True,
        help="Should a combination of Dice and binary cross entropy loss be used?",
    ),
    fourier_amplitude_aug: bool = Option(  # noqa: B008
        False,
        help="Should use Fourier amplitude matching as data augmentation?",
    ),
    missing_wedge_aug: bool = Option(  # noqa: B008
        False,
        help="Should additional, artificial missing wedges be used for data \
            augmentation?",
    ),
    exclude_deepict_from_dice: bool = Option(  # noqa: B008
        False,
        help="Should we compute only Surface Dice for Deepict data?",
    ),
    no_sdice_for_no_deepict: bool = Option(  # noqa: B008
        False,
        help="Should we compute only normal dice for non-Deepict samples?",
    ),
    cosine_annealing_interval: int = Option(  # noqa: B008
        None,
        help="If an integer is specified, cosine annealing with warm restarts \
            is performed in the given interval times.",
    ),
):
    """
    Initiates the MemBrain training routine with more advanced options.

    Parameters
    ----------
    data_dir : str
        Path to the data directory, structured as per the MemBrain's requirement.
        Use "membrain data_structure_help" for detailed information on the required
        data structure.
    log_dir : str
        Path to the directory where logs will be stored, by default 'logs/'.
    batch_size : int
        Number of samples per batch, by default 2.
    num_workers : int
        Number of worker threads for data loading, by default 1.
    on_the_fly_dataloading : bool
        Determines whether to load data on the fly, by default False.
    max_epochs : int
        Maximum number of training epochs, by default 1000.
    aug_prob_to_one : bool
        Determines whether to apply very strong data augmentation, by default True.
        If set to False, data augmentation still happens, but not as frequently.
        More data augmentation can lead to a better performance, but also increases the
        training time substantially.
    use_surface_dice : bool
        Determines whether to use Surface-Dice loss, by default True.
    surface_dice_weight : float
        Scaling factor for the Surface-Dice loss, by default 1.0.
    surface_dice_tokens : list
        List of tokens to use for the Surface-Dice loss, by default ["all"].
    use_deep_supervision : bool
        Determines whether to use deep supervision, by default True.
    project_name : str
        Name of the project for logging purposes, by default 'membrain-seg_v0'.
    sub_name : str
        Sub-name for the project, by default '1'.
    use_surf_dice : bool
        Determines whether to use Surface Dice loss, by default False.
    surf_sice_weight : float
        Surface Dice weight compared to BCE Dice loss, by default 1.0.
    use_BCE_dice : bool
        Determines whether to use a combination of Dice and binary cross entropy loss,
        by default True.
    fourier_amplitude_aug : bool
        Determines whether to use Fourier amplitude matching as data augmentation,
        by default False.
    missing_wedge_aug : bool
        Determines whether to use additional, artificial missing wedges for data
        augmentation, by default False.
    exclude_deepict_from_dice : bool
        Determines whether to compute only Surface Dice for Deepict data,
        by default False.
    no_sdice_for_no_deepict : bool
        Determines whether to compute only normal Dice for non-Deepict samples,
        by default False.
    cosine_annealing_interval : int
        If an integer is specified, cosine annealing with warm restarts is
        performed in the given interval times.


    Note

    ----

    The actual training logic resides in the function '_train'.
    """
    from ..train import train as _train

    _train(
        data_dir=data_dir,
        log_dir=log_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        on_the_fly_dataloading=on_the_fly_dataloading,
        max_epochs=max_epochs,
        aug_prob_to_one=aug_prob_to_one,
        use_deep_supervision=use_deep_supervision,
        use_surf_dice=use_surface_dice,
        surf_dice_weight=surface_dice_weight,
        surf_dice_tokens=surface_dice_tokens,
        project_name=project_name,
        sub_name=sub_name,
        use_BCE_dice=use_BCE_dice,
        missing_wedge_aug=missing_wedge_aug,
        fourier_amplitude_aug=fourier_amplitude_aug,
        exclude_deepict_from_dice=exclude_deepict_from_dice,
        no_sdice_for_no_deepict=no_sdice_for_no_deepict,
        cosine_annealing_interval=cosine_annealing_interval,
    )


@cli.command(name="data_structure_help")
def data_dir_help():
    """
    Display information about the training data directory structure.

    Note:
    ----
        The file names of images and their corresponding labels should match.
        The segmentation algorithm uses this assumption to pair images with labels.
    """
    print(
        "The data directory structure should be as follows:\n"
        "data_dir/\n"
        "├── imagesTr/       # Directory containing training images\n"
        "│   ├── img1.nii.gz    # Image file (currently requires nii.gz format)\n"
        "│   ├── img2.nii.gz    # Image file\n"
        "│   └── ...\n"
        "├── imagesVal/      # Directory containing validation images\n"
        "│   ├── img3.nii.gz    # Image file\n"
        "│   ├── img4.nii.gz    # Image file\n"
        "│   └── ...\n"
        "├── labelsTr/       # Directory containing training labels\n"
        "│   ├── img1.nii.gz  # Label file (currently requires nii.gz format)\n"
        "│   ├── img2.nii.gz  # Label file\n"
        "│   └── ...\n"
        "└── labelsVal/      # Directory containing validation labels\n"
        "    ├── img3.nii.gz  # Label file\n"
        "    ├── img4.nii.gz  # Label file\n"
        "    └── ..."
    )
