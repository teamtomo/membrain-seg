def print_training_parameters(
    data_dir: str = "",
    log_dir: str = "logs/",
    batch_size: int = 2,
    num_workers: int = 8,
    max_epochs: int = 1000,
    aug_prob_to_one: bool = False,
    use_deep_supervision: bool = False,
    project_name: str = "membrain-seg_v0",
    sub_name: str = "1",
    use_surf_dice: bool = False,
    surf_dice_weight: float = 1.0,
    surf_dice_tokens: list = None,
):
    """
    Print a formatted overview of the training parameters with explanations.

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
    print("\033[1mTraining Parameters Overview:\033[0m\n")
    print(
        "Data Directory:\n   '{}' \n   Path to the directory containing "
        "training data.".format(data_dir)
    )
    print("————————————————————————————————————————————————————————")
    print(
        "Log Directory:\n   '{}' \n   Directory where logs and outputs will "
        "be stored.".format(log_dir)
    )
    print("————————————————————————————————————————————————————————")
    print(
        "Batch Size:\n   {} \n   Number of samples processed in a single batch.".format(
            batch_size
        )
    )
    print("————————————————————————————————————————————————————————")
    print(
        "Number of Workers:\n   {} \n   Subprocesses to use for data "
        "loading.".format(num_workers)
    )
    print("————————————————————————————————————————————————————————")
    print(f"Max Epochs:\n   {max_epochs} \n   Maximum number of training epochs.")
    print("————————————————————————————————————————————————————————")
    aug_status = "Enabled" if aug_prob_to_one else "Disabled"
    print(
        "Augmentation Probability to One:\n   {} \n   If enabled, sets all "
        "augmentation probabilities to 1. (strong augmentation)".format(aug_status)
    )
    print("————————————————————————————————————————————————————————")
    deep_sup_status = "Enabled" if use_deep_supervision else "Disabled"
    print(
        "Use Deep Supervision:\n   {} \n   If enabled, activates deep "
        "supervision in model.".format(deep_sup_status)
    )
    print("————————————————————————————————————————————————————————")
    print(
        "Project Name:\n   '{}' \n   Name identifier for the current"
        " training session.".format(project_name)
    )
    print("————————————————————————————————————————————————————————")
    print(
        "Sub Name:\n   '{}' \n   Additional sub-identifier for organizing"
        " outputs.".format(sub_name)
    )
    print("————————————————————————————————————————————————————————")
    surf_dice_status = "Enabled" if use_surf_dice else "Disabled"
    print(
        "Use Surface Dice:\n   {} \n   If enabled, includes Surface-Dice in the loss "
        "calculation.".format(surf_dice_status)
    )
    print("————————————————————————————————————————————————————————")
    print(
        "Surface Dice Weight:\n   {} \n   Weighting of the Surface-Dice"
        " loss, if enabled.".format(surf_dice_weight)
    )
    print("————————————————————————————————————————————————————————")
    if surf_dice_tokens:
        tokens = ", ".join(surf_dice_tokens)
        print(
            "Surface Dice Tokens:\n   [{}] \n   Specific tokens used for "
            "Surface-Dice loss. Other tokens will be neglected.".format(tokens)
        )
    else:
        print(
            "Surface Dice Tokens:\n None \n No specific tokens are used for "
            "Surface-Dice loss."
        )
    print("\n")
