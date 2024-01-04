"""Copied from https://github.com/teamtomo/fidder/blob/main/src/fidder/_cli.py."""

import typer
from click import Context
from typer import Option
from typer.core import TyperGroup

from .extract_normals import extract_normals as _extract_normals


class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}
PKWARGS = OPTION_PROMPT_KWARGS


@cli.callback()
def callback():
    """
    [green]MemBrain-seg's[/green] normal processing module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    membrain_normals <command> --help

    -------

    Example:
    -------
    membrain_normals extract_normals

    -------
    """


@cli.command(name="extract_normals_for_GT", no_args_is_help=True)
def extract_normals(
    task_dir: str = Option(  # noqa: B008
        ..., help="Path to the directory containing training directories.", **PKWARGS
    ),
    min_dist_thres: float = Option(  # noqa: B008
        3.0, help="Maximal distance to a membrane to extract normals."
    ),
    decimation_degree: float = Option(  # noqa: B008
        0.5,
        help="Decimation of mesh for efficiency. Higher values are faster,"
        " but less accurate. Value must be between 0 and 1.",
    ),
):
    """
    Extract normal vectors from ground truth segmentation files.

    This function processes each .nii.gz file in the task's training and
    validation directories, computes normals, and saves them in a specified
    output directory. The directory structure is assumed to have separate
    subdirectories for training ('labelsTr') and validation
    ('labelsVal') data.

    Parameters
    ----------
    task_dir : str
        Path to the directory containing training and validation directories.
        This directory should contain the 'labelsTr' and 'labelsVal'
        subdirectories.
    min_dist_thres : float, optional
        The maximum distance to a membrane for extracting normals. This parameter
        defines how close a point must be to the surface to consider its normal.
        A lower value means only points very close to the actual membrane are
        considered.
        Default is 3.0.
    decimation_degree : float, optional
        The decimation level of the mesh for efficiency. This value must be between
        0 and 1, where higher values represent a greater degree of decimation leading
        to faster processing but less accuracy in the normal calculations.
        Default is 0.5.

    Directory Structure Expected:
    .
    └── task_dir
        ├── labelsTr       # Training labels directory
        ├── imagesTr       # Training images directory
        ├── labelsVal      # Validation labels directory
        └── imagesVal      # Validation images directory

    Notes
    -----
    The function assumes the presence of 'labelsTr' and 'labelsVal' directories
    within the task_dir. Each .nii.gz file within these directories will be
    processed to extract surface normals and save them accordingly.
    """
    assert 0 <= decimation_degree < 1, "Decimation degree must be between 0 and 1."
    _extract_normals(
        task_dir=task_dir,
        min_dist_thres=min_dist_thres,
        decimation_degree=decimation_degree,
    )
