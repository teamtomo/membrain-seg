from typer import Option

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli
from .extract_normals import extract_normals_GT as _extract_normals
from .extract_normals import match_coords_to_membrane_normals as _match_coords


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


@cli.command(name="match_coords_to_normals", no_args_is_help=True)
def match_coords(
    coords_file: str = Option(  # noqa: B008
        ...,
        help="Path to the .csv file containing coordinates to compute normals for",
        **PKWARGS,
    ),
    out_coords_file: str = Option(  # noqa: B008
        ...,
        help="Path to the .csv file that should store computed normals and angles",
        **PKWARGS,
    ),
    membrane_seg_path: str = Option(  # noqa: B008
        ...,
        help="Path to the membrane segmentation file.",
        **PKWARGS,
    ),
    normal_array_path: str = Option(  # noqa: B008
        None, help="Path to the membrane normal array file."
    ),
    euler_conversion: str = Option(  # noqa: B008
        None, help="The convention to use for Euler angles ('zxz' or 'zyz')."
    ),
    min_dist_thres: float = Option(  # noqa: B008
        10.0, help="Maximal distance to a membrane to extract normals."
    ),
    smoothing: int = Option(  # noqa: B008
        2000, help="Number of smoothing iterations to apply to the resulting mesh."
    ),
    decimation_degree: float = Option(  # noqa: B008
        0.5,
        help="Decimation of mesh for efficiency. Higher values are faster,"
        " but less accurate. Value must be between 0 and 1.",
    ),
):
    """
    Matches input coordinates to membrane normal vectors.

    The output is saved in the specified CSV file, and
    optionally, Euler angles are calculated for the normals.

    The command requires a path to a CSV file containing coordinates and
    an output path for the resulting matched normals and angles. Either a
    membrane segmentation file or a precomputed normal array path must be
    provided to compute or load membrane normals.

    Parameters
    ----------
    coords_file : str
        Input .csv file with coordinates.
    out_coords_file : str
        Output .csv file for computed normals and angles.
    membrane_seg_path : str, optional
        Membrane segmentation file path.
    normal_array_path : str, optional
        Precomputed membrane normal array file path.
        Provide path to one of the three normal component files.
        (other two should be within the same directory)
    euler_conversion : str, optional
        Convention for Euler angles ('zxz' or 'zyz').
    min_dist_thres : float, optional
        Max distance to a membrane for extracting normals.
    smoothing : int, optional
        Smoothing iterations for the mesh.
    decimation_degree : float, optional
        Mesh decimation degree (0-1) for efficiency.

    """
    assert 0 <= decimation_degree < 1, "Decimation degree must be between 0 and 1."
    assert (
        membrane_seg_path is not None or normal_array_path is not None
    ), "Either membrane_seg_path or normal_array_path must be provided."

    _match_coords(
        coords_file=coords_file,
        out_coords_file=out_coords_file,
        membrane_seg_path=membrane_seg_path,
        membrane_normals_path=normal_array_path,
        euler_conversion=euler_conversion,
        min_dist_thres=min_dist_thres,
        smoothing=smoothing,
        decimation_degree=decimation_degree,
    )
