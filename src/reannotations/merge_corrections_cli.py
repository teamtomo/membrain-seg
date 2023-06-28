from typer import Option

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli
from .merge_corrections import convert_single_nrrd_files as _convert_single_nrrd_files


@cli.command(name="merge_corrections", no_args_is_help=True)
def merge_corrections(
    labels_dir: str = Option(  # noqa: B008
        help="Path to the directory containing label files.", **PKWARGS
    ),
    corrections_dir: str = Option(  # noqa: B008
        help="Path to the directory containing correction folders. Each correction \
        folder should have the \
        same name as the corresponding label file (without the file extension).",
        **PKWARGS,
    ),
    out_dir: str = Option(  # noqa: B008
        help="Path to the directory where corrected label files will be saved.",
        **PKWARGS,
    ),
):
    r"""
    Merge manual corrections into a single file.\n.

    Process all label files in a given directory, find corresponding correction
    folders,\n
    apply corrections and write corrected files to an output directory.\n

    Notes\n
    -----\n
    This function expects each label file to have a corresponding folder in
    corrections_dir.\n
    If no such folder is found, a warning will be printed.\n

    Required data structure\n
    -----\n

    root_directory/\n
    ├── labels_dir/\n
    │   ├── label_file1\n
    │   ├── label_file2\n
    │   ├── label_file3\n
    │   └── ...\n
    ├── corrections_dir/\n
    │   ├── label_file1/\n
    │   │   ├── Add_file1\n
    │   │   ├── Remove_file1\n
    │   │   ├── Ignore_file1\n
    │   │   └── ...\n
    │   ├── label_file2/\n
    │   │   ├── Add_file2\n
    │   │   ├── Remove_file2\n
    │   │   ├── Ignore_file2\n
    │   │   └── ...\n
    │   ├── label_file3/\n
    │   │   ├── Add_file3\n
    │   │   ├── Remove_file3\n
    │   │   ├── Ignore_file3\n
    │   │   └── ...\n
    │   └── ...\n
    └── out_dir/ (This directory will be filled with the corrected files \n
    after running the script)\n

    """
    _convert_single_nrrd_files(labels_dir, corrections_dir, out_dir)
