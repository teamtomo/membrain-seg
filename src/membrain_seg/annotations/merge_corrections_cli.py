from typer import Option

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli
from .merge_corrections import convert_single_nrrd_files as _convert_single_nrrd_files


@cli.command(name="merge_corrections", no_args_is_help=True)
def merge_corrections(
    labels_dir: str = Option(  # noqa: B008
        ..., help="Path to the directory containing label files.", **PKWARGS
    ),
    corrections_dir: str = Option(  # noqa: B008
        ...,
        help="Path to the directory containing correction folders. Each correction \
        folder should have the \
        same name as the corresponding label file (without the file extension).",
        **PKWARGS,
    ),
    out_dir: str = Option(  # noqa: B008
        ...,
        help="Path to the directory where corrected label files will be saved.",
        **PKWARGS,
    ),
):
    r"""
    Merge manual corrections into a single file.

    Process all label files in a given directory, find corresponding correction
    folders,
    apply corrections and write corrected files to an output directory.

    Notes
    -----
    This function expects each label file to have a corresponding folder in
    corrections_dir.
    If no such folder is found, a warning will be printed.

    Required data structure
    -----

    root_directory/
    ├── labels_dir/
    │   ├── label_file1
    │   ├── label_file2
    │   ├── label_file3
    │   └── ...
    ├── corrections_dir/
    │   ├── label_file1/
    │   │   ├── Add1.nrrd
    │   │   ├── Add2.nrrd
    │   │   ├── Remove1.nrrd
    │   │   ├── Ignore1.nrrd
    │   │   ├── Ignore2.nrrd
    │   │   ├── Ignore3.nrrd
    │   │   └── ...
    │   ├── label_file2/
    │   │   ├── Add1.nrrd
    │   │   ├── Add2.nrrd
    │   │   ├── Ignore1.nrrd
    │   │   └── ...
    │   ├── label_file3/
    │   │   ├── Add1.nrrd
    │   │   ├── Ignore1.nrrd
    │   │   └── ...
    │   └── ...
    └── out_dir/ (This directory will be filled with the corrected files

    after running the script)

    """
    _convert_single_nrrd_files(
        labels_dir=labels_dir, corrections_dir=corrections_dir, out_dir=out_dir
    )
