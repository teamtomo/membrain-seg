"""CLI init function."""

# These imports are necessary to register CLI commands. Do not remove!
from .cli import cli  # noqa: F401
from .fine_tune_cli import finetune  # noqa: F401
from .segment_cli import segment  # noqa: F401
from .ske_cli import skeletonize  # noqa: F401
from .train_cli import data_dir_help, train  # noqa: F401
