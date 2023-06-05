"""Copied from https://github.com/teamtomo/fidder/blob/main/src/fidder/_cli.py."""

import typer
from click import Context
from typer.core import TyperGroup


class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(cls=OrderCommands, add_completion=False, no_args_is_help=True)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}


@cli.callback()
def callback():
    """
    MemBrain-seg's preprocessing module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:
        preprocessing <command> --help

    Example:
    -------
    To extract the spectrum, type:
        "preprocessing match_pixel_size --input_tomogram <tomogram_file>
            --output_path <matched_tomogram_file>
            --pixel_size_out 10.0 --pixel_size_in 15.0"

    """
