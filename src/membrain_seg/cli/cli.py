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
PKWARGS = OPTION_PROMPT_KWARGS


@cli.callback()
def callback():
    """
    MemBrain-seg's training / prediction module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    membrain <command> --help

    -------

    Example:
    -------
    membrain predict --tomogram-path <path-to-your-tomo>
        --ckpt-path <path-to-model-checkpoint>
        --out-folder ./segmentations

    -------
    """
