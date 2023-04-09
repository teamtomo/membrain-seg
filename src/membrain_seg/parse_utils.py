import argparse


def str2bool(v):
    """Converts a parsed string to a boolean value."""
    if v.lower() in ("yes", "true", "t", "y", "1", "True"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0", "False"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
