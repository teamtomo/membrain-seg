"""Normal processing module for membrain-seg."""
from .cli import cli  # noqa: F401
from .extract_normals import (  # noqa: F401
    extract_normals_GT,
    match_coords_to_membrane_normals,
)
