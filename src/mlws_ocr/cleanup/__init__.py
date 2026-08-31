"""Image cleanup stages: deskew, illumination, binarize, despeckle.

Importing this package registers every built-in cleanup implementation.
"""
from . import binarize, deskew, despeckle, illumination  # noqa: F401
