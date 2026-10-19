"""MIMYR: generative modeling of spatial transcriptomics."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mimyr")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"
