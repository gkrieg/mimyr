"""MIMYR: generative modeling of spatial transcriptomics.

    import mimyr
    model = mimyr.Mimyr.from_pretrained("merscope")
    section = model.generate(reference=[left, right], position=10.9)

See mimyr.api for the full interface.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mimyr")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"

__all__ = ["Mimyr", "prepare", "validate", "evaluate"]


def __getattr__(name):
    # Imported on first use: the API pulls in torch and scanpy, and `import mimyr`
    # should stay cheap for the submodules (mimyr.pretrained, mimyr.metrics).
    if name in __all__:
        from mimyr import api

        return getattr(api, name)
    raise AttributeError(f"module 'mimyr' has no attribute {name!r}")
