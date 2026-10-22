"""Paths to the metadata files bundled with the package."""

import os

METADATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "generative_transformer", "metadata"
)
META_INFO_FILE = "4hierarchy_metainfo_mouse_geneunion2_DAG.pt"
META_INFO_PATH = os.path.join(METADATA_DIR, META_INFO_FILE)
ID_TO_SUBCLASS_PATH = os.path.join(METADATA_DIR, "id_to_subclass.pkl")
