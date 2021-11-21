"""CONFIG FILE."""

import os


class Config:
    """Configuration."""

    ROOT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

    # BIRDS Constants
    BIRDS_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data", "birds")
    BIRDS_IMG_WIDTH = 224
    BIRDS_IMG_HEIGHT = 224
