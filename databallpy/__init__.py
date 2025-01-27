from importlib.metadata import version

from databallpy.utils.get_game import (
    Game,
    get_game,
    get_match,
    get_open_game,
    get_open_match,
    get_saved_game,
    get_saved_match,
)
from databallpy.utils.logging import create_logger

__version__ = version("databallpy")
create_logger(__name__).info(f"Using databallpy version {__version__}")
