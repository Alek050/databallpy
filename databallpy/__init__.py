from importlib.metadata import version

from databallpy.utils.get_game import get_game, get_open_game, get_saved_game
from databallpy.utils.logging import create_logger

__version__ = version("databallpy")
create_logger(__name__).info(f"Using databallpy version {__version__}")
