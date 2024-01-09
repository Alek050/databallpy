from importlib.metadata import version

from databallpy.get_match import get_match, get_open_match, get_saved_match
from databallpy.utils.logging import create_logger

__version__ = version("databallpy")
create_logger(__name__).info(f"Using databallpy version {__version__}")
