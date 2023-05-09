# read version from installed package
from importlib.metadata import version

from databallpy.get_match import get_match, get_open_match, get_saved_match

__version__ = version("databallpy")
