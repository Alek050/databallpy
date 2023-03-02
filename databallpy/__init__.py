# read version from installed package
from importlib.metadata import version

__version__ = version("databallpy")

from databallpy.match import get_match, get_open_match
