# read version from installed package
from importlib.metadata import version

__version__ = version("databallpy")


class DataBallPyError(Exception):
    "Error class specific for databallpy"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
