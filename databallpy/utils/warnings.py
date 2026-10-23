import functools
import sys
import warnings


class DataBallPyWarning(Warning):
    "Warning class specific for databallpy"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


DATABALLPY_EVENT_DEPRECATION = (
    "The 'databallpy_event' column is deprecated and will be removed in v0.11.0. "
    "Use the SPADL columns 'action_type', 'action_result' and 'action_bodypart' "
    "instead."
)


def warn_databallpy_event_deprecation(key) -> None:
    """Function to warn that the databallpy_event column is deprecated if it is
    accessed through the key. Should be called in __getitem__ or __getattr__. Access
    by databallpy itself, also through pandas or pandera, is not warned for."""
    if not (isinstance(key, str) and key == "databallpy_event"):
        return

    # frame 0 is this function, frame 1 is __getitem__/__getattr__
    frame, stacklevel = sys._getframe(2), 3
    packages = set()
    while frame is not None and _get_package(frame) in ("pandas", "pandera"):
        packages.add(_get_package(frame))
        frame, stacklevel = frame.f_back, stacklevel + 1
    # pandera accesses every column when validating
    if "pandera" in packages or (
        frame is not None and _get_package(frame) == "databallpy"
    ):
        return
    warnings.warn(DATABALLPY_EVENT_DEPRECATION, DataBallPyWarning, stacklevel=stacklevel)


def _get_package(frame) -> str:
    """Function to get the top level package of the code of a frame."""
    return frame.f_globals.get("__name__", "").split(".")[0]


def deprecated(message):
    """This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted with a custom message when the function is used."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=DeprecationWarning, stacklevel=3)
            return func(*args, **kwargs)

        return wrapper

    return decorator
