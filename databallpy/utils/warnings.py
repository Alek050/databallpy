import functools
import warnings


class DataBallPyWarning(Warning):
    "Warning class specific for databallpy"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


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
