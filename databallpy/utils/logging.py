import logging
import os
import traceback
from configparser import ConfigParser
from functools import wraps
from typing import Callable, TypeVar

R = TypeVar("R")


def logging_wrapper(file_name: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        logger = create_logger(file_name)

        @wraps(func)
        def wrapper(*args, **kwargs) -> R:
            try:
                logger.info(f"Trying to run the function `{func.__name__}`")
                result = func(*args, **kwargs)
                logger.info(f"Successfully ran the function `{func.__name__}`")
                return result
            except Exception as e:
                logger.error(
                    f"Error in the function `{func.__name__}`: {e}\n{traceback.format_exc()}"
                )
                raise e

        return wrapper

    return decorator


def create_logger(
    name: str, user_config_path: str = "databallpy_config.ini"
) -> logging.Logger:
    """Creates a logger with the given name, using the configuration files in the
    databallpy package and the current working directory.

    To overwrite the base configuration of logging, create a `databallpy_config.ini`
    in your current working directory. The file can look something like this:

    [logging]
    file_log_level = DEBUG
    output = file
    log_level = DEBUG

    This sets the file log level to debug, outputs only in the file, and sets the
    general log level to debug (otherwise output will not be shown in the file).
    The logging file `databallpy.log` will appear in your working directory.

    Args:
        name (str): The name of the logger to create.

    Returns:
        logging.Logger: The logger with the given name.
    """

    user_config_path = (
        os.path.join(os.getcwd(), user_config_path)
        if not os.path.isabs(user_config_path)
        else user_config_path
    )
    user_config = ConfigParser()
    user_config.read(user_config_path)

    base_config_path = os.path.join(
        os.path.dirname(__file__), "..", "base_logging_config.ini"
    )
    base_config = ConfigParser()
    base_config.read(base_config_path)

    log_format = user_config.get(
        "logging", "format", fallback=base_config.get("logging", "format")
    )
    log_level = user_config.get(
        "logging", "log_level", fallback=base_config.get("logging", "log_level")
    )
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.hasHandlers():
        if "console" in user_config.get(
            "logging", "output", fallback=base_config.get("logging", "output")
        ):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            level = user_config.get(
                "logging",
                "console_log_level",
                fallback=base_config.get("logging", "console_log_level"),
            )
            console_handler.setLevel(level)
            logger.addHandler(console_handler)
        if "file" in user_config.get(
            "logging", "output", fallback=base_config.get("logging", "output")
        ):
            filename = user_config.get(
                "log_file", "filename", fallback=base_config.get("log_file", "filename")
            )
            filename = os.path.join(os.getcwd(), filename)
            filemode = user_config.get(
                "log_file", "filemode", fallback=base_config.get("log_file", "filemode")
            )
            level = user_config.get(
                "logging",
                "file_log_level",
                fallback=base_config.get("logging", "file_log_level"),
            )
            log_file_handler = logging.FileHandler(filename, mode=filemode)
            log_file_handler.setLevel(level)
            log_file_handler.setFormatter(formatter)
            logger.addHandler(log_file_handler)
    return logger
