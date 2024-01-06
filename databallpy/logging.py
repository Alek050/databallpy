import logging
from configparser import ConfigParser
import os


def create_logger(name:str) -> logging.Logger:
    """Creates a logger with the given name, using the configuration files in the 
    databallpy package and the current working directory.

    Args:
        name (str): The name of the logger to create.

    Returns:
        logging.Logger: The logger with the given name.
    """
    base_config_path = os.path.join(os.path.dirname(__file__), 'base_logging_config.ini')
    base_config = ConfigParser()
    base_config.read(base_config_path)
    user_config_path = os.path.join(os.getcwd(), 'databallpy_config.ini')
    user_config = ConfigParser()
    user_config.read(user_config_path)

    log_format = user_config.get('logging', 'format', fallback=base_config.get('logging', 'format'))
    log_level = user_config.get("logging", "log_level", fallback=base_config.get("logging", "log_level"))
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    if 'console' in user_config.get('logging', 'output', fallback=base_config.get('logging', 'output')):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        level = user_config.get('logging', 'console_log_level', fallback=base_config.get('logging', 'console_log_level'))
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    if 'file' in user_config.get('logging', 'output', fallback=base_config.get('logging', 'output')):
        filename = user_config.get('log_file', "filename", fallback=base_config.get('log_file', "filename"))
        filename = os.path.join(os.getcwd(), filename)
        filemode = user_config.get('log_file', "filemode", fallback=base_config.get('log_file', "filemode"))
        level = user_config.get('logging', 'file_log_level', fallback=base_config.get('logging', 'file_log_level'))
        log_file_handler = logging.FileHandler(filename, mode=filemode)
        log_file_handler.setLevel(level)
        log_file_handler.setFormatter(formatter)
        logger.addHandler(log_file_handler)
    return logger

