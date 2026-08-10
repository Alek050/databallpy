from importlib.metadata import version

from databallpy.utils.get_game import (
    Game,
    get_game,
    get_game_from_kloppy,
    get_open_game,
    get_saved_game,
)
from databallpy.utils.logging import create_logger
from databallpy.utils.to_xml import Event, LabelDict, events_to_xml

__version__ = version("databallpy")
create_logger(__name__).info(f"Using databallpy version {__version__}")
