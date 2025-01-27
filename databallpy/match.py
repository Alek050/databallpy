from databallpy.game import Game
from databallpy.utils.warnings import deprecated


@deprecated(
    "The `Match` class is deprecated and will removed in version 0.8.0. Please use the `Game` class instead."
)
class Match(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
