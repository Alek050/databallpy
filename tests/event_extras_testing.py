"""Helper to check the extra information of the shot, pass and dribble events that
the event data parsers return next to the event data."""

import pandas as pd

from databallpy.utils.event_extras import (
    EVENT_TYPES,
    EXTRA_COLUMNS,
    get_event_extras_from_dataframe,
)
from tests.pandas_testing import assert_frame_equal


def to_event_extras(
    shot_events: pd.DataFrame | None = None,
    pass_events: pd.DataFrame | None = None,
    dribble_events: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Get the expected extras from dataframes with all attributes of the events."""
    dfs = {"shot": shot_events, "pass": pass_events, "dribble": dribble_events}
    return {
        f"{event_type}_events": get_event_extras_from_dataframe(df, event_type)
        for event_type, df in dfs.items()
    }


def assert_event_extras_equal(
    result: dict[str, pd.DataFrame], expected: dict[str, pd.DataFrame]
) -> None:
    """Assert that the values of the extras are equal, ignoring the row order."""
    assert set(result.keys()) == set(expected.keys())
    for key in expected:
        res = result[key].sort_values("event_id").reset_index(drop=True)
        exp = expected[key].sort_values("event_id").reset_index(drop=True)
        if len(res) == 0 and len(exp) == 0:
            assert list(res.columns) == list(exp.columns), key
            continue
        assert_frame_equal(res, exp, check_dtype=False, obj=key)


def assert_event_extras(
    extras: dict[str, pd.DataFrame],
    event_data: pd.DataFrame,
    event_types: list[str] = EVENT_TYPES,
) -> None:
    """Assert that the extras have the expected columns, hold all events of the event
    data with the matching databallpy event, and only hold events of the event data
    (own goals for instance have shot extras, but no databallpy event)."""
    event_data = pd.DataFrame(event_data)
    for event_type in EVENT_TYPES:
        df = extras[f"{event_type}_events"]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == EXTRA_COLUMNS[event_type]
        extra_ids = set(df["event_id"].tolist())
        if event_type not in event_types or len(event_data) == 0:
            assert extra_ids == set(), event_type
            continue
        expected_ids = set(
            event_data.loc[
                event_data["databallpy_event"] == event_type, "event_id"
            ].tolist()
        )
        assert expected_ids <= extra_ids, event_type
        assert extra_ids <= set(event_data["event_id"].tolist()), event_type
