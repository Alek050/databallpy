import unittest

import numpy as np
import pandas as pd

from tests.expected_outcomes import MD_METRICA, TD_METRICA
from databallpy.load_data.tracking_data._add_periods_to_tracking_data import _add_periods_to_tracking_data

class TestAddPeriodsToTrackingData(unittest.TestCase):
    def test_add_periods_to_tracking_data(self):
        input = TD_METRICA.drop(["period"], axis=1)
        res = input
        res["period"] = _add_periods_to_tracking_data(input["timestamp"], MD_METRICA.periods_frames)

        pd.testing.assert_frame_equal(res, TD_METRICA)
