import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.metrica_event_data import (
    _get_event_data,
    load_metrica_event_data,
    load_metrica_open_event_data,
)
from databallpy.load_data.metadata import Metadata

ED_RAW = """{
	"data": [
		{
			"index": 3,
			"team": {
				"name": "Team A",
				"id": "FIFATMA"
			},
			"type": {
				"name": "SET PIECE",
				"id": 5
			},
			"subtypes": {
				"name": "KICK OFF",
				"id": 35
			},
			"start": {
				"frame": 1,
				"time": 14.44,
				"x": null,
				"y": null
			},
			"end": {
				"frame": 1,
				"time": 14.44,
				"x": null,
				"y": null
			},
			"period": 1,
			"from": {
				"name": "Player 11",
				"id": "P3578"
			},
			"to": null
		},
		{
			"index": 4,
			"team": {
				"name": "Team B",
				"id": "FIFATMB"
			},
			"type": {
				"name": "PASS",
				"id": 1
			},
			"subtypes": null,
			"start": {
				"frame": 3,
				"time": 64.22,
				"x": 0.5,
				"y": 0.4
			},
			"end": {
				"frame": 3,
				"time": 65.08,
				"x": 0.3,
				"y": 0.2
			},
			"period": 1,
			"from": {
				"name": "Player 34",
				"id": "P3699"
			},
			"to": {
				"name": "Player 35",
				"id": "P3700"
			}
		},
		{
			"index": 5,
			"team": {
				"name": "Team A",
				"id": "FIFATMA"
			},
			"type": {
				"name": "CARRY",
				"id": 10
			},
			"subtypes": null,
			"start": {
				"frame": 5,
				"time": 75.08,
				"x": 0.7,
				"y": 0.6
			},
			"end": {
				"frame": 6,
				"time": 75.36,
				"x": 0.1,
				"y": 0.5
			},
			"period": 2,
			"from": {
				"name": "Player 1",
				"id": "P3568"
			},
			"to": null
		}
    ]
}"""

MD_RAW = """<?xml version="1.0" encoding="utf-8"?>
<main xmlns:tns="FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS" \
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation=\
        "FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS \
            FIFADataTransferFormatEPTS.xsd">
  <Metadata>
    <GlobalConfig>
      <FileDate>2020-05-08T12:38:28.766Z</FileDate>
      <FileName>game_DEMO_1002_FIFATMA_FIFATMB_MetadataFifaFormat</FileName>
      <TrackingType>Optical</TrackingType>
      <ProviderName>Metrica Sports</ProviderName>
      <FrameRate>2</FrameRate>
      <ProviderGlobalParameters>
        <ProviderParameter>
          <Name>first_half_start</Name>
          <Value>1</Value>
        </ProviderParameter>
        <ProviderParameter>
          <Name>first_half_end</Name>
          <Value>3</Value>
        </ProviderParameter>
        <ProviderParameter>
          <Name>second_half_start</Name>
          <Value>4</Value>
        </ProviderParameter>
        <ProviderParameter>
          <Name>second_half_end</Name>
          <Value>6</Value>
        </ProviderParameter>
        <ProviderParameter>
          <Name>first_extra_half_start</Name>
          <Value/>
        </ProviderParameter>
        <ProviderParameter>
          <Name>first_extra_half_end</Name>
          <Value/>
        </ProviderParameter>
        <ProviderParameter>
          <Name>second_extra_half_start</Name>
          <Value/>
        </ProviderParameter>
        <ProviderParameter>
          <Name>second_extra_half_end</Name>
          <Value/>
        </ProviderParameter>
      </ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions>
      <Session id="9999">
        <SessionType>Match</SessionType>
        <Start>2019-02-21T03:30:07.000Z</Start>
        <MatchParameters>
          <Score idLocalTeam="FIFATMA" idVisitingTeam="FIFATMB">
            <LocalTeamScore>0</LocalTeamScore>
            <VisitingTeamScore>2</VisitingTeamScore>
          </Score>
        </MatchParameters>
        <FieldSize>
          <Width>100</Width>
          <Height>50</Height>
        </FieldSize>
        <Competition id="DEMO">Demo</Competition>
        <ProviderSessionParameters>
          <ProviderParameter>
            <Name>matchday</Name>
            <Value/>
          </ProviderParameter>
        </ProviderSessionParameters>
      </Session>
    </Sessions>
    <Teams>
      <Team id="FIFATMA">
        <Name>Team A</Name>
        <ProviderTeamsParameters>
          <ProviderParameter>
            <Name>attack_direction_first_half</Name>
            <Value>right_to_left</Value>
          </ProviderParameter>
        </ProviderTeamsParameters>
      </Team>
      <Team id="FIFATMB">
        <Name>Team B</Name>
        <ProviderTeamsParameters>
          <ProviderParameter>
            <Name>attack_direction_first_half</Name>
            <Value>left_to_right</Value>
          </ProviderParameter>
        </ProviderTeamsParameters>
      </Team>
    </Teams>
    <Players>
      <Player id="P3578" teamId="FIFATMA">
        <Name>Player 11</Name>
        <ShirtNumber>11</ShirtNumber>
        <ProviderPlayerParameters>
          <ProviderParameter>
            <Name>position_type</Name>
            <Value>Goalkeeper</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_index</Name>
            <Value>0</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_x</Name>
            <Value>0.02</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_y</Name>
            <Value>0.5</Value>
          </ProviderParameter>
        </ProviderPlayerParameters>
      </Player>
      <Player id="P3568" teamId="FIFATMA">
        <Name>Player 1</Name>
        <ShirtNumber>1</ShirtNumber>
        <ProviderPlayerParameters>
          <ProviderParameter>
            <Name>position_type</Name>
            <Value>Right Back</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_index</Name>
            <Value>1</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_x</Name>
            <Value>0.11</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_y</Name>
            <Value>0.9</Value>
          </ProviderParameter>
        </ProviderPlayerParameters>
      </Player>
      <Player id="P3699" teamId="FIFATMB">
        <Name>Player 34</Name>
        <ShirtNumber>34</ShirtNumber>
        <ProviderPlayerParameters>
          <ProviderParameter>
            <Name>position_type</Name>
            <Value>Left Forward (2)</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_index</Name>
            <Value>3</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_x</Name>
            <Value>0.39</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_y</Name>
            <Value>0.35</Value>
          </ProviderParameter>
        </ProviderPlayerParameters>
      </Player>
      <Player id="P3700" teamId="FIFATMB">
        <Name>Player 35</Name>
        <ShirtNumber>35</ShirtNumber>
        <ProviderPlayerParameters>
          <ProviderParameter>
            <Name>position_type</Name>
            <Value>Left Back</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_index</Name>
            <Value>1</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_x</Name>
            <Value>0.11</Value>
          </ProviderParameter>
          <ProviderParameter>
            <Name>position_y</Name>
            <Value>0.1</Value>
          </ProviderParameter>
        </ProviderPlayerParameters>
      </Player>
    </Players>
    <Devices>
      <Device id="device1">
        <Name>Device 1</Name>
        <Sensors>
          <Sensor id="position">
            <Name>Position</Name>
            <Channels>
              <Channel id="x">
                <Name>position_x</Name>
                <Unit>normalized</Unit>
              </Channel>
              <Channel id="y">
                <Name>position_y</Name>
                <Unit>normalized</Unit>
              </Channel>
            </Channels>
          </Sensor>
        </Sensors>
      </Device>
    </Devices>
    <PlayerChannels>
      <PlayerChannel channelId="x" id="player1_x" playerId="P3578"/>
      <PlayerChannel channelId="y" id="player1_y" playerId="P3578"/>
      <PlayerChannel channelId="x" id="player11_x" playerId="P3577"/>
      <PlayerChannel channelId="y" id="player11_y" playerId="P3577"/>
      <PlayerChannel channelId="x" id="player34_x" playerId="P3699"/>
      <PlayerChannel channelId="y" id="player34_y" playerId="P3699"/>
      <PlayerChannel channelId="x" id="player35_x" playerId="P3700"/>
      <PlayerChannel channelId="y" id="player35_y" playerId="P3700"/>
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification endFrame="3" separator=":" startFrame="1">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="player1_x"/>
          <PlayerChannelRef playerChannelId="player1_y"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="player11_x"/>
          <PlayerChannelRef playerChannelId="player11_y"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="player34_x"/>
          <PlayerChannelRef playerChannelId="player34_y"/>
        </SplitRegister>
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
      </SplitRegister>
    </DataFormatSpecification>
    <DataFormatSpecification endFrame="6" separator=":" startFrame="4">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="player11_x"/>
          <PlayerChannelRef playerChannelId="player11_y"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="player34_x"/>
          <PlayerChannelRef playerChannelId="player34_y"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="player35_x"/>
          <PlayerChannelRef playerChannelId="player35_y"/>
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
"""


class TestMetricaEventData(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.ed_loc = "tests/test_data/metrica_event_data_test.json"
        self.expected_metadata = Metadata(
            match_id=9999,
            pitch_dimensions=[100, 50],
            periods_frames=pd.DataFrame(
                {
                    "period": [1, 2, 3, 4, 5],
                    "start_frame": [1, 4, np.nan, np.nan, np.nan],
                    "end_frame": [3, 6, np.nan, np.nan, np.nan],
                    "start_time_td": [
                        pd.to_datetime("2019-02-21T03:30:07.000Z"),
                        pd.to_datetime("2019-02-21T03:30:08.500Z"),
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "end_time_td": [
                        pd.to_datetime("2019-02-21T03:30:08.000Z"),
                        pd.to_datetime("2019-02-21T03:30:09.500Z"),
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                }
            ),
            frame_rate=2,
            home_team_id="FIFATMA",
            home_team_name="Team A",
            home_players=pd.DataFrame(
                {
                    "id": [3578, 3568],
                    "full_name": ["Player 11", "Player 1"],
                    "formation_place": [0, 1],
                    "position": ["Goalkeeper", "Right Back"],
                    "starter": [True, True],
                    "shirt_num": [11, 1],
                }
            ),
            home_formation="1100",
            home_score=0,
            away_team_id="FIFATMB",
            away_team_name="Team B",
            away_players=pd.DataFrame(
                {
                    "id": [3699, 3700],
                    "full_name": ["Player 34", "Player 35"],
                    "formation_place": [3, 1],
                    "position": ["Left Forward (2)", "Left Back"],
                    "starter": [True, False],
                    "shirt_num": [34, 35],
                }
            ),
            away_formation="0001",
            away_score=2,
        )
        self.expected_event_data = pd.DataFrame(
            {
                "event_id": [3, 4, 5],
                "type_id": [5, 1, 10],
                "event": ["set piece", "pass", "carry"],
                "period_id": [1, 1, 2],
                "minutes": [0, 1, 1],
                "seconds": [14.44, 4.22, 15.08],
                "player_id": [3578, 3699, 3568],
                "player_name": ["Player 11", "Player 34", "Player 1"],
                "team_id": ["FIFATMA", "FIFATMB", "FIFATMA"],
                "outcome": [np.nan, np.nan, np.nan],
                "start_x": [np.nan, 0.0, 20.0],
                "start_y": [np.nan, -5.0, 5],
                "to_player_id": [np.nan, 3700, np.nan],
                "to_player_name": [None, "Player 35", None],
                "end_x": [np.nan, -20.0, -40.0],
                "end_y": [np.nan, -15.0, 0.0],
                "td_frame": [1, 3, 5],
                "datetime": [
                    pd.to_datetime("2019-02-21T03:30:07.000Z"),
                    pd.to_datetime("2019-02-21T03:30:08.000Z"),
                    pd.to_datetime("2019-02-21T03:30:09.000Z"),
                ],
            }
        )
        self.expected_event_data["datetime"] = self.expected_event_data[
            "datetime"
        ].astype("object")

    def test_get_event_data(self):
        expected_event_data = self.expected_event_data.copy()
        for col in ["end_x", "start_x"]:
            expected_event_data[col] = (expected_event_data[col] + 50) / 100.0
        for col in ["end_y", "start_y"]:
            expected_event_data[col] = (expected_event_data[col] + 25) / 50.0

        expected_event_data.drop(["datetime"], axis=1, inplace=True)
        ed = _get_event_data(self.ed_loc)
        pd.testing.assert_frame_equal(ed, expected_event_data)

    def test_load_metrica_event_data(self):
        ed, md = load_metrica_event_data(self.ed_loc, self.md_loc)
        assert md == self.expected_metadata
        pd.testing.assert_frame_equal(ed, self.expected_event_data)

        with self.assertRaises(TypeError):
            ed, md = load_metrica_event_data(22, self.md_loc)

    @patch("requests.get", side_effect=[Mock(text=ED_RAW), Mock(text=MD_RAW)])
    def test_load_open_metrica_event_data(self, _):
        ed, md = load_metrica_open_event_data()
        assert md == self.expected_metadata
        pd.testing.assert_frame_equal(ed, self.expected_event_data)
