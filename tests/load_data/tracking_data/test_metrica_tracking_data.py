import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data.metrica_tracking_data import (
    _get_tracking_data,
    load_metrica_open_tracking_data,
    load_metrica_tracking_data,
)

TD_RAW = """1:0.9,0.5;0.1,0.6;0.3,0.4:NaN,NaN
2:0.8,0.5;0.2,0.5;0.4,0.9:0.5,0.5
3:0.7,0.4;0.3,0.4;0.5,0.7:0.6,0.5
5:0.6,0.5;0.5,0.6;0.6,0.6:NaN,NaN
6:0.5,0.1;0.7,0.5;0.7,0.5:0.2,0.9"""
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


class TestMetricaTrackingData(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.td_loc = "tests/test_data/metrica_tracking_data_test.txt"
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
        self.expected_td_channels = pd.DataFrame(
            {
                "start": [1, 4],
                "end": [3, 6],
                "ids": [
                    ["home_1", "home_11", "away_34"],
                    ["home_11", "away_34", "away_35"],
                ],
            }
        )
        self.expected_td = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "ball_x": [np.nan, 0, 10, np.nan, np.nan, -30],
                "ball_y": [np.nan, 0, 0, np.nan, np.nan, 20],
                "ball_z": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "ball_status": ["dead", "alive", "alive", np.nan, "dead", "alive"],
                "ball_posession": [None, None, None, None, None, None],
                "home_1_x": [40, 30, 20, np.nan, np.nan, np.nan],
                "home_1_y": [0, 0, -5, np.nan, np.nan, np.nan],
                "home_11_x": [-40, -30, -20, np.nan, 10, 0],
                "home_11_y": [5, 0, -5, np.nan, 0, -20],
                "away_34_x": [-20, -10, 0, np.nan, 0, 20],
                "away_34_y": [-5, 20, 10, np.nan, 5, 0],
                "away_35_x": [np.nan, np.nan, np.nan, np.nan, 10, 20],
                "away_35_y": [np.nan, np.nan, np.nan, np.nan, 5, 0],
                "matchtime_td": ["00:00", "00:00", "00:01", "00:01", "45:00", "45:00"],
            }
        )

    def test_get_tracking_data(self):
        expected = self.expected_td.copy()
        expected.drop(["matchtime_td"], axis=1, inplace=True)
        res = _get_tracking_data(
            self.td_loc, self.expected_td_channels, [100, 50], verbose=False
        )
        pd.testing.assert_frame_equal(res, expected)

    def test_load_metrica_tracking_data(self):
        res_td, res_md = load_metrica_tracking_data(
            self.td_loc, self.md_loc, verbose=False
        )
        pd.testing.assert_frame_equal(res_td, self.expected_td)
        assert res_md == self.expected_metadata

        with self.assertRaises(TypeError):
            load_metrica_tracking_data(22, self.md_loc, verbose=False)
        with self.assertRaises(AssertionError):
            load_metrica_tracking_data(
                "some_wrong_string.txt", self.md_loc, verbose=False
            )

    @patch("requests.get", side_effect=[Mock(text=TD_RAW), Mock(text=MD_RAW)])
    def test_load_metrica_open_tracking_data(self, _):
        td, md = load_metrica_open_tracking_data()
        assert md == self.expected_metadata
        pd.testing.assert_frame_equal(td, self.expected_td)
