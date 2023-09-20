ED_METRICA_RAW = """{
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
		},
    {
			"index": 6,
			"team": {
				"name": "Team A",
				"id": "FIFATMA"
			},
			"type": {
				"name": "SHOT",
				"id": 2
			},
			"subtypes": [
				{
					"name": "OFF TARGET",
					"id": 29
				},
				{
					"name": "OUT",
					"id": 31
				}
			],
			"start": {
				"frame": 7,
				"time": 76.08,
				"x": 0.7,
				"y": 0.6
			},
			"end": {
				"frame": 8,
				"time": 76.36,
				"x": 0.1,
				"y": 0.5
			},
			"period": 2,
			"from": {
				"name": "Player 1",
				"id": "P3568"
			},
			"to": null
		},
    {
			"index": 7,
			"team": {
				"name": "Team A",
				"id": "FIFATMA"
			},
			"type": {
				"name": "SHOT",
				"id": 2
			},
			"subtypes":
				{
					"name": "GOAL",
					"id": 30
				},
			"start": {
				"frame": 7,
				"time": 76.08,
				"x": 0.7,
				"y": 0.6
			},
			"end": {
				"frame": 8,
				"time": 76.36,
				"x": 0.1,
				"y": 0.5
			},
			"period": 2,
			"from": {
				"name": "Player 1",
				"id": "P3568"
			},
			"to": null
		},
    {
			"index": 8,
			"team": {
				"name": "Team A",
				"id": "FIFATMA"
			},
			"type": {
				"name": "SHOT",
				"id": 2
			},
			"subtypes": [
				{
					"name": "ON TARGET",
					"id": 33
				},
				{
					"name": "GOAL",
					"id": 30
				}
			],
			"start": {
				"frame": 7,
				"time": 76.08,
				"x": 0.7,
				"y": 0.6
			},
			"end": {
				"frame": 8,
				"time": 76.36,
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

MD_METRICA_RAW = """<?xml version="1.0" encoding="utf-8"?>
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
      <PlayerChannel channelId="x" id="player11_x" playerId="P3568"/>
      <PlayerChannel channelId="y" id="player11_y" playerId="P3568"/>
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

TD_METRICA_RAW = """1:0.5,0.5;0.1,0.6;0.7,0.4:NaN,NaN
2:0.8,0.5;0.2,0.5;0.4,0.9:0.5,0.5
3:0.7,0.4;0.3,0.4;0.5,0.7:0.9,0.1
5:0.4,0.5;0.5,0.6;0.6,0.6:NaN,NaN
6:0.5,0.1;0.7,0.5;0.7,0.5:0.1,0.9"""
