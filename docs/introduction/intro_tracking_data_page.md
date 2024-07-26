# Intro to Tracking Data

## The Raw Tracking Data

Tracking data contains the positions of all players and the ball on the pitch at a certain frequency, usually between 10 and 25 Hz. For matches, the most common way to collect the data is via cameras. Video footage is analyzed by deep learning models that track the players and the ball. Other options for obtaining the data are from LPM (local positioning systems), GPS (global positioning systems), or radio based devices. The data is often provided in text-like files (`.txt`, `.dat`). In this instance, we will utilize a Tracab file, although the majority of tracking data files exhibit similar formats. The unprocessed tracking data will typically resemble the following example:

```
1802276:-1,1,-1,12,-3590,0.00;1,2,15,553,-1529,0.54;3,3,2,-1651,3487,0.03;
0,4,10,-234,-1661,0.32;1,5,8,838,588,0.28;1,6,37,2339,496,0.00;0,7,9,-27,-944,
0.00;3,8,0,317,-1222,0.96;0,9,33,-1607,-1,9,0.34;0,10,18,-669,-618,0.09;0,11,
22,-689,1082,0.03;0,12,14,-1403,1624,0.29;0,13,8,-999,224,0.3;1,14,10,-19,-48,
0.06;1,15,18,-7,-1927,0.35;0,16,26,-29,1391,0.06;1,17,21,17,2228,0.34;1,18,19,
103,2186,0.87;1,19,12,2193,-583,0.52;0,20,17,-1146,-1405,0.66;1,21,9,38,941,
0.23;1,22,7,703,-634,.0;0,23,24,-1635,868,0.06;3,24,1,2330,-3431,0.00;0,25,1,
-5235,6,0.09;-1,26,-1,42,946,0.00;1,27,0,4528,24,0.62;4,28,-1,5550,4400,0.00;
4,29,-1,5550,4400,0.00;:-19,-48,0,0.04,A,Dead;:
1802277:-1,1,-1,12,-3590,0.00;1,2,15,551,-1527,0.55;3,3,2,-1649,3487,0.06;0,4,
10,-235,-1660,0.35;1,5,8,838,588,0.28;1,6,37,2339,496,0.00;0,7,9,-27,-944,0.00;
3,8,0,315,-1221,0.92;0,9,33,-1606,-1,0,0.34;0,10,18,-669,-619,0.03;0,11,22,-689,
1081,0.00;0,12,14,-1403,1623,0.18;0,13,8,-997,227,0.3;1,14,10,-19,-49,0.03;1,15,
18,-8,-1927,0.35;0,16,26,-28,1390,0.03;1,17,21,17,2227,0.34;1,18,19,1,01,2184,
0.87;1,19,12,2194,-584,0.52;0,20,17,-1147,-1404,0.66;1,21,9,38,940,0.22;1,22,7,
703,-634,.00;0,23,24,-1636,868,0.00;3,24,1,2330,-3431,0.00;0,25,1,-5235,4,0.19;
-1,26,-1,43,945,0.00;1,27,0,4530,25,0.61;4,28,-1,5550,4400,0.00;4,29,-1,5550,
4400,0.00;:-19,-49,0,0.05,A,Dead;:

``` 

Please note that this represents only 2 lines of tracking data, while standard providers offer up to 200,000 lines per match. As you can observe, the current format is difficult to interpret. It lacks column names and employs three different separators (`:`, `;`, and `,`). This variation is in line with the FIFA's implementation of a [standard format for tracking data](https://www.fifa.com/technical/football-technology/standards/epts/research-development-epts-standard-data-format). Essentially, the FIFA mandates that all details pertaining to the raw tracking data file should be included in the metadata file (kudos to FIFA for that!). Moreover, the standard also specifies the separators to be used. The `:` serves to differentiate the type of information, often appearing as follows:

`frame_count:player_info:ball_info:`

The `;` denotes the start of a new instance within the current information type. For instance, within the `player_info` section:

`player1;player2;player3`

The `,` is employed to separate the specific information within a given instance. For instance, for `player1`:

`team_id,channel_id,shirt_number,x,y,velocity`

It is crucial to comprehend how the data is collected and how it corresponds to the raw data. Tracking data in stadiums is typically gathered using camera systems, with machine learning used to translate the video data into positions. At the beginning of a match, an individual must match each player in the camera feed with their respective names since the model knows the players' locations but not their identities. Thus, before the match begins, someone links player names to specific locations on the pitch, enabling the model to establish the connection between location and player. Unfortunately, there are instances where the model loses track of which location corresponds to which player. This often occurs when players celebrate a goal, as they gather closely together, rendering it impossible to discern their identities in the video data. Consequently, the model requests manual validation once again. By examining the data, one can understand that `player1` does not always refer to the same player. For instance, if two players become misplaced, and the manual labeler modifies the order, player 2 may end up being assigned to the location intended for player 1, and vice versa. Thus, it might appear as `player2;player1;player3`. Fortunately, the `shirt_number` serves as a means of validating the correct player. Hence, it is preferable to refer to the player information segments as `channel1` and `channel2` rather than `player1` and `player2`. These channels represent the link between a player and their tracked position during a specific period.

Some tracking data providers do not include the shirt number in the raw tracking data. Therefore, to identify the player data, one must rely on the metadata. Typically, this metadata is provided in either `.xml` or `.json` format, both of which are semi-structured data types. The metadata file contains all relevant information about the match, such as player details, date, and location. By combining the raw metadata and tracking data files, we can extract all the available information. Depending on the tracking data provider, the raw data files may exhibit slight variations, with some providers offering more information than others. However, in general, the tracking data files primarily consist of tracking data, periods, substitutes, and the date of play. Information such as the score and goals is typically found only in event data.

## Processed Tracking Data

Processed tracking data is a tabular representation of the raw tracking data. Each row represents a single frame, and each column refers to a specific player or the ball. The columns contain information such as the player's position, velocity, and acceleration. The processed tracking data is often accompanied by a metadata file that provides additional information about the match. The processed tracking data is easier to work with than the raw tracking data, as it is in a clear tabular format. Below is an example of processed tracking data:

| frame | ball_x | ball_y | ball_status | ball_possession | home_1_x | home_1_y | ... | away_11_vx | away_11_vy | away_11_velocity |
|-------|:------:|:------:|:-----------:|:---------------:|:--------:|:--------:|:---:|:----------:|:----------:|:----------------:|
| 0     | 0.0    | 0.0    | dead        | home            | 10.0     | -3.2     | ... | 2.2        | -0.1       | 2.3              |
| 1     | 0.1    | 0.2    | alive       | home            | 10.1     | -3.1     | ... | 2.3        | -0.2       | 2.4              |
| 2     | 0.2    | 0.4    | alive       | home            | 10.2     | -3.0     | ... | 2.4        | -0.3       | 2.5              |

If we visualize this, it will look like this:

```{image} ../static/tracking_data_example.png
:alt: tracking_data_example
:width: 800px
:align: center
```

## Use Cases of Tracking Data

The first use cases for tracking data were mainly focused on the physical aspects of the game. For instance, tracking data was used to analyze the distance covered by players, their speed, and the number of sprints. However, as tracking data became more prevalent, analysts began to explore more advanced use cases. Tracking data is now used for a wide range of purposes, the most important being adding context to events:

1. Improving expected goals models

2. Analyse pitch control

3. Assessing which players are available for a pass

4. Evaluating the effectiveness of pressing, and which players are least influenced when pressured

5. Evaluating the effectiveness of a team's defensive shape

The most significant advantage of tracking data is that it provides context to events. However, this requires synchronisation of the tracking and the event data, which is, unfortunately, not as easy as it looks like. The main challenge is that the exact time points of the event and tracking data do not align, neither to the exact locations of the events. Different methods are proposed so synchronise tracking and event data. One of the key features of this package is this synchronisation of the tracking and the event data. But before we get there, lets first dive into the basics of the package.
