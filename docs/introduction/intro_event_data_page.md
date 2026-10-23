# Intro to Event Data

## The raw data

Event data is also called notational analysis data. It is the most common data type in football analytics. Event data is collected by human operators who watch the game and record events as they happen. This data is then used to answer questions about the game. Event data is often collected by companies like Opta, StatsBomb, or Wyscout. Nowedays, the collections process is done semi-automatically, but humans are still in place to tag certain events. The data is collected in real-time and is often available within minutes after the game has ended.

Event data refers to the events occurring on the pitch, such as passes, shots, dribbles, fouls, and goals. Usually, event data is accompanied by an event data file and a metadata file, both of which are in `.xml` or `.json` format. Generally, event data is presented either in string format (`event="Successful Pass"`) or as numeric IDs (`event_type_id=3`), where the number `3` represents a specific event, like a pass. Taking Opta as an example, event data might appear as follows:

```xml
<Event id="2529876573" event_id="3" type_id="1" period_id="1" min="0" sec="0" 
player_id="234961" team_id="425" outcome="1" x="50.0" y="50.0"
timestamp="2023-04-07T19:01:36.201" timestamp_utc="2023-04-07T18:01:36.201"
last_modified="2023-04-07T19:01:40" version="1680890500351">
   <Q id="4170351507" qualifier_id="212" value="10.7" />
   <Q id="4170351509" qualifier_id="213" value="3.9" />
   <Q id="4170351503" qualifier_id="140" value="42.3" />
   <Q id="4170351511" qualifier_id="279" value="S" />
   <Q id="4170351505" qualifier_id="141" value="39.8" />
   <Q id="4170351501" qualifier_id="56" value="Back" />
</Event>
```

This data provides information about the event, its outcome, timing, and location. Some event data providers also include qualifiers, as identified within the `Q` instances. For example, `qualifier_id="56" value="Back"` indicates that this pass is directed backward. However, understanding the meaning of `type_id` and `qualifier_id` requires translation, which is why we cannot parse data from certain providers like Ortec, as we lack knowledge about the specific IDs' meanings.

## Processed Event Data

Processed event data is actually very simple. It is in a clear tabular format. Each row represents a single event, and eacht column refers to a qualifier. Not all qualifiers provided by the event data provider are as interesting, but in the very least you get the type of event, the player involved, the outcome, the start location, and the timing of the event. See the example below.

| event_id | minutes | seconds | x    | y     | event   | outcome | player   |
|----------|:-------:|:-------:|:----:|:-----:|:-------:|:-------:|:--------:|
| 121      | 23      | 51.05   | 33.2 | -10.1 | pass    | 1       | Sneijder |
| 122      | 23      | 58.20   | 48.1 | -22.3 | take-on | 1       | Robben   |
| 123      | 24      | 2.88    | 52.1 | -20.5 | cross   | 1       | Robben   |

So to visualize the second row, could see something like this:

```{image} ../static/event_data_dribble_robben.png
:alt: Dribble by Robben
:width: 400px
:align: center
```

## SPADL

Every event data provider uses its own vocabulary, which makes it hard to combine data of different providers or to reuse models. The Soccer Player Action Description Language ([SPADL](https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html), Decroos et al., 2019) solves this by describing every on-ball action in the same way: a type (e.g. `pass`, `cross`, `shot`, `dribble`, `tackle`), a result (e.g. `success`, `fail`, `offside`), a body part, and a start and end location. SPADL is used by many models, such as VAEP and xT in [socceraction](https://github.com/ML-KULeuven/socceraction).

DataBallPy maps the events of all providers to the SPADL vocabulary when the event data is loaded, in the `action_type`, `action_result` and `action_bodypart` columns of the event data. Events that are not an on-ball action, like substitutions, have no `action_type`. To get the SPADL actions of a game, use:

```python
from databallpy import get_open_game

game = get_open_game(provider="metrica")
actions = game.to_spadl(playing_direction="home_oriented", coordinate_system="spadl")
```

The output has the same columns and ids as socceraction, and the conversion follows the same rules: clearances end where the next action starts, and dribbles are added between consecutive actions of the same team when the ball moved without an event in between. The `type_id`, `result_id` and `bodypart_id` columns map to the names in `databallpy.utils.constants.SPADL_ACTION_TYPES`, `SPADL_RESULTS` and `SPADL_BODYPARTS`, which are also given in the `type_name`, `result_name` and `bodypart_name` columns. The `original_event_id` and `event_id` columns link each action to the event data, they are empty for the added dribbles.

The `playing_direction` sets which team plays from left to right:
- `home_oriented` (default): the home team.
- `away_oriented`: the away team.
- `possession_oriented`: the team performing the action, like `socceraction.spadl.play_left_to_right`.
- `out_of_possession_oriented`: the team not performing the action.

The `coordinate_system` sets the units of the locations:
- `databallpy` (default): meters on the actual pitch, with (0, 0) at the centre spot.
- `spadl`: the 105x68 SPADL field, with (0, 0) in the bottom left corner, as used by socceraction.

```{note}
The mapping to SPADL follows socceraction as closely as possible, but the native parsers of DataBallPy use all information of the provider. Therefore, the actions can differ slightly from socceraction, for instance, the native Metrica parser also maps recoveries and challenges to interceptions and tackles, and skips carries in which the ball did not move. Data loaded with the Kloppy integration is mapped like the socceraction Kloppy converter. SPADL is not supported for InStat data.
```

### Atomic-SPADL

[Atomic-SPADL](https://socceraction.readthedocs.io/en/latest/documentation/spadl/atomic_spadl.html) (Decroos et al., 2020) splits every SPADL action into atomic actions without a result. A pass is followed by a `receival`, `interception`, `out` or `offside`, a shot by a `goal` or `out`, and a foul with a card by a `yellow_card` or `red_card`. Instead of a start and end location, every atomic action has a location (`x`, `y`) and a displacement (`dx`, `dy`). To get the Atomic-SPADL actions of a game, use:

```python
actions = game.to_atomic_spadl(playing_direction="home_oriented", coordinate_system="spadl")
```

The `playing_direction` and `coordinate_system` options are the same as for `to_spadl`, in the `databallpy` coordinate system only `x` and `y` are moved to the centre spot. The `type_id` maps to the names in `databallpy.utils.constants.ATOMIC_SPADL_ACTION_TYPES`, the `bodypart_id` to `SPADL_BODYPARTS`, which are also given in the `type_name` and `bodypart_name` columns. Atomic-SPADL has no result, so there is no `result_name` column.

The conversion follows `socceraction.atomic.spadl.convert_to_atomic`, including the extra dribbles after a receival or interception when the ball moved before the next action, with a few fixes:
- There is a single `interception` type, socceraction has two with a different id.
- `corner_crossed`, `corner_short`, `freekick_crossed` and `freekick_short` are not in the vocabulary, since they are always simplified to `corner` and `freekick`. Therefore, the type ids differ from socceraction.
- A pass followed by a corner of the other team is `out`, not an interception.
- `shot_freekick` is kept as a shot type, socceraction simplifies it to `freekick`.
- No dribble is added from an `out` to the throw in that follows it.

## Use Cases of Event Data

Event data is the most available data type in football analytics. It is used for a wide range of purposes. Mostly, it is used for descriptive analysis which helps to answer the following questions:

1. How many (successful) passes did a player make?

2. How many shots did a team take and from where?

3. Which player tends to lose the ball the most?

For years, answers to these questions were extremely valuable and gave new insights into football. For instance, it was also used to compute the first simple expected goals models. This lead to a decrease in the number of shots taken from long distances and an increase in the number of shots taken from close range. However, event data also has some short commings. Since it is often collected semi-automatically, it is not always 100% accurate. On top of that, it is often collected from broadcast footage, which means that events that happen during replays (like goal kicks) are often missed. Lastly, and probably the most important setback, is that event data does not provide any information about the context in which the event happened. For instance, shooting from the penalty spot when the goal is empty is a lot easier than when there are 5 defenders in front of you. This is generally not collected in event data (with the exception of freeze-frame data). This is where tracking data comes in.

