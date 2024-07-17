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

## Use Cases of Event Data

Event data is the most available data type in football analytics. It is used for a wide range of purposes. Mostly, it is used for descriptive analysis which helps to answer the following questions:

1. How many (successful) passes did a player make?

2. How many shots did a team take and from where?

3. Which player tends to lose the ball the most?

For years, answers to these questions were extremely valuable and gave new insights into football. For instance, it was also used to compute the first simple expected goals models. This lead to a decrease in the number of shots taken from long distances and an increase in the number of shots taken from close range. However, event data also has some short commings. Since it is often collected semi-automatically, it is not always 100% accurate. On top of that, it is often collected from broadcast footage, which means that events that happen during replays (like goal kicks) are often missed. Lastly, and probably the most important setback, is that event data does not provide any information about the context in which the event happened. For instance, shooting from the penalty spot when the goal is empty is a lot easier than when there are 5 defenders in front of you. This is generally not collected in event data (with the exception of freeze-frame data). This is where tracking data comes in.

