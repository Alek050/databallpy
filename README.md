# databallpy

A package for loading, preprocessing, vizualising and synchronizing soccer event- and tracking data.

This package is developed to create a standardized way to analyse soccer matches using both event- and tracking data. Other packages, like [kloppy](https://github.com/PySport/kloppy) and [floodlight](https://github.com/floodlight-sports/floodlight), already standardize the import of data sources. The current package goes a step further in combining different data streams from the same match. In this case, the `Match` object combines information from the event and tracking data.

### Changelog for version 0.2.0

- Added parser for Metrica, including an open dataset
- Added functionality to synchronize tracking and event data
- Added functionality to plot events
- Fixed bug, now both tracking and event data are normalized in direction
- Fixed unexpected behaviour, all date related objects are now datetime objects

### Planned changes

- Adding different filters to filter the tracking data
- Make the `Match` object more accesable if you don't have tracking or event data
- Adding features to quantify pressure, ball possession, etc. (if you have any ideas/requests, please open an issue!)
- Adding expected passing and goals models

## Installation

```bash
$ pip install databallpy
```

## Usage

The package is centered around the `Match` object. A `Match` has tracking data, event data metadata about the match.
For a more elaborate example, see the [example file](https://databallpy.readthedocs.io/en/latest/example.html)

```console
$ from databallpy.match import get_match, get_open_match
$
$ match = get_match(
$   tracking_data_loc="../data/tracking_data.dat",
$   tracking_metadata_loc="../data/tracking_metadata.xml",
$   tracking_data_provider="tracab"
$   event_data_loc="../data/event_data_f24.xml",
$   event_metadata_loc="../data/event_metadata_f7.xml",
$   event_data_provider="opta",
$ )
$
$ # or to load an open metrica dataset of tracking and event data
$ match = get_open_match()
$
$ match.home_team_name # the team name of the home playing team
$ match.away_players # pandas dataframe with the names, ids, shirt numbers and positions of the away team
$ match.tracking_data # pandas dataframe with tracking data of the match
$ match.event_data # pandas dataframe with event data of the match
```

See [the documentation](https://databallpy.readthedocs.io/en/latest/autoapi/databallpy/match/index.html) of the `Match` object and the [example usage](https://databallpy.readthedocs.io/en/latest/example.html) for more options. Note that this package is developed to combine event and tracking data, for now both datastreams are necessary to create a `Match` object.

## Synchronization of tracking and event data

Tracking and event data is often poorly synchronized. For instance, when taking the event data of Opta and tracking data of Tracab, and take the timing of the events to synchronize tracking and event data we get something like this:
git

[![Not synced tracking and events](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://user-images.githubusercontent.com/49450063/224354460-6dc45ecb-4774-43b5-aba5-d7b9f32c908f.mp4)


As you can see, the timing (and placing) of the events do not correspond good with the tracking data locations. Using the methodology of [this](https://kwiatkowski.io/sync.soccer) article, this package is able to synchronize tracking and event data using the Needleman-Wunsch algorithm. 

After running the following command, the events are better synchronized to the tracking data:

```batch
$ match.synchronise_tracking_and_event_data()
```

[![Synced tracking and events](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://user-images.githubusercontent.com/49450063/224354505-d9feece7-2ab4-4f97-a6d9-73461e6789a8.mp4)

## Documentation

The official documentation can be found [here](https://databallpy.readthedocs.io/en/latest/autoapi/databallpy/index.html).

## Providers

For now we limited providers. We are planning on adding more providers later on.

Event data providers:
- Opta
- Metrica

Tracking data providers:
- Tracab
- Metrica

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`databallpy` was created by Alexander Oonk & Daan Grob. It is licensed under the terms of the MIT license.

## Similar projects

Although we think this package helps when starting to analyse soccer data, other packages may be better suited for your specific needs. Make sure to check out the following packages as well:
- [kloppy](https://github.com/PySport/kloppy)
- [floodlight](https://github.com/floodlight-sports/floodlight)
- [codeball](https://github.com/metrica-sports/codeball)

And for a more specific toturials on how to get started with soccer data"
- [Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD)



