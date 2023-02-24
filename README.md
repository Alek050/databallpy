# databallpy

A package for loading, preprocessing, vizualising and synchronizing soccere event and tracking data.

This package is developed to create a standardized way to analyse soccer matches using both event- and tracking data. Other packages, like [kloppy](https://github.com/PySport/kloppy) and [floodligth](https://github.com/floodlight-sports/floodlight), already standardize the import of data sources. The current package goes a step further in combining different data streams from the same match. In this case, the `Match` object combines information from the event and tracking data.

We are currently working on adding more data sources and on creating a `Match.synchronise_tracking_and_event_data()` function to efficiently align all events with a timeframe in the tracking data. This would make it possible to get contextual information from the tracking data at the exact moment that the event is taking place.

## Installation

```bash
$ pip install databallpy
```

## Usage

The package is centered around the `Match` object. A `Match` has tracking data, event data metadata about the match.

```console
$ from databallpy.match import get_match, get_open_match
$
$ match = get_match(
$   tracking_data_loc="data/tracking_data.dat",
$   tracking_metadata_loc="data/tracking_metadata.xml",
$   tracking_data_provider="tracab"
$   event_data_loc="data/event_data_f24.xml",
$   event_metadata_loc="data/event_metadata_f7.xml",
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

See [the documentation](https://databallpy.readthedocs.io/en/latest/autoapi/databallpy/match/index.html) of the `Match` object for more options. Note that this package is developed to combine event and tracking data, therefore both datastreams are necessary to create a `Match` object.

## Visualizing

The packages also provides tools to visualise the data. Note that to save a match clip the package relies on the use of ffmpeg. Make sure to have installed it to your machine and added it to your python path, otherwise the `save_match_clip()` function will produce an error.

```console
$ from databallpy.match import get_match, get_open_match
$ from databallpy.visualize import save_match_clip
$
$ match = get_match(
$   tracking_data_loc="data/tracking_data.dat",
$   tracking_metadata_loc="data/tracking_metadata.xml",
$   tracking_data_provider="tracab"
$   event_data_loc="data/event_data_f24.xml",
$   event_metadata_loc="data/event_metadata_f7.xml",
$   event_data_provider="opta",
$ )
$
$ # or to load an open metrica dataset of tracking and event data
$ match = get_open_match()
$
$ save_match_clip(match, start_idx=0, end_idx=100, folder_loc="data", title="example")
```

This function will save a .mp4 file in `"data/"` directory of the `match.tracking_data` from index 0 untill index 100.

https://user-images.githubusercontent.com/49450063/217216924-748841e3-dddd-4149-a3c5-000320a91865.mp4

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



