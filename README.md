# databallpy

A package for loading, preprocessing, vizualising and synchronizing soccere event aand tracking data.

## Installation

```bash
$ pip install databallpy
```

## Usage

The package is centered around the `Match` object. A `Match` has tracking data, event data metadata about the match.

```console
$ from databallpy.match import get_match
$
$ match = get_match(
    tracking_data_loc="data/tracking_data.dat",
    tracking_metadata_loc="data/"tracking_metadata.xml",
    tracking_data_provider="tracab"
    event_data_loc="data/event_data_f24.xml",
    event_metadata_loc="data/event_metadata_f7.xml",
    event_data_provider="opta",
)
$ match.home_team_name # the team name of the home playing team
$ match.away_players # pandas dataframe with the names, ids, shirt numbers and positions of the away team
$ match.tracking_data # pandas dataframe with tracking data of the match
$ match.event_data # pandas dataframe with event data of the match
```

See the documentation of the `Match` object for more options. Note that this package is developed to combine event and tracking data, therefore both datastreams are necessary to create a `Match` object.

## Providers
For now we only have one tracking data and one event data provider. We are planning on adding more providers later on.

Event data providers:
1. Opta

Tracking data providers:
1. Tracab

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`databallpy` was created by Alexander Oonk & Daan Grob. It is licensed under the terms of the MIT license.
