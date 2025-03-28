
<div align="center">
  <img src="https://github.com/Alek050/databallpy/assets/49450063/56100e87-c680-4dc1-82e5-4aa8fdbc8a34" alt="Logo">
</div>

[version-image]: https://img.shields.io/pypi/v/databallpy?color=006666
[version-url]: https://pypi.org/project/databallpy/
[python-image]: https://img.shields.io/pypi/pyversions/databallpy?color=006666
[python-url]: https://pypi.org/project/databallpy/
[docs-image]: https://readthedocs.org/projects/databallpy/badge/?version=latest
[docs-url]: https://databallpy.readthedocs.io/en/latest/?badge=latest
[cicd-image]: https://github.com/Alek050/databallpy/actions/workflows/linters_and_tests.yml/badge.svg
[cicd-url]: https://github.com/Alek050/databallpy/actions/workflows/linters_and_tests.yml
[codecov-image]: https://codecov.io/gh/Alek050/databallpy/branch/develop/graph/badge.svg?token=MBI3380I0H
[codecov-url]: https://codecov.io/gh/Alek050/databallpy

[filter-data-url]: https://databallpy.readthedocs.io/en/latest/getting_started/preprocessing_options.html#filtering-data
[velocity-acc-url]: https://databallpy.readthedocs.io/en/latest/getting_started/preprocessing_options.html#adding-velocity-and-acceleration
[synchronisation-url]: https://databallpy.readthedocs.io/en/latest/getting_started/synchronisation_page.html
[covered-distance-url]: https://databallpy.readthedocs.io/en/latest/features/covered_distance_page.html
[pressure-url]: https://databallpy.readthedocs.io/en/latest/features/pressure_page.html
[team-possession-url]: https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html#team-possession
[individual_possession-url]: https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html#individual-player-possession-vidal-codina-et-al-2022
[Expected-Goals-url]: https://databallpy.readthedocs.io/en/latest/features/simple_xG_models.html
[Expected-Threat-url]: https://databallpy.readthedocs.io/en/latest/features/xT_models.html
[Voronoi-url]: https://databallpy.readthedocs.io/en/latest/features/space_occupation.html#voronoi-model
[Guassian-space-occupation-url]: https://databallpy.readthedocs.io/en/latest/features/space_occupation.html#gaussian-model-fernandez-born-2018
[visualizations-url]: https://databallpy.readthedocs.io/en/latest/getting_started/visualizations_page.html

[floodlight-url]:https://github.com/floodlight-sports/floodlight
[kloppy-url]:https://github.com/PySport/kloppy
[soccer-sync-url]: https://kwiatkowski.io/sync.soccer
[example-url]: https://databallpy.readthedocs.io/en/latest/example.html
[getting-started-url]: https://databallpy.readthedocs.io/en/latest/getting_started/installation_page.html

[![Latest Version][version-image]][version-url]
[![Python Version][python-image]][python-url]
[![Documentation Status][docs-image]][docs-url]
[![CI/CD Status][cicd-image]][cicd-url]
[![Codecov][codecov-image]][codecov-url]
[![PyPI Downloads](https://static.pepy.tech/badge/databallpy)](https://pepy.tech/projects/databallpy)

# DataBallPy

A package for loading, synchronizing, and analyzing your soccer event- and tracking data.

This package is developed to create a standardized way to analyse soccer games using both event- and tracking data. Other packages, like [kloppy][kloppy-url] and [floodlight][floodlight-url], already standardize the import of data sources. The current package goes a step further in combining different data streams from the same game. In this case, the `Game` object combines information from the event and tracking data. The main current feature is the smart synchronization of the tracking and event data. We utilize the Needleman-Wunch algorithm, inspired by [this article][soccer-sync-url], to align the tracking and even data, while ensuring the order of the events, something that is not done when only using (different) cost functions.

## Final goal for DataBallPy

Although reading in and synchronising data is already very helpfull to get started with your analysis, it's only the first step. Even after this first step, getting your first 'simple' metrics out of the data might be more difficult than anticipated. Therefore, the primary end goal for this package is to create a space where (scientific) soccer metrics are implemented and can be used in a few lines. We even plan to go further and show clear notebooks (to combine text and code) with visualizations for all the features we implement. This way, you will not only get easy access to the features/metrics, but also understand exactly how it is calculated. We hope this will inspire others (both developers and scientist) to further improve the current features, and come up with valuable new ones. If you are interested in some of the features we implemented, see our [official documentation][docs-url].

## Changelog 0.6.0

- Moved from function to an object oriented framework for all user-features and computations of game/match (special thanks to [DaanGro](https://github.com/DaanGro))
- Renamed the all classes and functions with `match` to `game` (to move away from the internal python `match` statement)
- Removed the function to save game/match objects to pickle, but created a more save way using parquet and json files
- Added functionality to export tracking data to long format.

#### Breaking changes
We sincerely appologize for all the changes you have to make, but we feel this will make the package more robust and easier to use for future projects. Just to be clear, **all the functionality that was in 0.5.4, is still in 0.6.0**. However we made to changes that impacts users.
1) We renamed all functions with `match` in it to `game.` (e.g. `get_match` was changed to `get_game`). This was chosen since `match` is an internal python command, and we do not want to imply to overwrite that (by using something like `match = get_match()`). A deprecation warning is raised when you try to call it from the current version onwards, we strongly encourage to take this warning serious as we do plan to remove it shorlty.
2) If you used any of the features in databallpy (`get_velocity`, `get_approximate_voronoi`, `get_covered_distance`, etc.), you need to refactor your code. All functionality is still available in the package, but likely as a method on the `game.tracking_data` class, for instance: `game.tracking_data.add_velocity(...)`. Please view our documentation of version 0.6.0 to see how you can refactor your code (spoiler: you will need to use less arguments and less lines of code!)
3) Saved games/matches are only usable in the version in which you saved them up and untill version 0.5.4. For example, if you saved your match in version 0.5.2, you can only load it while using version 0.5.2. From version 0.6.0, you will be able to load your game using the `get_saved_game` as long as your version is greater or equal to 0.6.0.

## Installation

```bash
$ pip install databallpy
```

## Usage

The package is centered around the `Game` object. A `Game` has tracking data, event data metadata about the game.
For a more elaborate example, see the [Getting Started][getting-started-url].

```python
from databallpy import get_game, get_open_game

game = get_game(
  tracking_data_loc="../data/tracking_data.dat",
  tracking_metadata_loc="../data/tracking_metadata.xml",
  tracking_data_provider="tracab"
  event_data_loc="../data/event_data_f24.xml",
  event_metadata_loc="../data/event_metadata_f7.xml",
  event_data_provider="opta",
)

# or get the open game provided by the DFL (Sportec Solutions)
game = get_open_game()
```

> [!note]
> The current supported tracking data providers are:
> - Tracab (including Sportec Solutions from the DFL)
> - Metrica
> - Inmotio
> 
> The accepted variables for the `tracking_data_provider` are `["tracab", "metrica", "inmotio", "dfl", "sportec"]`
> 
> The current supported event data provider are:
> - Opta
> - Metrica
> - Instat
> - SciSports
> - Sportec Solutions (from the DFL)
> - Statsbomb
> 
> The accepted variables for the `event_data_provider` are `["opta", "metrica", "instat", "scisports", "dfl", "sportec", "statsbomb"]`
>
> If you wish to use a different provider that is not listed here, please open an issue [here](https://github.com/Alek050/databallpy/issues)

### Open Data
A great project just released tracking and event data for open use. In total, 2 games of the 1st bundesliga and 5 of the 2nd bundesliga were open sourced [here](https://www.nature.com/articles/s41597-025-04505-y). The dataset was open sources via the CC-BY 4.0 licence, so please consider this when you make use of the data. The data is automatically downloaded and saved locally when you use the `get_open_game` function from DataBallPy. The match information can be imported from `databallpy.utils.constants.OPEN_GAME_IDS_DFL`:

```python
OPEN_GAME_IDS_DFL = {
    "J03WMX": "1. FC Köln vs. FC Bayern München",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Düsseldorf vs. 1. FC Nürnberg",
    "J03WOH": "Fortuna Düsseldorf vs. SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Düsseldorf vs. FC St. Pauli",
    "J03WOY": "Fortuna Düsseldorf vs. F.C. Hansa Rostock",
    "J03WR9": "Fortuna Düsseldorf vs. 1. FC Kaiserslautern",
}
```

See [the documentation][docs-url] of the `Game` object and the [example usage][example-url] for more options. Note that this package is developed to combine event and tracking data, for now both datastreams are necessary to create a `Game` object.

## Features

### Preprocessing

- [Data Filtering][filter-data-url]: Filter the data based on the quality of the tracking data.
- [Adding velocity and Acceleration][velocity-acc-url]: Add velocity and acceleration to the tracking data.


### Synchronization of tracking and event data

See our elaborate [synchronisation page][synchronisation-url] in the documentation for more information!

Tracking and event data is often poorly synchronized. For instance, when taking the event data of Opta and tracking data of Tracab, you can sync the fist frame with the kick-off pass. Now you can sync the other events with the tracking data based on the time difference between the event and the kick off pass. If you do this, how get something like this:

<video width="640" height="480" controls>
  <source src="https://raw.githubusercontent.com/Alek050/databallpy/main/docs/example_data/not_synced.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

https://user-images.githubusercontent.com/49450063/224564808-fa71735f-5510-464d-8499-9044227a02e8.mp4


As you can see, the timing (and placing) of the events do not correspond good with the tracking data locations, especially when events follow up quickly or around shots. Using the methodology of [this][soccer-sync-url] article, this package is able to synchronize tracking and event data using the Needleman-Wunsch algorithm. 

After running the following command, the events are better synchronized to the tracking data:

```batch
$ game.synchronise_tracking_and_event_data()
```

<video width="640" height="480" controls>
  <source src="https://raw.githubusercontent.com/Alek050/databallpy/main/docs/example_data/synced.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


https://user-images.githubusercontent.com/49450063/224564913-4091faf7-f6ef-4429-b132-7f93ce5e1d91.mp4

For a more elaborate example of how we synchronize the tracking and event data, see the [Synchronisation Page][getting-started-url] in our [documentation][docs-url].

### Visualizations

DataBallPy offers a variety of visualizations to help you understand the data better. For example, you can visualize the tracking data with synchronised event as shown above. Also, you can visualize events and tracking data separately. For more information, see the [Visualizations Page][visualizations-url] in our [documentation][docs-url].

### Soccer Specific Metrics

- [Covered Distance][covered-distance-url]: Calculate the covered distance in different velocity and acceleration zones.
- [Pressure][pressure-url]: Calculate the pressure any player in the game (Herold & Kempe, 2022).
- [Team Possession][team-possession-url]: Calculate the team possession based on the synchronised event data.
- [Individual Player Possession][individual_possession-url]: Calculate the individual player possession based on the tracking data (Vidal-Codina et al., 2022).
- [Simple Expected Goals (xG) model][Expected-Goals-url]: Calculate the simple expected goals model.
- [Expected Threat model][Expected-Threat-url]: Calculate the expected threat model from Karun Singh to on ball events.
- [Voronoi Model][Voronoi-url]: Calculate the Voronoi space occupation based on the tracking data.
- [Gaussian Model][Guassian-space-occupation-url]: Calculate the Gaussian space occupation based on the tracking data (Fernandez & Born, 2018).

## Documentation

The official documentation can be found [here][docs-url].

## Providers

For now we limited providers. We are planning on adding more providers later on.

Event data providers:
- Opta
- Metrica
- Instat
- SciSports
- Sportec Solutions (for the DFL)

Tracking data providers:
- Tracab (including Sportec Solutions format from the DFL)
- Metrica
- Inmotio

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Maintainers & owners

- [Alek050](https://github.com/Alek050/)

## Contributors

- [DaanGro](https://github.com/DaanGro/)
- [rdghe](https://github.com/rdghe/)
- [swopper050](https://github.com/Swopper050)
- [maritsloots](https://github.com/maritsloots)
- [jan-swiatek](https://github.com/jan-swiatek)

## License

`databallpy` was created by Alexander Oonk & Daan Grob. It is licensed under the terms of the MIT license.

## Similar projects

Although we think this package helps when starting to analyse soccer data, other packages may be better suited for your specific needs. Make sure to check out the following packages as well:
- [kloppy][kloppy-url]
- [floodlight][floodlight-url]
- [codeball](https://github.com/metrica-sports/codeball)
- [socceraction](https://github.com/ML-KULeuven/socceraction)

And for a more specific toturials on how to get started with soccer data"
- [Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD)



