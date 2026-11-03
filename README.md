
<div align="center">
  <img src="https://github.com/Alek050/databallpy/assets/49450063/56100e87-c680-4dc1-82e5-4aa8fdbc8a34" alt="Logo">
</div>

[version-image]: https://img.shields.io/pypi/v/databallpy?color=006666
[version-url]: https://pypi.org/project/databallpy/
[python-image]: https://img.shields.io/pypi/pyversions/databallpy?color=006666
[python-url]: https://pypi.org/project/databallpy/
[docs-image]: https://readthedocs.org/projects/databallpy/badge/?version=main
[docs-url]: https://databallpy.readthedocs.io/en/main/?badge=main
[cicd-image]: https://github.com/Alek050/databallpy/actions/workflows/linters_and_tests.yml/badge.svg
[cicd-url]: https://github.com/Alek050/databallpy/actions/workflows/linters_and_tests.yml
[codecov-image]: https://codecov.io/gh/Alek050/databallpy/branch/develop/graph/badge.svg?token=MBI3380I0H
[codecov-url]: https://codecov.io/gh/Alek050/databallpy
[joss-image]: https://joss.theoj.org/papers/10.21105/joss.10223/status.svg
[joss-url]: https://doi.org/10.21105/joss.10223

[getting-started-url]: https://databallpy.readthedocs.io/en/main/getting_started/installation_page.html
[loading-game-url]: https://databallpy.readthedocs.io/en/main/getting_started/loading_in_a_game_page.html
[kloppy-integration-url]: https://databallpy.readthedocs.io/en/main/getting_started/loading_in_a_game_page.html#databallpy-kloppy-integration
[filter-data-url]: https://databallpy.readthedocs.io/en/main/getting_started/preprocessing_options.html#filtering-data
[velocity-acc-url]: https://databallpy.readthedocs.io/en/main/getting_started/preprocessing_options.html#adding-velocity-and-acceleration
[synchronisation-url]: https://databallpy.readthedocs.io/en/main/getting_started/synchronisation_page.html
[visualizations-url]: https://databallpy.readthedocs.io/en/main/getting_started/visualizations_page.html
[covered-distance-url]: https://databallpy.readthedocs.io/en/main/features/covered_distance_page.html
[pressure-url]: https://databallpy.readthedocs.io/en/main/features/pressure_page.html
[team-possession-url]: https://databallpy.readthedocs.io/en/main/features/team_and_player_possession.html#team-possession
[individual-possession-url]: https://databallpy.readthedocs.io/en/main/features/team_and_player_possession.html#individual-player-possession-vidal-codina-et-al-2022
[expected-goals-url]: https://databallpy.readthedocs.io/en/main/features/simple_xG_models.html
[expected-threat-url]: https://databallpy.readthedocs.io/en/main/features/xT_models.html
[voronoi-url]: https://databallpy.readthedocs.io/en/main/features/space_occupation.html#voronoi-model
[gaussian-space-occupation-url]: https://databallpy.readthedocs.io/en/main/features/space_occupation.html#gaussian-model-fernandez-born-2018
[dangerous-accessible-space-url]: https://databallpy.readthedocs.io/en/main/features/dangerous_accessible_space.html
[video-xml-url]: https://databallpy.readthedocs.io/en/main/features/video_analysis_xml_export.html
[optimization-url]: https://databallpy.readthedocs.io/en/main/features/optimization.html

[kloppy-url]: https://github.com/PySport/kloppy
[floodlight-url]: https://github.com/floodlight-sports/floodlight
[socceraction-url]: https://github.com/ML-KULeuven/socceraction
[spadl-url]: https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html
[accessible-space-url]: https://github.com/jonas-bischofberger/accessible-space
[soccer-sync-url]: https://kwiatkowski.io/sync.soccer
[open-data-url]: https://www.nature.com/articles/s41597-025-04505-y

[![Latest Version][version-image]][version-url]
[![Python Version][python-image]][python-url]
[![Documentation Status][docs-image]][docs-url]
[![CI/CD Status][cicd-image]][cicd-url]
[![Codecov][codecov-image]][codecov-url]
[![PyPI Downloads](https://static.pepy.tech/badge/databallpy)](https://pepy.tech/projects/databallpy)
[![DOI][joss-image]][joss-url]

# DataBallPy

Load, synchronise and analyse your soccer event and tracking data.

DataBallPy combines the event and tracking data of a game in a single `Game` object. It parses the data of many providers into one standardised format, synchronises the events with the tracking data, describes all events in the [SPADL][spadl-url] vocabulary and offers (scientific) soccer metrics that can be computed in a few lines of code. The package is described in [Oonk, Grob & Kempe (2026), Journal of Open Source Software][joss-url].

Every feature comes with a notebook in the [documentation][docs-url] that explains how it is computed, so you know exactly what you get and can build on it.

## Installation

```bash
pip install databallpy
```

Optional extras:

```bash
pip install 'databallpy[kloppy]'             # load any kloppy-supported provider
pip install 'databallpy[accessible-space]'   # Dangerous Accessible Space
```

## Quick start

```python
from databallpy import get_game, get_open_game

game = get_game(
    tracking_data_loc="../data/tracking_data.dat",
    tracking_metadata_loc="../data/tracking_metadata.xml",
    tracking_data_provider="tracab",
    event_data_loc="../data/event_data_f24.xml",
    event_metadata_loc="../data/event_metadata_f7.xml",
    event_data_provider="opta",
)

# or load one of the open DFL games
game = get_open_game()

# align the events with the tracking data
game.synchronise_tracking_and_event_data()

# all actions as a SPADL dataframe
actions = game.to_spadl()
```

See [Getting Started][getting-started-url] and [Loading in a game][loading-game-url] for more options, such as loading only part of the tracking data or loading any provider through [kloppy][kloppy-integration-url].

## Events in SPADL

All event data is translated to the SPADL vocabulary (Decroos et al., 2019) in the `action_type`, `action_result` and `action_bodypart` columns, while `original_event` keeps the provider's event. The `databallpy_event` column is deprecated and will be removed in v0.11.0.

- Carries (SPADL `dribble`) between consecutive actions of the same team are added to the event data when a game is loaded, so they are synchronised and visualised as well.
- `game.to_spadl()` returns a [socceraction][socceraction-url]-compatible actions dataframe, in DataBallPy or SPADL coordinates and in the playing direction of your choice. Missing end locations of actions that move the ball (e.g. failed passes) are filled with the start of the next action.

## Synchronisation of tracking and event data

The timestamps and locations of event data rarely match the tracking data. DataBallPy aligns them with the Needleman-Wunsch algorithm, inspired by [sync.soccer][soccer-sync-url], which keeps the order of the events intact. Each SPADL action type is matched to the moment it describes:

- **release**: passes, crosses, set pieces, clearances and bad touches, when the ball leaves the player;
- **shot**: shots, when the ball leaves the player towards goal;
- **arrival**: carries, take-ons, interceptions and keeper actions, when the ball arrives at the player;
- **general**: tackles and own goals.

After `game.synchronise_tracking_and_event_data()` the event data has a `tracking_frame`, a `sync_certainty` and the ball location in that frame (`tracking_start_x`, `tracking_start_y`), while `start_x`/`start_y` keep the provider's location. Before and after synchronisation:

https://user-images.githubusercontent.com/49450063/224564808-fa71735f-5510-464d-8499-9044227a02e8.mp4

https://user-images.githubusercontent.com/49450063/224564913-4091faf7-f6ef-4429-b132-7f93ce5e1d91.mp4

See the [synchronisation page][synchronisation-url] for how it works and how to use your own cost functions.

## Features

**Preprocessing and visualisation**
- [Data filtering][filter-data-url] and [velocity and acceleration][velocity-acc-url]
- [Visualisations][visualizations-url]: plot events, tracking frames and videos with synchronised events
- [Video analysis XML export][video-xml-url]: export events to video analysis software

**Metrics**
- [Covered distance][covered-distance-url] in velocity and acceleration zones
- [Pressure][pressure-url] (Herold & Kempe, 2022)
- [Team possession][team-possession-url] and [individual player possession][individual-possession-url] (Vidal-Codina et al., 2022)
- [Simple expected goals (xG)][expected-goals-url]
- [Expected threat (xT)][expected-threat-url] (Singh, 2018)
- Space occupation: [Voronoi][voronoi-url] and [Gaussian][gaussian-space-occupation-url] (Fernández & Born, 2018)
- [Dangerous Accessible Space][dangerous-accessible-space-url] (Bischofberger & Baca, 2025)
- [Defensive structure optimisation][optimization-url]

## Supported providers

| Provider | Event data | Tracking data |
| --- | :---: | :---: |
| Opta / StatsPerform (F7/F24 and MA2/MA13) | ✓ | |
| Sportec Solutions (DFL) | ✓ | ✓ |
| Tracab | | ✓ |
| Metrica | ✓ | ✓ |
| Inmotio | | ✓ |
| SciSports | ✓ | |
| StatsBomb | ✓ | |
| FIFA | ✓ | |
| InStat (deprecated, no SPADL) | ✓ | |
| Any other [kloppy][kloppy-url] provider | ✓ | ✓ |

Missing a provider? Please [open an issue](https://github.com/Alek050/databallpy/issues).

## Open data

`get_open_game()` downloads and caches one of the seven games (two 1. Bundesliga, five 2. Bundesliga) released by the DFL (Bassek et al., 2025, [Scientific Data][open-data-url]) under the CC-BY 4.0 licence. The available games are listed in `databallpy.utils.constants.OPEN_GAME_IDS_DFL`.

## Citing

If you use DataBallPy in your work, please cite the [JOSS paper][joss-url]:

```bibtex
@article{Oonk2026,
   author = {Gerard Alexander Oonk and Daan Grob and Matthias Kempe},
   doi = {10.21105/JOSS.10223},
   issn = {2475-9066},
   issue = {120},
   journal = {Journal of Open Source Software},
   month = {4},
   pages = {10223},
   title = {DataBallPy: Load, Synchronize, and Analyse your Soccer Data},
   volume = {11},
   url = {https://joss.theoj.org/papers/10.21105/joss.10223},
   year = {2026}
}
```

Please also cite the papers behind the methods you use (see below).

## Acknowledgements

DataBallPy builds on the work of others:

- **Packages**: [kloppy][kloppy-url] for loading many providers, [floodlight][floodlight-url] for its standardisation of soccer data, [socceraction][socceraction-url] for the SPADL format and [accessible-space][accessible-space-url] for Dangerous Accessible Space.
- **Methods**:
  - Synchronisation: [sync.soccer][soccer-sync-url] (Kwiatkowski)
  - SPADL: Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in soccer. *KDD '19*.
  - Pressure: Herold, M., & Kempe, M. (2022).
  - Individual player possession: Vidal-Codina, F., et al. (2022).
  - Expected threat: Singh, K. (2018).
  - Gaussian space occupation: Fernández, J., & Born, L. (2018).
  - Dangerous Accessible Space: Bischofberger, J., & Baca, A. (2025).
- **Data**: the DFL and Sportec Solutions for the open dataset (Bassek et al., 2025), and the providers whose formats are supported.

## Contributing and support

Contributions are welcome, see the [contributing guidelines](https://databallpy.readthedocs.io/en/main/contributing/quick_overview.html). This project is released with a Code of Conduct; by contributing you agree to its terms. Questions and ideas are welcome in the [discussions](https://github.com/Alek050/databallpy/discussions). See the [CHANGELOG](CHANGELOG.md) for what changed in each version.

**Maintainer**: [Alek050](https://github.com/Alek050/)

**Contributors**: [DaanGro](https://github.com/DaanGro/), [rdghe](https://github.com/rdghe/), [swopper050](https://github.com/Swopper050), [maritsloots](https://github.com/maritsloots), [jan-swiatek](https://github.com/jan-swiatek), [UnravelSports](https://github.com/UnravelSports), [migvidal5](https://github.com/migvidal5), [jonas-bischofberger](https://github.com/jonas-bischofberger), [ouyang1030](https://github.com/ouyang1030), [amarshah1999](https://github.com/amarshah1999)

## Similar projects

Other packages may suit your needs better, make sure to check out [kloppy][kloppy-url], [floodlight][floodlight-url], [socceraction][socceraction-url] and [codeball](https://github.com/metrica-sports/codeball). To get started with soccer data in general, see [Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD).

## License

DataBallPy was created by Alexander Oonk & Daan Grob and is licensed under the MIT license.
