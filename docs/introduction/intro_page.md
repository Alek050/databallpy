[floodlight-url]:https://github.com/floodlight-sports/floodlight
[kloppy-url]:https://github.com/PySport/kloppy
[soccer-sync-url]: https://kwiatkowski.io/sync.soccer
[example-url]: https://databallpy.readthedocs.io/en/latest/example.html

<div align="center">
  <img src="https://github.com/Alek050/databallpy/assets/49450063/56100e87-c680-4dc1-82e5-4aa8fdbc8a34" alt="Logo">
</div>

# Overview
This package is developed to create a standardized way to analyse soccer matches using both event- and tracking data. Other packages, like [kloppy][kloppy-url] and [floodlight][floodlight-url], already standardize the import of data sources. The current package goes a step further in combining different data streams from the same match. In this case, the `Match` object combines information from the event and tracking data. The main current feature is the smart synchronization of the tracking and event data. We utilize the Needleman-Wunch algorithm, inspired by [this article][soccer-sync-url], to align the tracking and even data, while ensuring the order of the events, something that is not done when only using (different) cost functions.

Although reading in and synchronising data is already very helpfull to get started with your analysis, it's only the first step. Even after this first step, getting your first 'simple' metrics out of the data might be more difficult than anticipated. Therefore, the primary end goal for this package is to create a space where (scientific) soccer metrics are implemented and can be used in a few lines. We even plan to go further and show clear notebooks (to combine text and code) with visualizations for all the features we implement. This way, you will not only get easy access to the features/metrics, but also understand exactly how it is calculated. We hope this will inspire others (both developers and scientist) to further improve the current features, and come up with valuable new ones. 

## Installation
DataBallPy is available on PyPi and can be installed using pip:

```bash
$ pip install databallpy
```

## Features
- Fast synchronisation of tracking and event data using a Needlemann-Wunch algorithm
- Seamless calculations of important features like expected goals (xG) and expected threat (xT)
- The use of notebooks to visualize and explain how features are calculated
- Creating and saving videos from the tracking data.

## Why DataBallPy?
DatabBallPy emerged from a drive to understand the full process of changing raw soccer data into valuable metrics. While other packages focus on importing data, DataBallPy goes a step further by combining different data streams from the same match. This package is developed to create a standardized way to analyse soccer matches using both event- and tracking data. We also acknowledge that this is just the starting point for most analysts, and that not every analyst might want to know all exact details of the calculations. Therefore, we aim to provide a clear and easy-to-use package that can be used by both developers and scientists.

We try to make this package accessible to all interested. All features and metrics are easy accessible with minimal code, which makes it easy to get started. For those who want to know more about the calculations, we provide notebooks with visualizations and explanations of the calculations. We hope that this will make it easier for others to understand exaclty what is happening, catch bugs earlier in the process, and inspire others to come up with new features and metrics that, in turn, can be added to the package. 

## Who is DataBallPy for?
DataBallPy is for anyone interested in soccer analytics. Whether you are a developer, a scientist, or just a soccer fan, this package is for you. We aim to make the package accessible to all interested, and provide clear explanations of the calculations for those who want to know more.

## How DataBallPy Works
The package is centered around the `Match` object. A `Match` has tracking data, event data, and metadata about the match.
Furthermore, some basic preprocessing (normalizing playing direction, scaling tracking and event data to meters, etc.) is done when loading the data. This ensures that the data is ready for further analysis, no matter the data source you used. We advice you to go through our [Getting-Started][getting-started-page] to get a solid understanding of the package and how to use it. But if you cannot wait to get your hands dirty, you can start by loading a match using the following code:

```python
from databallpy import get_match, get_open_match

match = get_match(
  tracking_data_loc="../data/tracking_data.dat",
  tracking_metadata_loc="../data/tracking_metadata.xml",
  tracking_data_provider="tracab"
  event_data_loc="../data/event_data_f24.xml",
  event_metadata_loc="../data/event_metadata_f7.xml",
  event_data_provider="opta",
)

# or to load an open metrica dataset of tracking and event data
match = get_open_match()
```

## About the Developers
This package is developed by a group of soccer enthusiasts who are passionate about soccer analytics. We aim to make soccer analytics accessible to all interested, and provide a clear and easy-to-use package that can be used by both developers and scientists. We hope that this package will inspire others to come up with new features and metrics that, in turn, can be added to the package.

### Maintainers & owners

- [Alek050](https://github.com/Alek050/)

### Contributors

- [DaanGro](https://github.com/DaanGro/)
- [rdghe](https://github.com/rdghe/)
- [swopper050](https://github.com/Swopper050)
