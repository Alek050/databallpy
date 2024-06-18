[floodlight-url]:https://github.com/floodlight-sports/floodlight
[kloppy-url]:https://github.com/PySport/kloppy
[soccer-sync-url]: https://kwiatkowski.io/sync.soccer

# Welcome to DataBallPy!

Welcome to the official documentation of DataBallPy!

This package is developed to create a standardized way to analyse soccer matches using both event- and tracking data. Other packages, like [kloppy][kloppy-url] and [floodlight][floodlight-url], already standardize the import of data sources. The current package goes a step further in combining different data streams from the same match. In this case, the `Match` object combines information from the event and tracking data. The main current feature is the smart synchronization of the tracking and event data. We utilize the Needleman-Wunch algorithm, inspired by [this article][soccer-sync-url], to align the tracking and even data, while ensuring the order of the events, something that is not done when only using (different) cost functions.

Although reading in and synchronising data is already very helpfull to get started with your analysis, it's only the first step. Even after this first step, getting your first 'simple' metrics out of the data might be more difficult than anticipated. Therefore, the primary end goal for this package is to create a space where (scientific) soccer metrics are implemented and can be used in a few lines. We even plan to go further and show clear notebooks (to combine text and code) with visualizations for all the features we implement. This way, you will not only get easy access to the features/metrics, but also understand exactly how it is calculated. We hope this will inspire others (both developers and scientist) to further improve the current features, and come up with valuable new ones. 

<div align="center">
  <img src="https://github.com/Alek050/databallpy/assets/49450063/56100e87-c680-4dc1-82e5-4aa8fdbc8a34" alt="Logo">
</div>
