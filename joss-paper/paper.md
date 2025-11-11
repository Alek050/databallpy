---
title: 'DataBallPy: Load, Synchronise, and Analyse your Soccer Data
tags:
  - Python
  - football
  - analysis
  - soccer
  - visualisation
authors:
  - given-name: Gerard Alexander 
    surname: Oonk
    corresponding: true 
    orcid: 0000-0003-4056-7274
    affiliation: 1 
  - given-name: Daan 
    surname: Grob
    affiliation: 2
  - given-name: Matthias 
    surname: Kempe
    orcid: 0000-0002-4709-6172
    affiliation: "1, 3"
affiliations:
 - name: Department of Sports Sciences, University of Groningen, the Netherlands
   ror: 03cv38k47 
   index: 1
 - name: Independent Researcher, the Netherlands
   index: 2
 - name: Centre for Sport Science and University Sports, University of Vienna, Austria
   ror: 03prydq77 
   index: 3
date: 05-11-2025
bibliography: paper.bib

# Summary

`DataBallPy` is a Python package designed to quickstart the analysis of soccer data by integrating and synchronizing event and tracking data. It provides a standardized interface for loading, preprocessing, and visualizing match data, enabling researchers and analysts to extract insights with minimal setup. By combining multiple data streams into a unified `Game` object, `DataBallPy` simplifies complex workflows and supports reproducible, data-driven soccer research. Last, DataBallpy is serious on documentation. Every preprocessing option and feature is elaborately explained in the  documentation. `DataBallPy` does not only help in computing scientific features, but the down-to-earth docoumentations helps you better understand how features are computed.

# Statement of need

Modern soccer analytics increasingly rely on both event data and tracking data. Event data captures specific information about events (e.g. passes and shots) like their location, success, start location, and the athlete involved in the action. This information on itself is primarily used for aggregated statistics that can be uses for tactical match and player analysis [@Goes2020] and is widely used in scouting because of the low cost and wide spread availability of the data [@vanArem2025]. Tracking data, on the other hand, captures spatiotemporal information of all athletes and the ball at frequencies ranging between 10 and 25 Hz [@Linke2020]. This data is primarily used to quantify physical performance, but also for detection of dynamic formation [@sotudeh2025]. Current package allow for parsing [Kloppy](https://kloppy.pysport.org) and analysis[@Raabe2022] of either datastream independently. However, there has been a growing interest in combining event and tracking data to enrich event information with spatiotemporal context. This added context provides insights and nuances, primarily on a tactical level, that neither event and tracking data can not provide independently. 

`DataBallPy` addresses this gap by combining all game related data in a standardized `Game` object. The `Game` object includes event, tracking, and metadata. The primary feature of `DataBallPy` is the robust and efficient synchronistation between event and tracking data. Although event and tracking data often both provide timestamps, their alignment has shown to be extremely poor with reported errors of 1.82 (+-4.06) seconds. Especially the random error is concerning since it does not allow for easy correction and within 4 seconds the game might have evolved to an entirely different situation. Although specific approaches have been introduced to solve this problem, they can take between 3 and 10 minutes per game of runtime, may skip certain events, and shuffle the order of events [@VanRoy2024; @Kim2025]. `DataBallPy` allows for a state of the art synchronisation algorithm that ensures the synchronisation of all events in the right order within a few seconds [@Oonk2025] in just one line of code. 

The synchronisation of event and tracking data allows for deeper analysis. For example `DataBallPy` provides functionality to (re)compute a frame-wise assessment of which team has ball possession, as some tracking data provides do not provide it. Furthermore, a propper analysis of goal scoring probability (xG) and expected threat (xT) is included based on the combined information of both tracking and event data which have been shown to perform better when there is a propper alignment between the event and tracking data [@Oonk2025]. 

Next to the practical value of `DataBallPy`, as it provides low code access to scientific features, `DataBallPy` also provides as an educational tool. Often, open-source python packages provide information on how to get working code, but not on how the code works. `DataBallPy` explicitly goes a step further by elaborately explaining step by step how scientific papers are transformed into code, often refering to specific mathematical formulas as presented in the paper. This explenation is crucial since it (1) allows researchers and practitioners to better understand the strengths and weaknesses of features, and (2) teaches users on how to transform scientific papers into modular, Pythonic code. Both these characteristics provide users of `DataBallPy` a better understanding of their own analysis.


# Features

The features and functionalities in `DataBallPy` can be catagorised in five categories: parsing data, preprocessing, synchronisation, Performance Indicators, and visualisation. 

## Parsing Data

The core goal of parsing data in `DataBallPy` is obtaining a `Game` object. `DataBallPy` allows for parsing data from Tracab, Metrica, Inmotio, Opta, Instat, SciSports, Sportec, and Statsbomb internally using the `get_game` function. The `Game` object contains the event and tracking data internally as Pandas dataframes, making them intuitive to work with [@reback2020pandas]. Alternatively, one can use [Kloppy](https://kloppy.pysport.org/) to parse data from more providers and use the `get_game_from_kloppy` function to transform it to a `Game` object. Last, `DataBallPy` has included a function to load openly available data directly in a `Game` object using `get_open_game` which allows users that do not have access to data to still work with soccer data in `DataBallPy` [@Bassek2025]. Since parsing and the analytical pipeline of soccer data takes time and resources, `DataBallPy` can also save your processed `Game` object. Normally, raw tracking and event data together can take up to 400 MB per game, 'DataBallPy' downscales this to less then 20 MB per game and can be reloaded by using the `get_saved_game` function.

## Preprocessing

Tracking data is often captured via video footage. Depending on the quality and number of camera's, some noise is present in both the athelte and ball positions. `DataBallPy` allows for filtering of the tracking data, differentation of positions to compute velocity and acceleration. Furthermore, the tracking data allows for computation of individual athlete possession [@Vidal-Codina2022] and together with the event data team level possession can be estimated. 

## Synchronisation

`DataBallPy` uses a soccer specific implementation of the Needleman-Wunch algorithm to synchronise the event and tracking data[@Oonk2025]. The game can be synchronised via using the following code
```python
>>> from databallpy import get_open_game
>>> game = get_open_game()
>>> game.synchronise_tracking_and_event_data()
```

## Performance Indicators

`DataBallPy` has an elabore list of scientific features included in the package. All features can be computed in a few lines of code after obtaining a `Game` object. Next the the functionality, the documentation covers an elaborate explenation of how the code works that computes the features. Using `DataBallPy` the following features can be computed:

- Covered Distance (in specific velocity and acceleration zones) [@Jerome2024]
- Pressure [@Andrienko2017; @Herold2022]
- Individual player possession [@Vidal-Codina2022]
- Expected Goals [@Anzer2021]
- Expected Threat [@Singh2019]
- Voronoi Space Occupation [@Rein2017]
- Pitch Control [@Fernandez2018]

## Visualisation

![Example plot of soccer tracking data with pitch control heatmap as introduced in @Fernandez2018](fig1.png)

`DataBallPy` includes elaborate functionality to visualise the data in the `Game` object. Events locations can be visualised on a pitch using the `plot_events()` function, which allows for coloring of events by outcome, team or event type during specific periods in the game. Similarly, the locations and velocities of all players can be plotted using `plot_tracking_data()` function. If the event and tracking data is synchronised, one can also show information of the event in the same plot. Other features like pitch control heatmaps, player possession, and any custom feature can also be visualised simultaneously with the event and tracking data (Figure 1). Last, the tracking data (with heatmaps and custom features) can be transformed to a video.mp4 to show the true spatiotemporal progression over time.

```python
import matplotlib.pyplot as plt

from databallpy import get_open_game
from databallpy.visualize import plot_tracking_data

game = get_open_game()
game.tracking_data.add_velocity(game.get_column_ids() + ["ball"])

pitch_control = game.tracking_data.get_pitch_control(
    game.pitch_dimensions,
    start_idx=100,
    end_idx = 101
)


fig, ax = plot_tracking_data(
    game,
    idx=100,
    add_velocities=True,
    heatmap_overlay=pitch_control[0],
    overlay_cmap="plasma",
    team_colors=["#00FFFF", "#00FF00"]
)
plt.show()
```



# References