---
title: 'DataBallPy: Load, synchronize, and Analyse your Soccer Data'
tags:
  - Python
  - Association Football
  - Tactical Analysis
  - synchronization
  - Tracking Data
  - Event Data
authors:
  - given-names: Gerard Alexander 
    surname: Oonk
    corresponding: true 
    orcid: 0000-0003-4056-7274
    affiliation: 1 
  - given-names: Daan 
    surname: Grob
    affiliation: 2
  - given-names: Matthias 
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
---


# Summary

Over the last decade, there has been a growing interest in soccer analytics from different backgrounds and for different use cases. Example use cases include: (1) practical decision-making and player benchmarking based on aggregated metrics such as pass success percentage and expected goals (xG) [@Goes2020a]; (2) training periodization and injury prediction using internal and external load metrics [@Hader2019]; and (3) behavioral science research focusing on group and subgroup dynamics [@Goes2020b]. The interest in soccer analysis has also increased as data has become more openly available [@Bassek2025]. However, a key challenge is that every data provider uses their own data format, which makes it hard to compare and switch between different providers and create large datasets that encompass different leagues and competitions. Currently, open-source packages like [Kloppy](https://kloppy.pysport.org) address this challenge by providing a uniform data format. Similarly, the scientific community has proposed a common data format for soccer game data [@Anzer2025]. While Kloppy focuses primarily on parsing soccer data, Floodlight [@Raabe2022] provides a framework for physical analysis of team sports, and [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) is widely utilized for visualizing soccer data. 

Recently, there has been a growing interest in combining event and tracking data for contextualized tactical analysis of soccer games. This enables analysts to not only identify when a pass occurred (event data) but also assess the defensive structure during the pass [@Forcher2022; @Herold2022] and evaluate other passing options available at that moment (tracking data) [@Spearman2017]. Contextual analysis goes beyond aggregated metrics, enabling quantative analysis of specific moments or phases in the game [@Oonk2025a; @Jerome2024]. Merging tracking and event data is a key challenge for contextualized analysis of soccer games. [`DataBallPy`](https://databallpy.readthedocs.io/en/latest/) is an open-source Python package for contextual analysis of soccer games, achieved by: (1) using a standardized data format for both event and tracking data; (2) bundling all game data into a unified framework, rather than treating them as separate objects; (3) incorporating a high-quality, learning-free synchronization algorithm compatible with any combination of tracking and event data providers; and (4) integrating multiple practical and scientific features for efficient computation with minimal user input. 


# Statement of need

Modern soccer analytics increasingly rely on both event data and tracking data for a more in-depth analysis. Event data captures specific information about events (e.g., passes and shots) including their location, success, start location, and the athlete involved in the action. This information is primarily aggregated for tactical game and player analysis [@Goes2020a] but is also widely used in scouting because of the low cost and widespread availability of the data [@vanArem2025]. In contrast, tracking data provides spatiotemporal information for all athletes and the ball at frequencies ranging between 10 and 25 Hz [@Linke2020]. This data is primarily used to quantify physical performance but also for detecting dynamic formations [@sotudeh2025], identifying events [@Vidal-Codina2022], classifying game phases [@Bauer2023], analyzing space occupation [@Spearman2017; @Rein2017], and quantifying danger [@Link2016]. However, there has been a increasing interest in combining event and tracking data to enrich event information with spatiotemporal context.

This added context provides insights and nuances, primarily on a tactical level, that neither event nor tracking data can provide independently. For example, shot events are enriched with defensive and goalkeeper positioning data to improve expected goals models [@Anzer2021]; passes are evaluated using risk-reward assessments of all possible passing options [@Goes2021]; determinants of successful 1v1 actions are modeled from spatiotemporal features [@Oonk2025a]; and the spatiotemporal context of events is used to predict the dangerousity of a game state [@Fernandez2021].

A contextual analysis requires a proper synchronization of event and tracking data. Although both event and tracking data provide timestamps, their alignment has been shown to be extremely poor, with reported errors of 1.82 (±4.06) seconds [@Anzer2021]. The random error is particularly concerning because it precludes easy correction; within 4 seconds, the game may have evolved to an entirely different situation, which impacts the stability of the found effects. @Oonk2025b demonstrated that the expected goal model decreased in Brier loss from 0.096 to 0.082 (lower is better) when using the synchronized data compared to naive timestamp alignment. Similarly, the feature importance of features that relied on combined tracking and event data information were close to 0 in the naive timestamp synchronization model, unlike the properly synchronized situation [@Oonk2025b]. Thus, there is a need for user-friendly software with a state-of-the-art synchronization algorithm and a convenient data structure for subsequent analysis.

# State of the field

Currently available packages enable parsing and analysis of tracking and event data separately. ([Kloppy](https://kloppy.pysport.org)) is a well-known data parsing package in the soccer analytics field. Its primary focus is to simplify and standardize the parsing of soccer tracking and event data from various providers. A similar project aims to establish a standardized format for soccer, which could make different parsers redundant in the future [@Anzer2025]. Other packages support analysis [@Raabe2022] and visualization ([mplsoccer](https://github.com/andrewRowlinson/mplsoccer)) of soccer tracking and event data. 

Some different open-source projects focus less on parsing and analysis, but do incorporate synchronization between tracking and event data. @VanRoy2024 synchronizes events with the tracking data using event-specific cost functions. However, the approach takes up to three minutes per match and can leave events unassigned to the tracking data. @Kim2025 recognized that event locations often contain large errors and thus developed an algorithm to merge event and tracking data without relying on event positions. However, both approaches take considerable time, may shuffle events during chaotic situations, and do not provide a convenient data structure for subsequent analysis. `DataBallPy` uses an earlier proposed Needleman-Wunsch algorithm to merge tracking and event data [@Kwiakowski] while optimizing it so it runs in mere seconds [@Oonk2025b] instead of ten minutes. Next to a state-of-the-art synchronization algorithm, `DataBallPy` offers a unified data structure for different data providers and an intuitive data structure for subsequent analysis.

# Software Design

A core design choice has been to center `DataBallPy` around the `Game` object. Instead of treating data as separate, independent streams (as in existing packages), the `Game` object consolidates all information for a single game: it consists of metadata, tracking, and event data. This unified approach reinforces the idea that combined information holds more value than independent sources. It also allows for simple usage by creating single methods that rely on all information available on the game without excessive user input. Finally, by combining all information in a single `Game` object, it becomes more convenient to deliver optimized functionalities. Some of the most important utilities of `DataBallPy` are listed below.

## Parsing Data

`DataBallPy` parses data from different commercial data providers such as Tracab, Metrica, Inmotio, Opta, Instat, SciSports, Sportec, and Statsbomb internally using the `get_game` function. The `Game` object contains the event and tracking data internally as Pandas dataframes, making them intuitive to work with [@reback2020pandas]. Alternatively, users can parse data from various providers using Kloppy and convert Kloppy event and tracking datasets into a `Game` object with the `get_game_from_kloppy` function. Last, `DataBallPy` has included a function to load openly available data directly in a `Game` object using `get_open_game`, which allows users who do not have access to data to still work with soccer data in `DataBallPy` [@Bassek2025]. As parsing and preprocessing a single game can take between a few seconds and several minutes on a standard device (comparable to other packages), `DataBallPy` allows users to efficiently save preprocessed `Game` objects as Parquet or JSON files. This offers two key benefits: (1) preprocessed `Game` objects can be loaded in milliseconds using `get_saved_game`, rather than minutes, and (2) raw tracking data files (up to 400 MB per game) are reduced to 20–100 MB when saved as `DataBallPy` objects, which include both event and tracking data.

## Preprocessing

Tracking data typically captured from video footage using computer vision. Depending on the quality and number of cameras, some noise may affect both athlete and ball positions [@Linke2020]. `DataBallPy` allows for filtering of the ball and positional data as well as differentiation of positions to compute velocity and acceleration. Furthermore, tracking data enables computation of individual athlete possession [@Vidal-Codina2022], while combined with event data, team-level possession can be estimated.

## synchronization

`DataBallPy` uses a soccer-specific implementation of the Needleman-Wunsch algorithm to synchronize the event and tracking data, which is more elaborately described in [@Oonk2025b]. The game can be synchronized using the following code
```python
>>> from databallpy import get_open_game
>>> game = get_open_game()
>>> game.synchronise_tracking_and_event_data()
```

## Performance Indicators

`DataBallPy` includes an extensive list of scientific features. All features can be computed with minimal code after obtaining a `Game` object. The documentation provides detailed explanations of the feature-computing code, enabling clear reporting and reproducibility. Using `DataBallPy`, the following features can be computed:

- Covered Distance (in specific velocity and acceleration zones) [@Jerome2024]
- Pressure [@Andrienko2017; @Herold2022]
- Individual player possession [@Vidal-Codina2022]
- Expected Goals [@Anzer2021]
- Expected Threat [@Singh2019]
- Voronoi Space Occupation [@Rein2017]
- Pitch Control [@Fernandez2018]
- Dangerous Accessible Space [@Bischofberger2025]

## Visualisation

![Example plot of soccer tracking data with pitch control heatmap as introduced in @Fernandez2018](fig1.png)

`DataBallPy` includes elaborate functionality to visualize the data in the `Game` object. Event locations can be visualized on a pitch using the `plot_events()` function, which allows for coloring of events by outcome, team, or event type during specific periods in the game. Similarly, the locations and velocities of all players can be plotted using the `plot_tracking_data()` function. If the event and tracking data are synchronized, users can overlay event information in the same plot. Other features like pitch control heatmaps, player possession, and any custom feature can also be visualized simultaneously with the event and tracking data (Figure 1). Finally, the tracking data (with heatmaps and custom features) can be transformed into a video (mp4) to show the true spatiotemporal progression over time.


```Python
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

# Research impact statement

`DataBallPy` has been increasingly adopted by developers, practitioners, and researchers. The project has over 60 GitHub stars. Issues and PRs are being opened by users outside of the network of the original owners and maintainers. Additionally, `DataBallPy` has been mentioned in numerous published scientific papers [@Robertson2023; @Anzer2025; @Zhang2025; @Oonk2025a]. Moreover, the largest currently open-sourced dataset of tracking and event data showcased how `DataBallPy` can be used to synchronize the two sources [@Bassek2025]. Finally, authors who introduce new metrics propose to open a PR with their metric so it is easily available for the scientific community [@Bischofberger2025]. Collectively, this demonstrates `DataBallPy`'s broad user base and its growth beyond the original mentainers' network.

# AI usage disclosure

Generative AI was used for reformulation of sentences in this manuscript. No generative AI tools were used in the development of the core functionalities and architecture of `DataBallPy`. Except for unittests, there is no explicit restriction on the usage of generative AI in the further development of `DataBallPy` (e.g., optimizing code, docstrings, reviewing, writing documentation, etc.). All code and documentation are checked and verified by human maintainers before merging into the code base.

# References
