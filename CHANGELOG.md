# Changelog

## Version 0.6

## V0.6.0 (DATE)

- Moved from function to an object oriented framework for all user-features and computations of game/match (special thanks to [DaanGro](https://github.com/DaanGro))
- Renamed the all classes and functions with `match` to `game` (to move away from the internal python `match` statement)
- Removed the function to save game/match objects to pickle, but created a more save way using parquet and json files
- Added functionality to export tracking data to long format.
- the `"ball_possession"` column in the tracking data (`game.tracking_data`) was renamed to `"team_posession"` to be more explicit that it is about a team possession and not an individual possession.

#### Breaking changes
We sincerely appologize for all the changes you have to make, but we feel this will make the package more robust and easier to use for future projects. Just to be clear, **all the functionality that was in 0.5.4, is still in 0.6.0**. However we made to changes that impacts users.
1) We renamed all functions with `match` in it to `game.` (e.g. `get_match` was changed to `get_game`). This was chosen since `match` is an internal python command, and we do not want to imply to overwrite that (by using something like `match = get_match()`). A deprecation warning is raised when you try to call it from the current version onwards, we strongly encourage to take this warning serious as we do plan to remove it shorlty.
2) If you used any of the features in databallpy (`get_velocity`, `get_approximate_voronoi`, `get_covered_distance`, etc.), you need to refactor your code. All functionality is still available in the package, but likely as a method on the `game.tracking_data` class, for instance: `game.tracking_data.add_velocity(...)`. Please view our documentation of version 0.6.0 to see how you can refactor your code (spoiler: you will need to use less arguments and less lines of code!)
3) Saved games/matches are only usable in the version in which you saved them up and untill version 0.5.4. For example, if you saved your match in version 0.5.2, you can only load it while using version 0.5.2. From version 0.6.0, you will be able to load your game using the `get_saved_game` as long as your version is greater or equal to 0.6.0.

## Version 0.5

### V0.5.4 (19/03/2025)

- Support for Python 3.13
- Added event id of StatsBomb in event data (by [jan-swiatek](https://github.com/jan-swiatek))
- Fixed bug in sportec tracking data parser.

### V0.5.3 (10/12/2024)

- Added StatsBomb event data as provider (by [DaanGro](https://github.com/DaanGro))
- Added json parser for tracab metadata (by [jan-swiatek](https://github.com/jan-swiatek))
- Made last batch from smart batches in synchronisation last longer to include all events
- Fixed bug in parsing tracab xml data
- Fixed bug in combining players info from tracking and event metadata

### V0.5.2 (15/11/2024)

- Added `match.get_column_ids` with filters for team, player positions and minimal minutes played filters.
- Added parser for DFL/Sportec Solutions event data
- Added parser for Tracab xml format, used by DFL and Sportec solutions
- Added integration for open data from DFL (open sourced by Bassek et al.)

#### Breaking changes
- From now on, `match.home_players_column_ids()` and `match.away_players_column_ids()` are depreciated and will be removed in V0.7.0. Please use `match.get_column_ids()` in future version.
- `get_open_match()` will now, by default, load in match `J03WMX` (1. FC Köln vs. FC Bayern München) instead of the anonimysed match from Metrica. To load in the metrica match, please parse `provider="metrica"` in the key word arguments.

### V0.5.1 (10/10/2024)

- Added support for numpy 2.x
- Fixed bug in calculating covered distances when no accelerations are present
- Fixed bug in rare cases where _filter_data does not work as expected and returns errors
- Fixed bug in match.add_tracking_data_features_to_shots, did not update properly after last release

### V0.5.0 (30/07/2024)

- Added event data parser for SciSports
- Added expected threat model from Karun Singh to on ball events
- Add function to add which team has ball possession based on the synchronised event data.
- Solved encoding bug for Windows
- Added function to obtain acceleration
- Improved standardization and performance of synchronisation
- Made save match clip more robust
- Added function to calculate covered distances in different velocity and acceleration zones
- Updated the documentation

## Version 0.4

### V0.4.3 (15/1/2024)

- Added simple expected goals (xG) model
- Added anonimisation function
- Added pitch control model
- Fixed minor bugs
- Added logging possibilities
- Refactor of file structure and folder naming

### V0.4.2 (27/10/2023)

- fixed bug in get_valid_gains (player possessions)
- Changed 'period' to 'period_id' in all tracking and event dataframes
- Solved bug with reading in timestamps for Opta
- Made get_opponents_in_passing_lane more robust.
- Made function to automatically check if the proposed start and end frames are reliable, adjusts it if this is not the case
- Fixed bug in normalize_playing_direction
- Added an offset to the alignment in datetimes between tracking and event data
- Added extra sanity checks for the ball status of the tracking data
- Set "smart" as default for synchronising the tracking and event data
- Added check if datetime of tracking and event data are not close enough
- Added extra tests for the code
- Added extra metadata to the Match object on which periods the player positions were normalized

### V0.4.1 (10/10/2023)

- Fixed bugs with reading in tracab data in special cases
- Fixed bus with reading in opta data in special cases
- Added tracking data passes features
- Added tracking data shot features
- optimized performance of processing data

### V0.4.0 (21/09/2023)

- Added databallpy_events to get a unified type of event data
- Added individual player possessions (Adrienko et al., 2016)
- Added pressure feature (Herrold et al., 2022)
- Added differentiate features with filters
- Optimized loading in tracab tracking data
- In quality checks, fixed some minor bugs.

# Version 0.3

### V0.3.0 (02/06/2023)

- Added way to save Match objects, and to load saved Match objects
- Fixed bug in opta event data, own goals are now parsed as seperate event type
- Added parser for Inmotio tracking data
- Added parser for Instat event data
- Added quality checks for the data, raises warning if quality is not good enough

<<<<<<< HEAD
## Version 0.2

### v0.2.0 (10/03/2023)

- Added parser for Metrica, including an open dataset
- Added functionality to synchronize tracking and event data
- Added functionality to plot events
- Fixed bug, now both tracking and event data are normalized in direction
- Fixed unexpected behaviour, all date related objects are now datetime objects

## Version 0.1

### v0.1.1 (10/02/2023)

- Hot fix to make documentation visible.

### v0.1.0 (10/02/2023)

- First version of `databallpy` with utilities! You can now read in data from tracab and opta and create short videos of the tracking data!

### v0.0.1 (24/01/2023)

- First release of `databallpy`!
=======
## V0.5.2 (15/11/2024)

- Added `match.get_column_ids` with filters for team, player positions and minimal minutes played filters.
- Added parser for DFL/Sportec Solutions event data
- Added parser for Tracab xml format, used by DFL and Sportec solutions
- Added integration for open data from DFL (open sourced by Bassek et al.)

### Breaking changes
- From now on, `match.home_players_column_ids()` and `match.away_players_column_ids()` are depreciated and will be removed in V0.7.0. Please use `match.get_column_ids()` in future version.
- `get_open_match()` will now, by default, load in match `J03WMX` (1. FC Köln vs. FC Bayern München) instead of the anonimysed match from Metrica. To load in the metrica match, please parse `provider="metrica"` in the key word arguments.


## V0.5.3 (10/12/2024)

- Added StatsBomb event data as provider (by [DaanGro](https://github.com/DaanGro))
- Added json parser for tracab metadata (by [jan-swiatek](https://github.com/jan-swiatek))
- Made last batch from smart batches in synchronisation last longer to include all events
- Fixed bug in parsing tracab xml data
- Fixed bug in combining players info from tracking and event metadata

## V0.5.4 (19/03/2025)

- Support for Python 3.13
- Added event id of StatsBomb in event data (by [jan-swiatek](https://github.com/jan-swiatek))
- Fixed bug in sportec tracking data parser.
>>>>>>> main
