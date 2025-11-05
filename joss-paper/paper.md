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
  - give-name: Matthias 
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

`DataBallPy` is a Python package designed to streamline the analysis of soccer matches by integrating and synchronizing event and tracking data. It provides a standardized interface for loading, preprocessing, and visualizing match data, enabling researchers and analysts to extract meaningful insights with minimal setup. By combining multiple data streams into a unified `Game` object, `DataBallPy` simplifies complex workflows and supports reproducible, data-driven soccer research. Last, DataBallpy is serious on documentation. Every preprocessing option and feature is elaborately explained in the  documentation. `DataBallPy` does not only help in computing scientific features, but the down-to-earth docoumentations helps you better understand how features are computed.

# Statement of need

Modern soccer analytics increasingly rely on both event data (e.g., passes, shots) and tracking data (e.g., player positions over time). While existing tools like `Kloppy` and `Floodlight` offer support for importing these data types, they often treat them separately [@kloppy, @floodlight]. `DataBallPy` addresses this gap by offering robust synchronization between event and tracking data using an soccer data specific Needleman-Wunsch algorithm, ensuring temporal alignment and preserving event order [@Oonk]. This integration is essential for developing advanced metrics and models, such as expected goals (xG), player movement analysis, and tactical evaluations. `DataBallPy` thus serves as a foundational tool for researchers and practitioners seeking to build reproducible and interpretable soccer analytics pipelines.

`DataBallPy` goes further by implementing core features that are neccessary for almost all soccer analytics project (e.g. determining which team and/or player has ball possession, filtering the tracking data, computing velocity and/or acceleration of players and the ball, etc.). All these features give any soccer analytics project a headstart. On top of that, it includes a built in way to visualise single frames of tracking data and create a mp4 file of a subset of the game for more in depth analysis.  


`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References