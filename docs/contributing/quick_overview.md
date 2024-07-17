# Contributing: Quick Overview

Contributing to a package means participating in the development and improvement of the package by offering your knowledge, skills, and time. By contributing, you become part of the community surrounding the package and help shape its future. Some common ways to contribute to a Python package are code contributions, issue reporting, and writing documentation. So even if you are not comfortable writing code yourself, you can contribute by opening issues whenever you find bugs, or would like to see improvements. Contributing to a project means working together on that project. For version control coders often use GitHub. GitHub is an online platform for hosting and managing code repositories, enabling collaboration and version control for software development projects. In the remainder of this chapter we will look at how to use GitHub to contribute to DataBallpy, how to use Poetry as our dependency manager and how to get your added code released in new versions of DataBallPy.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `databallpy` for local development.

1. Install [poetry](https://python-poetry.org/) on your device. 
2. Create a fork of the `databallpy` repository on GitHub.
3. Download a the fork  of `databallpy` locally.
4. Install `databallpy` dependencies using `poetry` by running one of the following commands:

    ```console
    $ poetry install
    $ make deps # for Linux/macOs
    $ poetry install --with docs,developing # for Windows
    ```
5. Use `git` (or similar) to create a branch for local development and make your changes (Make sure to first checkout to the development branch before checking out to your local brach):

    ```console
    $ git checkout -b <issue number>-<type of issue>-<short description of issue>
    $ # example for issue 5, a new features that normalizes playing direction in tracking data
    $ git checkout development
    $ git checkout -b 5-feat-normalize_playing_direction_td
    ```

6. Make changes to the code. Make sure to only change and add code that is within the scope of your issue. If you find any bugs or inconsistency on the way, please open a new issue in the git repository. This will help in keeping pull requests clear.

7. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.
      
8. Commit your changes and open a pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

  1. All tests should pass (add/update tests for added code if applicable): 
     
      ```console
      $ # for linux and macos
      $ make test
      $ # for windows
      $ poetry run py.test tests/ --pythonwarnings=once --cov-report=term-missing --cov-report=html --cov=databallpy --cov-config=.coveragerc
      ```

  2. All linters should pass:
      
      ```console
      $ # for linux and macos
      $ make formatlint
      $ # for windows
      $ poetry run isort --filter-files tests/ databallpy/
      $ poetry run black tests/ databallpy/
      $ poetry run flake8 tests/ databallpy/
      ```
  
  3. All documentation should be updated and run without errors:
        
      ```console
      $ # for linux and macos
      $ make documentation
      $ # for windows
      $ poetry run jupyter-book build docs/ --all
      ```

  4. All functions variables that are not intended to call for users should start with a `_`.
  5. All functions and classes should contain docstrings in the [google](https://github.com/NilsJPWerner/autoDocstring/blob/HEAD/docs/google.md) format.
  6. If applicable, for instance when adding a new feature, docs should be updated.
  7. The code should work for all currently supported operating systems and versions of Python
  8. Make sure to open a pull request to the `develop` branch. Do not open pull requests to the `main` branch unless you made a hot fix!

When opening the pull request, make sure to link to the approriate issue. For example, when opening a pull request that solves issue 5, add this in the title of the 
of the pull request and add in the description of the pull request `resolve #5`. This will close issue 5 automatically when the pull request is approved.

## Code of Conduct

Please note that the `databallpy` project is released with a
Code of Conduct. By contributing to this project you agree to abide by its terms.
