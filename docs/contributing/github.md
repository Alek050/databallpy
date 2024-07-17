# Getting Started with GitHub

GitHub is a version control service for software development projects. Getting started with GitHub might take some time, but eventually it becomes natural and is a very convenient tool to work together on the same project. The alternative is emailing code to others once you are finished, in that way you would have to copy and paste all changes manually, without any checks for if the code still works. This just is not suitable when you are working on bigger projects with multiple contributors. Learning GitHub workflows involves learning about commands like fork, commit, merge, clone, pull, push and a lot more is possible. We will start with branches.

```{image} https://i.redd.it/vlgt7ongcjo71.jpg
:alt: Git Meme
:width: 800px
:align: center
```

## Branches
Branches in GitHub serve as separate paths of development within a repository. When you create a branch, it acts as a copy of the main codebase, allowing you to work on specific features, bug fixes, or experiments without affecting the stability of the main branch. By creating branches, you can isolate your changes and work on them independently, providing a controlled environment for development. This way, multiple people can work on different branches simultaneously without interfering with each other's work. Branches are particularly useful for collaborative development. For example, if you are working on a new feature, you can create a dedicated branch for it. This allows you to make changes, test them, and iterate until the feature is complete and ready for integration. Once you have made the desired changes in a branch, you can initiate a pull request to propose merging those changes into the main branch. Pull requests facilitate collaboration and code review by allowing other team members to provide feedback, suggest improvements, and ensure the changes meet the project's requirements and coding standards.

Let's apply all this new theory to a new feature you'd like to add to DataBallPy. The default branch of DataBallPy is `main`, which is only changes when new versions are released. Below that we have the `develop` branch. All new features are added to `develop`, which is eventually merged into `main` when a new version is released. Below `develop`, there are different branches with small code changes. The structure will look something like this:

* Main
    * Develop
        * feature 1
        * feature 2
        * feature 3

Whenever feature 1 is finished, it will be added to the development branch, thus the structure will now look like this:

* Main
    * Develop (with feature 1 included)
        * feature 2
        * feature 3

As you can see, the `main` branch stays untouched, making sure that nothing changes for the users. The `develop` branch gets updated, so new features can be added in new releases. 

Now that you get the concept of branches, we will dive further into the GitHub rabbit whole based on the schematic overview below. Note that is just an introduction, not a full guide on how to use GitHub, there are plenty of sources online to learn how to use GitHub.

```{image} ../static/gitWorkflow.png
:alt: Git Workflow
:width: 800px
:align: center
```

## Creating Issues

In GitHub, an issue is a tool used for tracking and managing tasks, bugs, feature requests, and general discussions related to a repository or project. It serves as a centralized platform for communication and collaboration among developers, project managers, and users. An issue can be opened by anyone with access to the repository and allows individuals to report problems they encounter, suggest improvements, or initiate discussions on specific topics. Issues are typically used for the following purposes: reporting bugs, requesting new features, and for discussions/questions. To open an issue, or look at the existing once, you go to the main page of the project and click on "Issues". In the case of DataBallPy the main project is located [here](https://github.com/Alek050/databallpy). An issue has an id, title, description, and a label. These will come in handy later on. When opening an issue, it is of high importance to be very clear what you mean. For instance, if you want to report a bug because an unexpected error is raised, you need show the code that raises the error. If the maintainers can not reproduce the error, it becomes very difficult to fix the bug. Most of the times a short discussion will start on how the bug should be fixed or the new feature should look like. After the small discussion the issue is either further specified to change the code or closed because no changes to the code will be made. 

<br/>

## Forking the Repository

We will start in the grey box. This is the global environment, meaning that it is available online on GitHub. The Organization Repository is the repository where the original project is located. Your Personal gitHub Repository is what is your own GitHub page, most likely something like "https://github.com/{your GitHub name}". If you want to contribute, the first thing you have to do is fork the repository. Forking is essentially copying all the code from the original location of the code to your own GitHub account. You can do this with any public code and play with it without ever changing any of original code, however, if you want to contribute to DataBallPy, you in fact want to change the code. 

## Creating a New Branch

Right now, the code is in an online environment, namely your GitHub page, but you want to work on it in a local (offline) environment, the blue part of the image. To accomplish this, you need to clone your own repository. Cloning the code essentially downloads all the files of the project on to your disk, making it possible to make changes to the code. In this project, you have all the branches that are available in the original project. Since we want to keep the `main` and `develop` branch as stable as possible, you never directly change code on these branches, but create a new branch under the `develop` branch. So your first step is to `checkout` to the `develop` branch. Now you want to create a new branch. The new branch should get a name. The naming of the new branch has three components to ensure consistency over the whole project. The first component is the issue id, the second the type of change, and the third the name of the change. For instance, when issue 88 is open that requests a new feature to add accelerations of a player, the name of the branch would be `88-feat-player_accelerations`. To create this new branch you can use the GitHub Desktop program or enter `git checkout -b 88-feat-player_accelerations` in your terminal. The `-b` stands for create a new branch.

## Pull, Merge, Commit, and Push

Now is the time to make changes to the code. Continuing with the example to add accelerations, you would need to pull, merge, commit, and push changes. A pull is request to add global changes (on GitHub) to be merged into your local files. For instance when two people work on the same branch, locally, the pull request adds the changes of the other contributor to your local code. A clear example of this is when you have developed on a new feature for a longer time. In this case, the develop branch might have been updated with new code, lets say a small bug fix. However, this bug fix is not yet in your local branch. Therefore, it is good common practice to pull the develop branch from GitHub regularly, and if changes are added, merge develop into your local develop branch to make sure you are still up to date with all latest changes. A merge is the process of combining changes from one branch (in this case the develop branch) into another (your local branch), integrating the modifcations seamlessly. Sometimes, when merging changes to the same code you are changing locally, merge conflicts emerge. This means that GitHub does not know which changes to keep and discard, you have to change this manually. 

```{image} https://programmerhumor.io/wp-content/uploads/2023/04/programmerhumor-io-programming-memes-204cbca529a841d-758x314.jpg
:alt: Merge Conflicts
:width: 800px
:align: center
```

After you have made changes, the changes are only stored locally on your device and are not yet visible for any other users/contributors. It is generally good practice to make sure your code is up to date online, so other contributors can see what you have been doing and may help you. To do this, you first have to commit your changes. Committing is basically making a snapshot of changes made to your local branch. It is good practice to add a message to your commit, so it is clear what new code changes you have made. For instance, you could add a message "added doccstrings to acceleration function" However, committing is still local, it does not add anything to GitHub. To accomplish this, you have to push your code. A push command uploads and applies your committed changes to the remote repository, in this case your forked repository on GitHub. Now the code is available on your personal GitHub account and other coders can view and comment on your code. 

## Pull Request

You now have all your proposed changes for the new feature on your personal GitHub accounts, however, you of course want to add them to the code of the package. The last step is to open a pull request. A pull request is a formal way to propose changes to a codebase. It allows others to review your code, provide feedback, and ultimately merge the changes into the main repository of DataBallPy. You've now made clear you think your code is ready to be added to the main repository. The maintainers of the package will review your code and may ask for clarification or elaboration. Generally, this will take a few iterations of adding/removing code and rewriting the documentation and tests. Whenever they are satisfied with the code, they will approve and merge your changes to the main repository. In the next release of the package, your changes will be included! Congratulations, you are now a contributor to DataBallPy. 

Note that GitHub as a lot more options to work together and produce working, clean, and maintainable code. For this manual we will stop here, keeping it with the basics, but it is much recommended to go more in dept to everything GitHub has to offer. It really makes the collaboration process easier, less buggy, and more robust for any software development project. 

