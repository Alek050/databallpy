# Dependency Managing: Poetry

Whenever you are writing code, you generally use external packages. In Python, common packages are `numpy` and `pandas`. A package is just a fancy word for a directory that contains code. Whenever you install a package, it stores the code of that package locally on your device, making it possible to use the code in your own project. Somethimes this code gets updated, to add new features or remove bugs. In this case, a new version of the package is published. When using one package, there is not really an issue, but if you start using more packages, issues might occur. For instance, the package `pandas` relies on `numpy`, let's say version 1.0.0. If you install `pandas`, `numpy` version `1.0.0` is automatically also installed to ensure `pandas` is working properly. However, if your project also uses `numpy`, but relies on a feature that is only available on version 2.0.0, problems occur. In short, when you import `numpy`, Python does not know if it has to import version 1.0.0 or 2.0.0, thus it raises an error. This raises an interesting dillema for when you are writing packages. You want to make sure that all packages are compatible together, but also that the range of package that can be used by the user is as wide as possible. You can imagine that this problem becomes quadratically more difficult for every package you add. Secondly, what happens when you are working on two different projects? If you are working on DataBallPy and on Project2 on the same device, problems might occur as well. Similarly, DataBallPy might rely on `numpy` 1.0.0 and Project2 on `numpy` 2.0.0. Poetry can solve both (and more) these problems.

Poetry is in its basis a dependency manager: it manages the package depencies of your project. This solves the first problem. Poetry is able to find out how all packages are compatible together, and make the version of packages that can be used as wide as possible. The second problem can be tackled with virtual environments, which is incorporated in Poetry. Poetry creates isolated virtual environments for your projects, making it possible to work on multiple project with different depencies of the same package on your local device. Poetry is a high level framework that automates a lot of these processes. Installing Poetry might be a bit harder than simply installing packages, but it will definitely make the workflow and compatibility of the package smoother in the future. Using Poetry is obligatory when contributing to DataBallPy. Here follow 3 steps to install Poetry for Windows:

```{image} https://programmerhumor.io/wp-content/uploads/2023/07/programmerhumor-io-programming-memes-0c335885632c096.jpg
:alt: deps
:width: 800px
:align: center
```

## Step 1: Download and Install Poetry
1. Open Command Prompt (CMD) or PowerShell as an administrator.
2. Use the following command to download and install Poetry using the official installer script:
`curl -sSL https://install.python-poetry.org | python3 -`

## Step 2: Add Poetry to PATH
After the installation, you need to add Poetry to your PATH. Adding Poetry to the PATH allows you to run the Poetry command from any command prompt. The PATH is an environment variable that contains a list of directories where the operating system looks for executable files. By adding Poetry to the PATH, you can run the Poetry command from any directory without specifying the full path to the executable.

### Windows
By default, Poetry is installed in the user's 'AppData\Roaming\Python\Scripts' directory.
1. Find the installation path:
- Open a new Command Prompt (CMD) or PowerShell window.
- Run the following command to find out where Poetry is installed:
`poetry where`
- The typical output will be something like 'C:\Users\YourUserName\AppData\Local\Programs\Python\Python310\Scripts\poetry.exe'.
2. Add the installation path to the PATH environment variable:
- Open the Start Menu, search for "System properties", and select "Advanced", see 1 in the figure below. Then click "Environment Variables", see 2 in the figure below. 
- In the Environment Variables window, find the "Path" variable under "User variables" (see 3 in the figure below) and select it. Then click "Edit...", see 4 in the figure below.
- In the Edit Environment Variable window, click "New" (see 5) and paste the path you got from the previous command, e.g., 'C:\Users\YourUserName\AppData\Local\Programs\Python\Python310\Scripts\' (see 6, in my case, the path where poetry.exe is installed, is already added to the system)
- Click "OK" to close each window.

```{image} ../static/add_to_path.png
:alt: Add Poetry to PATH
:width: 800px
:align: center
```

### MacOs
By default, Poetry is installed in the user's '~/.local/bin' directory.
1. Find the installation path: if not specified differently, it is the location mentioned above.
2. Create or open a new file in your home directory called `.zshrc` or `.bashrc` (if you are using bash) and add the following line:
`export PATH="$HOME/.local/bin:$PATH"`. This will add the Poetry executable to your PATH. Save the file. This file will be executed every time you open a new terminal window.

## Step 3: Verify the Installation
1. (Re-)open your command prompt (CMD) or PowerShell window to apply the changes to the PATH.
2. Run the following command to verify that Poetry is installed and accessible from any command prompt:
`poetry --version`
You should see the version of Poetry that was installed.

See [their website](https://python-poetry.org/docs/) for more information on the installation and workflow.
