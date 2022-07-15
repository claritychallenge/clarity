# Contributing to Clarity


## Using Clarity

You can contribute to the development of Clarity simply by using it. If something isn't clear then please do ask
questions in the [Discussion](). If you find errors when running the code then please report them using the [Issues Bug
Template](), or if there is a feature or improvement you think Clarity would benefit from then suggest it using the
[Issues Feature Template]().

If you are new to GitHub and working collaboratively you may find the [GitHub Issues](https://docs.github.com/en/issues)
documentation useful.

## Contributing Code

You are welcome to [fork]() the repository and address bugs and features yourself and then create a [pull request](),
however there are a number of practices and standards we ask that you adhere to in order to maintain the quality and
maintainability of the code base.

### Use a Virtual Environment

It is recommended that you use a [Virtual Environment]().

### Install Development Dependencies

Before undertaking development you should install the development requirements defined by Clarity.  To do so you can do one
of the following

``` bash
# Install from Pip
pip install clarity.[dev]
# Install from GitHub using Pip
pip install git+ssh://github.com:claritychallenge/clarity.[dev]
# Install from local Git repository
pip install '.[dev]'
```

### Create an issue

Before undertaking any development please create an [Issue]() for it here on the Clarity repository. There are templates
for [Bug Reports]() and [Feature Requests](). This allows maintainers to keep an overview of what work is being
undertaken and gives you the opportunity to discuss with them your intended solutions.

### Create a Fork

Once you have created an issue you can
[fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks) the
Clarity repository to your own account and create a branch on which to undertake development.


### Coding Style

We ask that you adhere to the [PEP8](https://pep8.org/) coding style when writing your code. To facilitate this we use
[flake8](https://flake8.pycqa.org/en/latest/) and [isort](https://pycqa.github.io/isort/). Most popular Integrated
Development Environments (IDEs) include plugins that will apply flake8 to your code on saving a file.

For information on how to configure some popular IDEs see the following links.

* [Linting Python in Visual Studio Code](https://code.visualstudio.com/docs/python/linting)
* [Flake8 support - PyCharm Plugin](https://plugins.jetbrains.com/plugin/11563-flake8-support)
* [Emacs flymake-python-pyflakes](https://github.com/purcell/flymake-python-pyflakes/)


Further we have implemented a [`pre-commit`](https://pre-commit.com/) hook to ensure flake8 and isort are applied each
time you make a commit.
