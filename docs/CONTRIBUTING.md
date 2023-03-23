# Contributing to pyClarity

We welcome and encourage you to contribute to the development of the pyClarity code base. These guidelines outline how you
can do so.

## Using pyClarity

You can contribute to the development of pyClarity simply by using it. If something isn't clear then please do ask
questions in the [Discussion](https://github.com/claritychallenge/clarity/discussions). If you find errors when running the code then please report them using the [Issues Bug
Template](https://github.com/claritychallenge/clarity/issues/new?assignees=&labels=bug%2C+question&template=bug_report.md&title=%5BBUG%5D), or if there is a feature or improvement you think pyClarity would benefit from then suggest it using the
[Issues Feature Template](https://github.com/claritychallenge/clarity/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFEATURE%5D).

If you are new to GitHub and working collaboratively you may find the [GitHub Issues](https://docs.github.com/en/issues)
documentation useful.

## Contributing Code

If you have algorithms or code that you would like to contribute then please get in touch by emailing us at
[claritychallengecontact@gmail.com](mailto:claritychallengecontact@gmail.com). We will be happy to help you integrate
your contribution into the pyClarity framework; we can even help translate contributions from other languages, e.g. MATLAB.

You are also very welcome to [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository and address bugs
and features yourself and then create a [pull request from the
fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).
However, there are a number of practices and standards we ask that you adhere to in order to maintain the quality and
maintainability of the code base.

### Use a Virtual Environment

It is recommended that you use a [Virtual Environment](https://realpython.com/python-virtual-environments-a-primer/) to
undertake development.

### Install Development Dependencies

Before undertaking development you should install the development requirements defined by pyClarity.  To do so you can do one
of the following

``` bash
# Install from Pip
pip install pyclarity.[dev]
# Install from GitHub using Pip
pip install git+ssh://github.com:claritychallenge/clarity.[dev]
# Install from local cloned Git repository
pip install '.[dev]'
```

### Create an issue

Before undertaking any development please create an [Issue](https://github.com/claritychallenge/clarity/issues) for it
here on the pyClarity repository. There are templates for [Bug
Reports](https://github.com/claritychallenge/clarity/issues/new?assignees=&labels=bug&template=bug_report.md&title=%5BBUG%5D)
and [Feature
Requests](https://github.com/claritychallenge/clarity/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFEATURE%5D). This
allows maintainers to keep an overview of what work is being undertaken and gives you the opportunity to discuss with
them your intended solutions.

### Create a Fork

Once you have created an issue you can
[fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks) the
pyClarity repository to your own account and create a branch on which to undertake development.

## Coding Style

We ask that you adhere to the [PEP8](https://pep8.org/) coding style when writing your code. To facilitate this we use a
number of linting tools.

* [flake8](https://flake8.pycqa.org/en/latest/)
* [black](https://black.readthedocs.io/en/stable/index.html)
* [isort](https://pycqa.github.io/isort/)
* [mypy](https://mypy.readthedocs.io/en/stable/)
* [pylint](pylint.org)

### pre-commit

To ensure these coding conventions are applied to code that is submitted to the `main` branch PyClarity uses
[`pre-commit`](https://pre-commit.com/) and [`pre-commit.ci`](https://pre-commit.ci) which run Git Hooks before each
commit is made (the former locally, the latter on pull requests) for the above linting (and more e.g. Markdown style is
also checked).

Whilst the `pre-commit` Python package will have been installed in your environment when you `pip install .[dev]` you
need to install the configured hooks (which are defined in
[`.precommit-config.yaml`](https://github.com/claritychallenge/clarity/blob/main/.pre-commit-config.yaml)) in your local
copy of the PyClarity repository. To do so run the following from the `clarity` directory...

``` bash
pre-commit install
```

This installs `.git/hooks/pre-commit` commit hook which is triggered each time you make a new commit.

If your commit fails to pass the checks please read the error messages carefully. Some changes are fixed automatically
(e.g. `black` will format files in place where it can), but not all changes can be fixed and you should read the output
carefully  to find out what you need to manually fix.

### `pathlib` over `os`

PyClarity uses the object-orientated package [`pathlib`](https://docs.python.org/3/library/pathlib.html)
throughout the code base for any operations involving the file system and path components as it makes code easier to
read. We ask that any contributions follow this convention and use `pathlib` rather than `os`. For more on `pathlib` see
[1] and [2].

### IDEs

Most popular Integrated Development Environments (IDEs) include plugins that will apply `flake8` / `black` to your code
on saving a file.

For information on how to configure some popular IDEs see the following links.

* [Linting Python in Visual Studio Code](https://code.visualstudio.com/docs/python/linting)
* [Flake8 support - PyCharm Plugin](https://plugins.jetbrains.com/plugin/11563-flake8-support)
* [Emacs flymake-python-pyflakes](https://github.com/purcell/flymake-python-pyflakes/)

## Testing

All new code should be covered by [unit
tests](https://carpentries-incubator.github.io/python-testing/04-units/index.html) with at least 70% coverage and where
appropriate [regression tests](https://carpentries-incubator.github.io/python-testing/07-integration/index.html). This
includes bug fixes, when a test should be added that captures the bug.

The [pytest](https://docs.pytest.org/en/7.1.x/) framework is used and the conventions are followed under
pyClarity. Tests reside under the `tests` directory (and optionally within a module directory), fixtures should be defined
in `conftest.py` files whilst resources used should be placed under `tests/resources`.

The Continuous Integration in place on GitHub Actions runs `pytest` on newly created pull-requests and these have to
pass successfully before merging so it is useful to ensure they pass before you create pull requests at the very
least. Sometimes it may be sensible to run them against commits too.

To run the tests you should install the additional dependencies and then call `pytest` from the root of the project
folder.

``` bash
pip install .[tests]
pytest
```

[1]: https://treyhunner.com/2018/12/why-you-should-be-using-pathlib/
[2]: https://treyhunner.com/2019/01/no-really-pathlib-is-great/
