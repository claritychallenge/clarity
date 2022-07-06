# Installation

It is recommended that you install and use Clarity under a [Virtual
Environment](https://realpython.com/python-virtual-environments-a-primer/). If you opt to use Conda it is recommended
that you use the minimal [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version which avoids installing
tools you won't use.



Installing Python packages relies on the [`pip`](https://pip.pypa.io/en/stable/) package which should be available in your virtual environment.


## PyPi

Clarity is available on [PyPI](https://pypi.org/project/clarity/) to install the latest stable release...

``` bash
pip install clarity
```

## GitHub

You can alternatively install the latest development version from GitHub. There are two methods for doing so.

### Pip GitHub install

To install the `main` brand directly from GitHub using `pip`
``` bash
pip install git+https://github.com/claritychallenge/clarity.git@main
```

### Manual Cloning and Installation

You will have to have [git])(https://git-scm.com) installed on your system to install in this manner.

``` bash
git clone git@github.com:claritychallenge/clarity.git
cd clarity
pip install -e .
```
