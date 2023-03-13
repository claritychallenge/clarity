# Installation

It is recommended that you install and use Clarity under a [Virtual
Environment](https://realpython.com/python-virtual-environments-a-primer/). If you opt to use Conda it is recommended
that you use the minimal [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version which avoids installing
tools you won't use. Once you have installed Conda you can create and activate a virtual environment by...

``` bash
cond create --name clarity python=3.8
conda activate clarity
```

The following steps assume that you have activated the `clarity` virtual environment.

## PyPi

The latest stable release of Clarity is available on [PyPI](https://pypi.org/project/pyclarity/) and can be installed
using `pip`. To install simply...

``` bash
pip install pyclarity
```

## GitHub

You can alternatively install the latest development version from GitHub. There are two methods for doing so.

### Pip GitHub install

To install the `main` branch directly from GitHub using `pip`...

``` bash
pip install -e git+https://github.com/claritychallenge/clarity.git@main
```

### Manual Cloning and Installation

You will have to have [git](https://git-scm.com) installed on your system to install in this manner.

``` bash
git clone git@github.com:claritychallenge/clarity.git
cd clarity
pip install -e .
```
