import re
from io import open
from os import path

from setuptools import find_packages, setup

requirements = [
    "audioread>=2.1.9",
    "hydra-core==1.1.1",  # pinned, hydra 1.2 requires some changes
    "hydra-submitit-launcher>=1.1.6",
    "librosa>=0.8.1",
    "matplotlib",
    "numpy>=1.21.6",
    "omegaconf>=2.1.1",
    "pandas>=1.3.5",
    "pyloudnorm>=0.1.0",
    "scikit-learn>=1.0.2",
    "scipy>=1.7.3",
    "SoundFile>=0.10.3.post1",
    "tqdm>=4.62.3",
]

dev_requirements = [
    "black==19.10b0",
    "coverage",
    "flake8",
    "flake8-print",
    "isort",
    "mypy",
    "pre-commit",
    "pycodestyle",
    "pytest",
    "pytest-regtest",
    "pytest-cov",
    "yamllint",
]


# Get version
def read(*names, **kwargs):
    with open(
        path.join(path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("clarity", "__init__.py")


setup(
    name="pyclarity",
    version=version,
    description="Tools for the Clarity Challenge",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="The PyClarity team",
    author_email="clarity-group@sheffield.ac.uk",
    project_urls={
        "Bug Tracker": "https://github.com/claritychallenge/clarity/issues",
        "Documentation": "https://claritychallenge.github.io/clarity_CC_doc",
        "Source": "https://github.com/claritychallenge/clarity",
    },
    packages=find_packages(exclude=("docs", "examples", "tests")),
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    setup_requires=["setuptools>=38.6.0"],
    license="MIT",
    keywords="machine learning, challenges, speech-enhancement, hearing-aids",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Hearing Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
    ],
    url="https://github.com/claritychallenge/clarity",
)
