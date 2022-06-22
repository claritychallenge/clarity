from distutils.core import setup

import setuptools

setup(
    name="clarity",
    version="2.0.1",
    description="Tools for the Clarity Challenge",
    author="Clarity Team",
    packages=setuptools.find_packages(),
    install_requires=[
        "audioread",
        "hydra-core",
        "hydra-submitit-launcher",
        "librosa",
        "numpy",
        "omegaconf",
        "pandas",
        "pyloudnorm",
        "scikit-learn",
        "scipy",
        "SoundFile",
        "tqdm",
    ],
    python_requires=">=3.7",
    url="https://claritychallenge.github.io/clarity_CC_doc/",
)
