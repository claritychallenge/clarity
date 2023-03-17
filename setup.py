"""Package setup"""
import versioneer
from setuptools import setup

setup(
    version=versioneer.get_version(),  # pylint: disable=no-member
    cmdclass=versioneer.get_cmdclass(),  # pylint: disable=no-member
)
