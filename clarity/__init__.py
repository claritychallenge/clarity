"""pyClarity"""
from importlib_metadata import metadata

__version__ = metadata("clarity").get("version")
