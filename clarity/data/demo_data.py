"""Functions for downloading demo data."""
import os
from pathlib import Path

import gdown

TARGET_DIR = "clarity_data/demo"
PACKAGE_NAME = "data.tgz"


def get_demo_data(metadata_url: str, target_dir: str) -> None:
    """Download demo data.

    Args:
        metadata_url (str): URL to download data from (should be a link on Google Drive)
        target_dir (str): Directory to save to (default 'clarity_data/demo'), will be
            created if it doesn't exist.

    Returns:
        None
    """
    gdown.download(metadata_url, PACKAGE_NAME, quiet=False)
    p = Path(target_dir)
    if p.exists() is False:
        p.mkdir(parents=True, exist_ok=True)
    os.system(f"tar -xvzf {PACKAGE_NAME} --directory {target_dir}/")
    os.system(f"rm {PACKAGE_NAME}")


def get_metadata_demo(target_dir: str = TARGET_DIR) -> None:
    """Download metadata."""
    url = (
        "https://drive.google.com/"
        "uc?export=download&id=14KGm2GaRgwlrZvtmMwWTYu7itaRVQV8f"
    )
    get_demo_data(url, target_dir)


def get_targets_demo(target_dir: str = TARGET_DIR) -> None:
    """Download targets."""
    url = (
        "https://drive.google.com/"
        "uc?export=download&id=1uu2Hes1fzqNaZSCiFNhxZM3bE_fAVKsD"
    )
    get_demo_data(url, target_dir)


def get_interferers_demo(target_dir: str = TARGET_DIR) -> None:
    """Download interferers."""
    url = (
        "https://drive.google.com/"
        "uc?export=download&id=1_ssD238Qv-EETzC0hJze7JhLE7bHyqwG"
    )
    get_demo_data(url, target_dir)


def get_rooms_demo(target_dir: str = TARGET_DIR) -> None:
    """Download rooms."""
    url = (
        "https://drive.google.com/"
        "uc?export=download&id=1FBC8DI4Ru-g3Set0fDzoKmXTqHqNXV8n"
    )
    get_demo_data(url, target_dir)


def get_scenes_demo(target_dir: str = TARGET_DIR) -> None:
    """Download secnes."""
    url = (
        "https://drive.google.com/"
        "uc?export=download&id=1PB0CfGXhpkYNk8HbE5lTWowm2016x6Hl"
    )
    get_demo_data(url, target_dir)


def get_hrirs_demo(target_dir: str = TARGET_DIR) -> None:
    """Download hiris."""
    url = (
        "https://drive.google.com/"
        "uc?export=download&id=1USrHLFhOE_jdAQcEKqumG3M5dEZVcwSd"
    )
    get_demo_data(url, target_dir)
