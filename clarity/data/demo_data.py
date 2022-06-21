import os
from pathlib import Path

import gdown

TARGET_DIR = "clarity_data/demo"
PACKAGE_NAME = "data.tgz"


def get_demo_data(metadata_url, target_dir):
    gdown.download(metadata_url, PACKAGE_NAME, quiet=False)
    p = Path(target_dir)
    if p.exists() is False:
        p.mkdir(parents=True, exist_ok=True)
    os.system(f"tar -xvzf {PACKAGE_NAME} --directory {target_dir}/")
    os.system(f"rm {PACKAGE_NAME}")


def get_metadata_demo(target_dir=TARGET_DIR):
    url = "https://drive.google.com/uc?export=download&id=14KGm2GaRgwlrZvtmMwWTYu7itaRVQV8f"
    get_demo_data(url, target_dir)


def get_targets_demo(target_dir=TARGET_DIR):
    url = "https://drive.google.com/uc?export=download&id=1uu2Hes1fzqNaZSCiFNhxZM3bE_fAVKsD"
    get_demo_data(url, target_dir)


def get_interferers_demo(target_dir=TARGET_DIR):
    url = "https://drive.google.com/uc?export=download&id=1_ssD238Qv-EETzC0hJze7JhLE7bHyqwG"
    get_demo_data(url, target_dir)


def get_rooms_demo(target_dir=TARGET_DIR):
    url = "https://drive.google.com/uc?export=download&id=1FBC8DI4Ru-g3Set0fDzoKmXTqHqNXV8n"
    get_demo_data(url, target_dir)


def get_scenes_demo(target_dir=TARGET_DIR):
    url = "https://drive.google.com/uc?export=download&id=1PB0CfGXhpkYNk8HbE5lTWowm2016x6Hl"
    get_demo_data(url, target_dir)


def get_hrirs_demo(target_dir=TARGET_DIR):
    url = "https://drive.google.com/uc?export=download&id=1USrHLFhOE_jdAQcEKqumG3M5dEZVcwSd"
    get_demo_data(url, target_dir)
