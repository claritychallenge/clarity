"""Baseline enhancement for CAD2 task1."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    print(config)

    pass


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
