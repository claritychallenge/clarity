""" Run the dummy enhancement. """

# pylint: disable=too-many-locals
# pylint: disable=import-error
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig

from recipes.cad1.task2.baseline.enhance import enhance as enhance_set

logger = logging.getLogger(__name__)


def pack_submission(
    team_id: str,
    root_dir: str | Path,
    base_dir: str | Path = ".",
) -> None:
    """
    Pack the submission files into an archive file.

    Args:
        team_id (str): Team ID.
        root_dir (str | Path): Root directory of the archived file.
        base_dir (str | Path): Base directory to archive. Defaults to ".".
    """
    # Pack the submission files
    logger.info(f"Packing submission files for team {team_id}...")
    shutil.make_archive(
        f"submission_{team_id}",
        "zip",
        root_dir=root_dir,
        base_dir=base_dir,
    )


@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The baseline system is a dummy processor that returns the input signal.

    Args:
        config (dict): Dictionary of configuration options for enhancing music.
    """
    enhance_set(config)

    pack_submission(
        team_id=config.team_id,
        root_dir=Path("enhanced_signals"),
        base_dir=config.evaluate.split,
    )

    logger.info("Evaluation complete.!!")
    logger.info(
        f"Please, submit the file submission_{config.team_id}.zip to the challenge "
        "using the link provided. Thank you.!!"
    )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
