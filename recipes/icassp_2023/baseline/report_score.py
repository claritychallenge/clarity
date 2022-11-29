""" Run the dummy enhancement. """
import json
import logging

import hydra
import pandas as pd
from evaluate import make_scene_listener_list
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def report_score(cfg: DictConfig) -> None:
    """Run the dummy enhancement."""

    with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    results_df = pd.read_csv("scores.csv")

    # Make list of all scene listener pairs that are expected in results file
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )
    selection_df = pd.DataFrame(scene_listener_pairs, columns=["scene", "listener"])

    # Select the expected scene listener pairs from the results file
    selected_results_df = pd.merge(results_df, selection_df, on=["scene", "listener"])

    if len(selected_results_df) != len(selection_df):
        print("The following results were not found:")
        difference = pd.concat(
            [selected_results_df[["scene", "listener"]], selection_df]
        ).drop_duplicates(keep=False)
        print(difference)
    else:
        # All expected results were found so report the mean score
        print(f"Scores based on {len(selected_results_df)} scenes.")
        print(selected_results_df[["haspi", "hasqi", "combined"]].mean(axis=0))


if __name__ == "__main__":
    report_score()
