"""Collate HASPI scores and report final results."""

import json
from pathlib import Path

import hydra
import pandas as pd

from recipes.icassp_2023.baseline.evaluate import make_scene_listener_list


def load_scenes_data(json_file):
    """Load the scenes data from a JSON file."""
    with open(json_file, encoding="utf8") as f:
        scenes_data = json.load(f)
    return scenes_data


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    """Score the evaluation results."""

    # Read the HASPI score
    results_dir = Path(cfg.path.exp) / "scores"
    dfs = [pd.read_csv(file) for file in results_dir.glob("*.csv")]
    results = pd.concat(dfs, ignore_index=True)

    # Make list of all scene listener pairs that should have been run
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    # Load the scenes data
    scenes_data = load_scenes_data(cfg.path.scenes_file)
    scenes_dict = {scene["scene"]: scene for scene in scenes_data}

    # Show results as overall mean and by SNR
    results["SNR"] = results["scene"].apply(lambda x: scenes_dict[x]["SNR"])
    bins = pd.cut(results["SNR"], bins=range(-12, 7, 3))
    print(f"Evaluation set size: {len(results)}")
    print(f"Mean HASPI score: {results['haspi'].mean()}")
    print()
    print(results.groupby(bins)[["SNR", "haspi"]].mean())
    print()
    # Validate data and warn if looks incomplete or non-standard
    if len(results) != len(scene_listener_pairs):
        print(
            f"WARNING: Expected {len(scene_listener_pairs)} scores, "
            f"but found {len(results)}. The evaluation is incomplete."
        )
    # if correct length then check it matches the expected pairs
    desired_pairs = {(rec.scene, rec.listener) for rec in results.itertuples()}
    evaluated_pairs = set(scene_listener_pairs)
    if desired_pairs != evaluated_pairs:
        print(
            "WARNING: The evaluation has not been performed"
            "with the standard evaluation set."
        )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
