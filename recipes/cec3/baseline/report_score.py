"""Collate HASPI scores and report final results."""

import json
from pathlib import Path

import hydra
import pandas as pd


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


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
