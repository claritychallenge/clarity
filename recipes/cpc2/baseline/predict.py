"""Make intelligibility predictions from HASPI scores."""
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


class LogisticModel:
    """Class to represent a logistic mapping.

    Fits a logistic mapping from input values x to output values y.
    """

    params = None  # The model params

    def _logistic_mapping(self, x, x_0, k):
        """
        Logistic function
            x_0 - x value of the logistic's midpoint
            k - the logistic growth rate or steepness of the curve
        """
        return 100.0 / (1 + np.exp(-k * (x - x_0)))

    def fit(self, x, y):
        """Fit a mapping from x values to y values."""
        initial_guess = [0.5, 1.0]  # Initial guess for parameter values
        self.params, *_pcov = curve_fit(self._logistic_mapping, x, y, initial_guess)

    def predict(self, x):
        """Predict y values given x."""
        # Note, fit() must be called before predictions can be made
        assert self.params is not None
        return self._logistic_mapping(x, self.params[0], self.params[1])


def read_jsonl(filename: str) -> list[dict]:
    """Read a jsonl file into a list of dictionaries."""
    with open(filename, "r", encoding="utf-8") as fp:
        records = [json.loads(line) for line in fp]
    return records


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config")
def predict(cfg: DictConfig):
    """Predict intelligibility from HASPI scores."""

    # Load the intelligibility dataset records
    dataset_filename = Path(cfg.path.metadata_dir) / f"{cfg.dataset}.json"
    with open(dataset_filename, "r", encoding="utf-8") as fp:
        records = json.load(fp)

    # load haspi scores and add them to the records
    haspi_score = read_jsonl(f"{cfg.dataset}.haspi.jsonl")

    haspi_score_index = {record["signal"]: record["haspi"] for record in haspi_score}
    for record in records:
        record["haspi_score"] = haspi_score_index[record["signal"]]

    # add split into train/test set...

    # fit logistic mapping from haspi score to correctness
    records_df = pd.DataFrame(records)
    model = LogisticModel()
    model.fit(records_df.haspi_score, records_df.correctness)

    # predict correctness from haspi scores using the model
    records_df["predicted_correctness"] = model.predict(records_df.haspi_score)

    # save results to csv file
    records_df[["signal", "predicted_correctness"]].to_csv(
        f"{cfg.dataset}.predict.csv",
        index=False,
        header=["signal_ID", "intelligibility_score"],
    )


if __name__ == "__main__":
    predict()
