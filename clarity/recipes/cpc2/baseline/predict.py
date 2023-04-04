"""Make intelligibility predictions from HASPI scores."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.optimize import curve_fit

from clarity.utils.file_io import read_jsonl

log = logging.getLogger(__name__)


class LogisticModel:
    """Class to represent a logistic mapping.

    Fits a logistic mapping from input values x to output values y.
    """

    params: np.ndarray | None = None  # The model params

    def _logistic_mapping(self, x, x_0, k):
        """Logistic function

        Args:
            x - the input value
            x_0 - logistic parameter: the x value of the logistic's midpoint
            k - logistic parameter: the growth rate of the curve

        Returns:
            The output of the logistic function.
        """
        return 100.0 / (1 + np.exp(-k * (x - x_0)))

    def fit(self, x, y):
        """Fit a mapping from x values to y values."""
        initial_guess = [0.5, 1.0]  # Initial guess for parameter values
        self.params, *_pcov = curve_fit(self._logistic_mapping, x, y, initial_guess)

    def predict(self, x):
        """Predict y values given x.

        Raises:
            TypeError: If the predict() method is called before fit().
        """
        if self.params is None:
            raise TypeError(
                "params is None. Logistic fit() must be called before predict()."
            )

        return self._logistic_mapping(x, self.params[0], self.params[1])


def make_disjoint_train_set(
    full_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Make a disjoint train set for given test samples."""
    # make sure that the train and test sets are disjoint
    # i.e. no signals, systems or listeners are shared
    train_df = full_df[~full_df.signal.isin(test_df.signal)]
    train_df = train_df[~train_df.system.isin(test_df.system)]
    train_df = train_df[~train_df.listener.isin(test_df.listener)]
    assert not set(train_df.signal).intersection(set(test_df.signal))
    return train_df


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config")
def predict(cfg: DictConfig):
    """Predict intelligibility from HASPI scores."""

    # Load the intelligibility dataset records
    dataset_filename = Path(cfg.path.metadata_dir) / f"{cfg.dataset}.json"
    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    # load haspi scores and add them to the records
    haspi_score = read_jsonl(f"{cfg.dataset}.haspi.jsonl")

    haspi_score_index = {record["signal"]: record["haspi"] for record in haspi_score}
    for record in records:
        record["haspi_score"] = haspi_score_index[record["signal"]]
    records_df = pd.DataFrame(records)

    # make predictions for each item in the data
    for i, _record in records_df.iterrows():
        test_df = records_df.iloc[[i]].copy()

        # The prediction is made using a logistic mapping from HASPI scores to
        # intelligibility. It is important that this mapping is trained using a
        # disjoint set of data, i.e. we define a training data set that does not
        # contain systems, listeners or signals that appear in the test data sample.

        train_df = make_disjoint_train_set(records_df, test_df)

        model = LogisticModel()
        model.fit(train_df.haspi_score, train_df.correctness)

        # predict correctness from haspi scores using the model
        test_df["predicted_correctness"] = model.predict(test_df.haspi_score)

        # save results to csv file
        header, mode = (
            (["signal_ID", "intelligibility_score"], "w") if i == 0 else (False, "a")
        )
        test_df[["signal", "predicted_correctness"]].to_csv(
            f"{cfg.dataset}.predict.csv",
            index=False,
            header=header,
            mode=mode,
        )


if __name__ == "__main__":
    predict()
