import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from clarity.utils.file_io import read_jsonl


class LogisticModel:
    """Class to represent a logistic mapping.

    Fits a logistic mapping from input values x to output values y.
    """

    params: Union[np.ndarray, None] = None  # The model params

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
    train_df = full_df[~full_df.signal.isin(test_df.signal)]
    train_df = train_df[~train_df.system.isin(test_df.system)]
    train_df = train_df[~train_df.listener.isin(test_df.listener)]
    assert not set(train_df.signal).intersection(set(test_df.signal))
    return train_df


def load_dataset_with_haspi(cfg, split: str) -> pd.DataFrame:
    """Load dataset and add HASPI scores."""
    dataset_filename = (
        Path(cfg.clarity_data_root) / cfg.dataset / "metadata" / f"CPC3.{split}.json"
    )
    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    # Load HASPI scores and add them to the records
    haspi_score = read_jsonl(f"{cfg.dataset}.{split}.haspi.jsonl")
    haspi_score_index = {record["signal"]: record["haspi"] for record in haspi_score}
    for record in records:
        record["haspi_score"] = haspi_score_index[record["signal"]]

    return pd.DataFrame(records)
