import json
import logging
import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr

logger = logging.getLogger(__name__)


def rmse_score(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def ncc_score(x, y):
    return pearsonr(x, y)[0]


def kt_score(x, y):
    return kendalltau(x, y)[0]


def std_err(x, y):
    return np.std(x - y) / np.sqrt(len(x))


class Model:
    """Class to represent the mapping from mbstoi parameters to intelligibility scores.
    The mapping uses a simple logistic function scaled between 0 and 100.
    The mapping parameters need to fit first using mbstoi, intelligibility score pairs, using fit().
    Once the fit has been made predictions can be made by calling predict()
    """

    params = None  # The model params

    def _logistic_mapping(self, x, x0, k):
        """
        Logistic function
            x0 - x value of the logistic's midpoint
            k - the logistic growth rate or steepness of the curve
        """
        L = 100  # correctness can't be over 100
        return L / (1 + np.exp(-k * (x - x0)))

    def fit(self, pred, intel):
        """Fit a mapping betweeen mbstoi scores and intelligibility scores."""
        initial_guess = [0.5, 1.0]  # Initial guess for parameter values
        self.params, pcov = curve_fit(
            self._logistic_mapping, pred, intel, initial_guess
        )

    def predict(self, x):
        """Predict intelligilbity scores from mbstoi scores."""
        # Note, fit() must be called before predictions can be made
        assert self.params is not None
        return self._logistic_mapping(x, self.params[0], self.params[1])


def compute_scores(predictions, labels):
    return {
        "RMSE": rmse_score(predictions, labels),
        "Std": std_err(predictions, labels),
        "NCC": ncc_score(predictions, labels),
        "KT": kt_score(predictions, labels),
    }


def read_data(pred_csv, label_json):
    df_pred = pd.read_csv(pred_csv).rename(
        columns={"signal_ID": "signal", "intelligibility_score": "prediction"}
    )
    df_label = pd.read_json(label_json).rename(columns={"correctness": "label"})
    data = df_pred.merge(df_label[["signal", "label"]])
    data["prediction"] = data["prediction"].apply(lambda x: x * 100)
    return data


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    logger.info("Run evaluation on the closed set.")
    data_tr = read_data(
        pred_csv=os.path.join(cfg.train_path.exp_folder, "sii.csv"),
        label_json=cfg.train_path.scenes_file,
    )
    data_eval = read_data(
        pred_csv=os.path.join(cfg.test_path.exp_folder, "sii.csv"),
        label_json="../test_listener_responses/CPC1.test.json",
    )
    logger.info("Apply logistic fitting.")
    model = Model()
    model.fit(data_tr["prediction"].to_numpy(), data_tr["label"].to_numpy())
    fit_pred = model.predict(data_eval["prediction"].to_numpy())
    closed_set_scores = compute_scores(fit_pred, data_eval["label"].to_numpy())

    logger.info("Run evaluation on the open set.")
    data_tr = read_data(
        pred_csv=os.path.join(cfg.train_indep_path.exp_folder, "sii.csv"),
        label_json=cfg.train_indep_path.scenes_file,
    )
    data_eval = read_data(
        pred_csv=os.path.join(cfg.test_indep_path.exp_folder, "sii.csv"),
        label_json="../test_listener_responses/CPC1.test_indep.json",
    )
    logger.info("Apply logistic fitting.")
    model = Model()
    model.fit(data_tr["prediction"].to_numpy(), data_tr["label"].to_numpy())
    fit_pred = model.predict(data_eval["prediction"].to_numpy())
    open_set_scores = compute_scores(fit_pred, data_eval["label"].to_numpy())

    with open("results.json", "w") as f:
        json.dump(
            {
                "closed_set scores:": closed_set_scores,
                "open_set scores:": open_set_scores,
            },
            f,
        )


if __name__ == "__main__":
    run()
