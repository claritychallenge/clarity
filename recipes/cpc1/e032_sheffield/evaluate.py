import json
import logging
from pathlib import Path

import hydra
import numpy as np
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
    The mapping parameters need to fit first using mbstoi, intelligibility score
    pairs, using fit().
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
        initial_guess = [50.0, 1.0]  # Initial guess for parameter values
        self.params, *_remaining_returns = curve_fit(
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


def read_data(pred_json: Path, label_json: Path):
    # read label_json to dict
    with label_json.open("r", encoding="utf-8") as fp:
        labels = json.load(fp)

    label_dict = {item["signal"]: item["correctness"] for item in labels}

    with pred_json.open("r", encoding="utf-8") as fp:
        pred_dict = json.load(fp)

    prediction = [pred * 100.0 for pred in pred_dict.values()]
    label = [label_dict[signal] for signal in pred_dict]

    return np.array(prediction), np.array(label)


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    if cfg.cpc1_track == "open":
        track = "_indep"
    elif cfg.cpc1_track == "closed":
        track = ""
    else:
        logger.error("cpc1_track has to be closed or open")

    # encoder representation evaluation
    prediction_dev, label_dev = read_data(
        Path(cfg.path.exp_folder) / "dev_enc_similarity.json",
        Path(cfg.path.cpc1_train_data) / f"metadata/CPC1.{'train'+track}.json",
    )
    prediction_test, label_test = read_data(
        Path(cfg.path.exp_folder) / "test_enc_similarity.json",
        Path(f"../test_listener_responses/CPC1.{'test'+track}.json"),
    )

    logger.info("Apply logistic fitting.")
    model = Model()
    model.fit(prediction_dev, label_dev)
    fit_pred = model.predict(prediction_test)
    enc_scores = compute_scores(fit_pred, label_test)

    # decoder representation evaluation
    prediction_dev, label_dev = read_data(
        Path(cfg.path.exp_folder) / "dev_dec_similarity.json",
        Path(cfg.path.cpc1_train_data) / f"metadata/CPC1.{'train'+track}.json",
    )
    prediction_test, label_test = read_data(
        Path(cfg.path.exp_folder) / "test_dec_similarity.json",
        Path(f"../test_listener_responses/CPC1.{'test'+track}.json"),
    )

    logger.info("Apply logistic fitting.")
    model = Model()
    model.fit(prediction_dev, label_dev)
    fit_pred = model.predict(prediction_test)
    dec_scores = compute_scores(fit_pred, label_test)

    results_file = Path(cfg.path.exp_folder) / "results.json"
    with results_file.open("w", encoding="utf-8") as fp:
        json.dump({"enc_results": enc_scores, "dec_results": dec_scores}, fp)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()
