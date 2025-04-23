"""Make intelligibility predictions from HASPI scores."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from recipes.cpc3.baseline.shared_predict_utils import (
    LogisticModel,
    load_dataset_with_haspi,
)

log = logging.getLogger(__name__)


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config", version_base=None)
def predict_dev(cfg: DictConfig):
    """Predict intelligibility from HASPI scores."""

    # Load the data
    log.info("Loading dataset...")
    records_train_df = load_dataset_with_haspi(cfg, "train")
    records_dev_df = load_dataset_with_haspi(cfg, "dev")

    # Compute the logistic fit
    log.info("Making the fitting model...")
    model = LogisticModel()
    model.fit(records_train_df.haspi_score, records_train_df.correctness)

    # Make predictions for all items in the dev data
    log.info("Starting predictions...")
    records_dev_df["predicted_correctness"] = model.predict(records_dev_df.haspi_score)

    # Save results to CSV file

    output_file = f"{cfg.dataset}.dev.predict.csv"
    records_dev_df[["signal", "predicted_correctness"]].to_csv(
        output_file,
        index=False,
        header=["signal_ID", "intelligibility_score"],
        mode="w",
    )
    log.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    predict_dev()
