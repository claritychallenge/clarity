"""Make intelligibility predictions from HASPI scores."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from shared_predict_utils import (
    LogisticModel,
    load_dataset_with_stoi,
)

log = logging.getLogger(__name__)


# pylint: disable = no-value-for-parameter
@hydra.main(config_path=".", config_name="config", version_base=None)
def predict_dev(cfg: DictConfig):
    """Predict intelligibility from HASPI scores."""

    # Load the data
    log.info("Loading dataset...")
    records_train_df = load_dataset_with_stoi(cfg, "train")
    records_valid_df = load_dataset_with_stoi(cfg, "valid")

    # Compute the logistic fit
    log.info("Making the fitting model...")
    model = LogisticModel()
    model.fit(records_train_df[f"stoi_score"], records_train_df.correctness)

    # Make predictions for all items in the dev data
    log.info("Starting predictions...")
    records_valid_df["predicted_correctness"] = model.predict(
        records_valid_df[f"stoi_score"]
    )

    # Save results to CSV file

    output_file = f"{cfg.data.dataset}.stoi.valid.predict.csv"
    records_valid_df[["signal", "predicted_correctness"]].to_csv(
        output_file,
        index=False,
        header=["signal_ID", "intelligibility_score"],
        mode="w",
    )
    log.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    predict_dev()
