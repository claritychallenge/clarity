"""Make intelligibility predictions from HASPI scores."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    LogisticModel,
    load_dataset_with_score,
)

log = logging.getLogger(__name__)


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def predict_dev(cfg: DictConfig):
    """Predict intelligibility for baselines.

    Set config.baseline to ```stoi``` or ```whisper_mixture``` or ```whisper_vocals```
    depending on which baseline you want to run.
    """

    # Load the metadata file for the dataset
    log.info("Loading dataset...")

    records_train_df = load_dataset_with_score(cfg, "train")
    records_valid_df = load_dataset_with_score(cfg, "valid")

    # Compute the logistic fit
    log.info("Making the fitting model...")
    model = LogisticModel()
    model.fit(records_train_df[f"{cfg.baseline.system}"], records_train_df.correctness)

    # Make predictions for all items in the dev data
    log.info("Starting predictions...")
    records_valid_df["predicted_correctness"] = model.predict(
        records_valid_df[f"{cfg.baseline.system}"]
    )

    # Save results to CSV file
    output_file = f"{cfg.data.dataset}.{cfg.baseline.system}.valid.predict.csv"
    records_valid_df[["signal", "predicted_correctness"]].to_csv(
        output_file,
        index=False,
        header=["signal_ID", "intelligibility_score"],
        mode="w",
    )
    log.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    predict_dev()
