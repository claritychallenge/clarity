"""Join batches scores into a single file."""

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="", config_name="config")
def join_batches(config: DictConfig) -> None:
    """
    Join batches scores into a single file.

    """
    batches_results = []
    for batch in range(config.evaluate.batch_size):
        batches_results.append(
            pd.read_csv(
                f"scores_{batch}-{config.evaluate.batch_size}.csv", index_col=False
            )
        )
    df_res = pd.concat(batches_results, ignore_index=True)
    df_res.to_csv("scores.csv", index=False)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    join_batches()
