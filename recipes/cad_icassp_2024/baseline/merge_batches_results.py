"""Join batches scores into a single file."""

# pylint: disable=import-error
import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="", config_name="config", version_base=None)
def join_batches(config: DictConfig) -> None:
    """
    Join batches scores into a single file.

    Args:
        config (DictConfig): Dictionary of configuration options.
            The `.evaluate.batch_size` is extracted to determine how many
            batches there are to combine.

    """
    batches_results = []
    for batch in range(config.evaluate.batch_size):
        batches_results.append(
            pd.read_csv(f"scores_{batch + 1}-{config.evaluate.batch_size}.csv")
        )
    df_res = pd.concat(batches_results, ignore_index=True)
    df_res.to_csv("scores.csv", index=False)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    join_batches()
