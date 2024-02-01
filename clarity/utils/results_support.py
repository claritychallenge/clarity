"""Dataclass to save challenges results to a CSV file."""
from __future__ import annotations

# pylint: disable=import-error
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResultsFile:
    """A utility class for writing results to a CSV file.

    Attributes:
        file_name (str | Path): The name of the file to write results to.
        header_columns (list[str]): The columns to write to the CSV file.
        append_results (bool): Whether to append results to an existing file.
            If False, a new file will be created and the header row will be written.
            Defaults to False.
    """

    file_name: str | Path
    header_columns: list[str]
    append_results: bool = False

    def __post_init__(self):
        """Write the header row to the CSV file."""
        if isinstance(self.file_name, str):
            self.file_name = Path(self.file_name)

        if self.append_results:
            if not Path(self.file_name).exists():
                raise FileNotFoundError(
                    "Cannot append results to non-existent file "
                    f"{self.file_name.as_posix()}"
                    " - please set append_results=False"
                )
        else:
            with open(self.file_name, "w", encoding="utf-8", newline="") as csv_file:
                csv_writer = csv.writer(
                    csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csv_writer.writerow(self.header_columns)

    def add_result(
        self,
        row_values: dict[str, str | float],
    ):
        """Add a result to the CSV file.

        Args:
            row_values (dict[str, str | float]): The values to write to the CSV file.
        """

        with open(self.file_name, "a", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            row = []
            for column in self.header_columns:
                row.append(row_values[column])

            csv_writer.writerow(row)
