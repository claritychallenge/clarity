"""test for results support module"""
# pylint: disable=import-error
from pathlib import Path

import pytest

from clarity.utils.results_support import ResultsFile

# Define some sample data for testing
sample_header = ["Name", "Score"]
sample_data = [{"Name": "Alice", "Score": 95}, {"Name": "Bob", "Score": 88}]


@pytest.fixture(name="results_file")
def fixture_results_file(tmpdir):
    # Create a temporary directory and a temporary CSV file for testing
    file_name = Path(tmpdir, "test_results.csv")
    return ResultsFile(file_name, sample_header)


def test_create_file_str(tmpdir):
    # Create a temporary directory and a temporary CSV file for testing
    file_name = f"{tmpdir}/test_results.csv"
    result_file = ResultsFile(file_name, sample_header)
    assert result_file.file_name.as_posix() == file_name


def test_add_result(results_file):
    # Test adding a result to the CSV file
    results_file.add_result({"Name": "Charlie", "Score": 75})

    # Read the CSV file and check if the added data is present
    with open(results_file.file_name, encoding="utf-8") as csv_file:
        lines = csv_file.readlines()
        assert len(lines) == 2  # There should be 2 lines (header + 1 data row)
        assert "Charlie,75\n" in lines  # Check if the added data is present


def test_header_written(results_file):
    # Test if the header row is written when the ResultsFile is created
    with open(results_file.file_name, encoding="utf-8") as csv_file:
        lines = csv_file.readlines()
        assert len(lines) == 1  # There should be 1 line (only header)
        assert lines[0].strip() == "Name,Score"  # Check the header content


def test_missing_column(results_file):
    # Test adding a result with a missing column
    with pytest.raises(KeyError):
        results_file.add_result({"Name": "Eve"})


def test_nonexistent_file(tmp_path):
    # Test creating a ResultsFile with a non-existent file
    file_name = Path(tmp_path) / "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        ResultsFile(file_name, sample_header, append_results=True)
