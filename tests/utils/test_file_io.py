"""Test the file_io module."""
import pytest

from clarity.utils.file_io import read_jsonl, write_jsonl


def test_read_jsonl():
    """Test the read_jsonl function."""
    expected = [
        {"id": 1, "name": "xxx"},
        {"id": 2, "name": "yyy"},
        {"id": 3, "name": "zzz"},
    ]
    data = read_jsonl("tests/test_data/filetypes/valid.jsonl")
    assert data == expected


@pytest.mark.parametrize(
    "records",
    [
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
    ],
)
def test_jsonl_write_read_loop(records, tmp_path):
    """Test the write_jsonl and read_jsonl functions."""
    write_jsonl(tmp_path / "test1.jsonl", records)
    assert read_jsonl(tmp_path / "test1.jsonl") == records


@pytest.mark.parametrize(
    "records",
    [
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
    ],
)
def test_jsonl_append_read_loop(records, tmp_path):
    """Test the write_jsonl correctly appends to existing files."""
    write_jsonl(tmp_path / "test2.jsonl", records)
    write_jsonl(tmp_path / "test2.jsonl", records)
    assert read_jsonl(tmp_path / "test2.jsonl") == records + records
