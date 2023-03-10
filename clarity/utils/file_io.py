"""File I/O functions for jsonl files."""
import json


def read_jsonl(filename: str) -> list:
    """Read a jsonl file into a list of dictionaries."""
    with open(filename, "r", encoding="utf-8") as fp:
        records = [json.loads(line) for line in fp]
    return records


def write_jsonl(filename: str, records: list) -> None:
    """Write a list of dictionaries to a jsonl file."""
    with open(filename, "a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")
