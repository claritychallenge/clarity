"""
inference.py — CLI wrapper around IntelligibilityPredictor
=============================================================

IntelligibilityPredictor now lives in lyric_intelligibility_model.py
alongside the model itself, so you can use it directly as a library:

    from lyric_intelligibility_model import IntelligibilityPredictor
    predictor = IntelligibilityPredictor.from_pretrained(
        "cadenzachallenge/CLIP2-BaselineMono"
    )
    result = predictor.predict("path/to/song.wav")   # or a directory

This script is just a command-line entry point around that class, writing
results to a CSV file instead of printing them.

Usage
-----
    python inference.py --repo_id cadenzachallenge/CLIP2-BaselineMono \
        --audio path/to/song.wav
    # Writes results.csv with one row: song,<score>

    # Score every audio file in a directory (top level only, not recursive):
    python inference.py --repo_id cadenzachallenge/CLIP2-BaselineMono \
        --audio path/to/songs_dir
    # Writes results.csv with one row per file: <stem>,<score>

    # Custom output path:
    python inference.py --repo_id cadenzachallenge/CLIP2-BaselineMono \
        --audio path/to/songs_dir --output scores.csv

    # Stereo audio downmixed to mono instead of better-ear left/right max:
    python inference.py --repo_id cadenzachallenge/CLIP2-BaselineMono \
        --audio path/to/song.wav --no-better-ear

    # Scores scaled to [0, 100] instead of [0, 1]:
    python inference.py --repo_id cadenzachallenge/CLIP2-BaselineMono \
        --audio path/to/song.wav --max-100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from lyric_intelligibility_model import IntelligibilityPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face Hub repo id, e.g. cadenzachallenge/CLIP2-BaselineMono",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to an audio file, OR a directory — every audio file at "
        "the top level of the directory will be scored (not recursive).",
    )
    parser.add_argument("--revision", default=None, help="Optional Hub revision/branch")
    parser.add_argument(
        "--device", default=None, help="cuda / cpu (default: auto-detect)"
    )
    parser.add_argument(
        "--better-ear",
        dest="better_ear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stereo handling: --better-ear (default) scores left/right "
        "separately and keeps the max; --no-better-ear downmixes to mono "
        "first and scores once. No effect on mono audio.",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="CSV file to write results to (default: results.csv)",
    )
    parser.add_argument(
        "--max-100",
        dest="max_100",
        action="store_true",
        help="Save scores scaled to [0, 100] instead of [0, 1] (default: [0, 1])",
    )
    args = parser.parse_args()

    predictor = IntelligibilityPredictor.from_pretrained(
        args.repo_id, revision=args.revision, device=args.device
    )
    result = predictor.predict(args.audio, better_ear=args.better_ear)

    def _scaled(score):
        if score == "ERROR":
            return score
        return score * 100 if args.max_100 else score

    if Path(args.audio).is_dir():
        rows = [
            (Path(filename).stem, _scaled(r.get("score", "ERROR")))
            for filename, r in result.items()
        ]
    else:
        rows = [(Path(args.audio).stem, _scaled(result.get("score", "ERROR")))]

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
