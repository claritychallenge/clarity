"""
clip_dataset.py — Lyric Intelligibility Dataset & DataLoader
for the CLIP1 and CLIP2 datasets
=========================================================

Directory layout expected
--------------------------
<root_path>/
    metadata/
        train_metadata.json   ← used for split="train" and split="dev"
        valid_metadata.json   ← used for split="valid"
        eval_metadata.json    ← used for split="eval"

    train/
        signals/
            c221f2084c780e8f212f4697.flac   ← processed / degraded signal (audio1)
            63de05fbfcab2d7748b83cb3.flac
            ...
        unprocessed/
            c221f2084c780e8f212f4697_unproc.flac   ← clean reference signal (audio2)
            63de05fbfcab2d7748b83cb3_unproc.flac
            ...
    eval/
        signals/
            ...
        unprocessed/
            ...

    valid/
        signals/
            ...
        unprocessed/
            ...

Each metadata JSON is a list of records with at least these keys:
    {
        "signal":       "c221f2084c780e8f212f4697",   # filename stem (no extension)
        "prompt":       "hello world",                # lyric / reference text
        "hearing_loss": "Mild",                       # severity category string
        "n_words": 2,                                  # number of words in the prompt
        "correctness":  0.75                          # listener correctness in [0, 1]
                                                      # OPTIONAL — omit for inference
    }

Mono vs. stereo
---------------
The ``mono`` parameter controls how multi-channel files are handled:

    mono=True  (default)
        All files are converted to a single-channel waveform by averaging
        across channels.  Shape returned per sample: ``(T,)``.
        Batch shape: ``[B, T]``.

    mono=False
        Channel layout is preserved.  Shape returned per sample: ``(C, T)``
        where C is the number of channels in the file (typically 2).
        Batch shape: ``[B, C, T]``.

        Use this when you want the model to process left and right channels
        independently (e.g. binaural hearing-loss simulation).

Batch output (matches train.py / eval.py contract)
----------------------------------------------------
    audio1        np.ndarray [B, T]    or [B, C, T]  – processed (degraded) waveforms
    audio2        np.ndarray [B, T]    or [B, C, T]  – clean reference waveforms
    severity_str  list[str]  [B]                     – raw hearing_loss strings
    severity      np.ndarray [B]                     – hearing_loss encoded as int label
                                                       (also available as ordered list
                                                       via dataset.severity_labels)
    ground_truth  list[str]  [B]                     – prompt / lyric text
    score         np.ndarray [B]                     – correctness in [0, 1]
                                                       nan when ``correctness`` is absent
                                                       from the metadata (inference mode)

Splits
------
    split="train"  →  training portion    (default 80 % from train metadata)
    split="dev"    →  development portion (default 20 % from train metadata)
    split="valid"  →  all samples, no splitting
    split="eval"   →  all samples, no splitting

Split strategies (only relevant for train/dev)
----------------------------------------------
    strategy="by_prompt"   (default / recommended)
        Groups all items that share the same prompt together, then assigns
        entire prompt groups to train or dev.  This prevents the model from
        seeing the same lyric at training time and being evaluated on it with
        a different severity level — i.e. no information leakage across splits.

    strategy="random"
        Shuffles individual samples and splits by percentage.
        Simpler, but risks prompt-level leakage between train and dev.

    strategy="full"
        Returns 100 % of the samples as the "train" split (dev will be empty).
        Useful for final training and when valid_metatda.json includes the correctness
        score for all samples.

Usage
-----
    from clip_dataset import LyricIntelligibilityDataset, build_dataloader

    # --- Mono (default) ---

    shared_params = dict(
        root_path    = "/path/to/cadenza_data/clip",
        strategy     = "by_prompt",
        train_ratio  = 0.8,
        seed         = 42,         # Same seed for both train and dev splits.
        mono         = True,       # waveforms are [T]
        sample_rate  = 16_000,
    )

    train_ds = LyricIntelligibilityDataset(
        shared_params*,
        split="train",
    )

    dev_ds = LyricIntelligibilityDataset(
        shared_params*,
        split="dev",
    )

    # --- Stereo ---
    train_ds = LyricIntelligibilityDataset(
        root_path   = "cadenza_data",
        split       = "train",
        mono        = False,      # waveforms are [C, T]
    )

    train_loader  = build_dataloader(train_ds, batch_size=16, shuffle=True)
    dev_loader    = build_dataloader(dev_ds,   batch_size=16, shuffle=False)

    # Valid split — all samples, no splitting applied
    valid_ds      = LyricIntelligibilityDataset("cadenza_data", split="valid")
    valid_loader  = build_dataloader(valid_ds, batch_size=16, shuffle=False)

    # Eval split — all samples, no splitting applied
    eval_ds       = LyricIntelligibilityDataset("cadenza_data", split="eval")
    eval_loader   = build_dataloader(eval_ds, batch_size=16, shuffle=False)
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import soxr
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity label registry
# ---------------------------------------------------------------------------

# Canonical ordered list of hearing-loss severity levels.
# The position of each label defines its integer encoding used during training.
# Add new categories here (in the desired order) if the dataset is extended.
SEVERITY_LEVELS: list[str] = [
    "No Loss",
    "Mild",
    "Moderate",
]

# Reverse lookup: severity string → integer index for fast encoding.
SEVERITY_TO_INT: dict[str, int] = {label: i for i, label in enumerate(SEVERITY_LEVELS)}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A single metadata record as loaded from JSON.
MetadataRecord = dict[str, str | float]

# Return type of __getitem__.
# audio shape is (T,) for mono=True and (C, T) for mono=False.
# score is float in [0, 1] for labelled data, or float('nan') for inference metadata.
# Full tuple: (audio1, audio2, severity_str, severity_int, prompt, score)
DatasetItem = tuple[np.ndarray, np.ndarray, str, int, str, float]

# Return type of collate_fn: batched versions of the above.
# audio shape is [B, T] for mono=True and [B, C, T] for mono=False.
Batch = tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str], np.ndarray]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LyricIntelligibilityDataset(Dataset):
    """
    PyTorch Dataset for the CLIP lyric-intelligibility corpora (CLIP1 / CLIP2).

    Each sample pairs a processed (degraded) audio signal with its clean
    reference, along with a hearing-loss severity label, the target lyric
    (ground-truth prompt), and a listener correctness score.

    Parameters
    ----------
    root_path : str
        Root directory of the dataset.  Must contain a ``metadata/`` sub-folder
        with ``train_metadata.json`` and/or ``eval_metadata.json``, plus
        ``train/`` and/or ``eval/`` audio sub-folders (e.g. cadenza_data/clip1
        or cadenza_data/clip2)
    split : {"train", "dev", "valid", "eval"}
        Which portion of the data to expose.
        ``"train"`` and ``"dev"`` are carved out of ``train_metadata.json``
        according to *strategy*.  ``"valid"`` and ``"eval"`` return all records
        from the respective metadata file without any splitting.
    strategy : {"by_prompt", "random", "full"}
        Split strategy — only used when *split* is ``"train"`` or ``"dev"``.
        See module docstring for a full description of each strategy.
    train_ratio : float
        Fraction of the training metadata assigned to the ``"train"`` split.
        The remaining ``1 - train_ratio`` becomes ``"dev"``.
        Ignored for ``"valid"`` / ``"eval"`` splits.  Default: ``0.8``.
    seed : int
        Random seed for reproducible splits.  Using the same seed for both
        the ``"train"`` and ``"dev"`` instantiations guarantees they are
        perfectly complementary (no overlap).  Default: ``42``.
    sample_rate : int
        Target sample rate in Hz.  Audio files are resampled on-the-fly with
        ``soxr`` (quality="HQ") if their native rate differs.  Default: ``16000``.
    mono : bool
        Channel handling mode.

        ``True`` (default) — average all channels to produce a mono waveform.
        Each sample's audio arrays have shape ``(T,)``; batches are ``[B, T]``.

        ``False`` — preserve the original channel layout.  Each sample's audio
        arrays have shape ``(C, T)`` where C is the file's channel count
        (typically 2 for stereo); batches are ``[B, C, T]``.

    Attributes
    ----------
    records : list[MetadataRecord]
        The subset of metadata records belonging to this split.
    severity_labels : list[str]
        Ordered list of severity label strings; index == integer encoding.
    prompts : list[str]
        All prompt strings in this split (may contain duplicates across
        different severity conditions).
    scores : np.ndarray
        All correctness scores in this split as a float32 numpy array.

    Examples
    --------
    >>> # Mono (default)
    >>> ds = LyricIntelligibilityDataset("cadenza_data/clip2", split="train", mono=True)
    >>> audio1, audio2, sev_str, sev_int, prompt, score = ds[0]
    >>> audio1.shape
    (T,)

    >>> # Stereo
    >>> ds = LyricIntelligibilityDataset("cadenza_data/clip2", split="train", mono=False)
    >>> audio1, *_ = ds[0]
    >>> audio1.shape
    (2, T)
    """

    def __init__(
        self,
        root_path: str,
        split: Literal["train", "dev", "valid", "eval"] = "train",
        strategy: Literal["by_prompt", "random", "full"] = "by_prompt",
        train_ratio: float = 0.8,
        seed: int = 42,
        sample_rate: int = 16_000,
        mono: bool = True,
    ) -> None:
        self.root_path   = Path(root_path)
        self.split       = split
        self.strategy    = strategy
        self.train_ratio = train_ratio
        self.seed        = seed
        self.sample_rate = sample_rate
        self.mono        = mono


        # "train" and "dev" both read from train_metadata; "valid"/"test" from test_metadata.
        self.set_name: str = "train" if split in ("train", "dev") else split

        # Load the full metadata list from JSON.
        metadata_path = self.root_path / "metadata" / f"{self.set_name}_metadata.json"
        with open(metadata_path, "r") as f:
            all_records: list[MetadataRecord] = json.load(f)

        # Drop any records whose audio files are absent on disk.
        all_records = self._validate_records(all_records)

        # Carve out the requested split (or keep everything for valid/eval).
        if split in ("valid", "eval"):
            self.records: list[MetadataRecord] = all_records
        else:
            self.records = self._split_records(all_records)

        # Emit a warning if any severity label is outside the known registry.
        self._check_severity_labels(self.records)

        log.info(
            f"[{split.upper()}] strategy={strategy}  mono={mono}  "
            f"samples={len(self.records)}  "
            f"unique_prompts={len({r['prompt'] for r in self.records})}"
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_records(self, records: list[MetadataRecord]) -> list[MetadataRecord]:
        """
        Filter out records whose audio files do not exist on disk.

        Checks both the processed signal (``signals/<id>.flac``) and the
        clean reference (``unprocessed/<id>_unproc.flac``).  Missing files
        are logged as warnings.

        Parameters
        ----------
        records : list[MetadataRecord]
            Full list of records loaded from the metadata JSON.

        Returns
        -------
        list[MetadataRecord]
            Subset of *records* for which both audio files exist.
        """
        valid: list[MetadataRecord] = []
        for r in records:
            sig  = self.root_path / self.set_name / "signals"     / f"{r['signal']}.flac"
            unpr = self.root_path / self.set_name / "unprocessed" / f"{r['signal']}_unproc.flac"

            if not sig.exists():
                log.warning(f"Missing signals file, skipping:     {sig}")
                continue
            if not unpr.exists():
                log.warning(f"Missing unprocessed file, skipping: {unpr}")
                continue

            valid.append(r)

        n_dropped = len(records) - len(valid)
        if n_dropped:
            log.warning(f"Dropped {n_dropped} records due to missing audio files.")
        return valid

    def _check_severity_labels(self, records: list[MetadataRecord]) -> None:
        """
        Warn if any record carries a severity label not listed in
        ``SEVERITY_LEVELS``.

        Unknown labels are not fatal — they are mapped to a catch-all index
        (``len(SEVERITY_LEVELS)``) at item access time — but they should be
        added to ``SEVERITY_LEVELS`` for proper training.

        Parameters
        ----------
        records : list[MetadataRecord]
            Records to inspect (typically ``self.records`` after splitting).
        """
        unknown: set[str] = {
            r["hearing_loss"]
            for r in records
            if r["hearing_loss"] not in SEVERITY_TO_INT
        }
        if unknown:
            log.warning(
                f"Unknown severity labels found: {unknown}. "
                f"They will be mapped to index {len(SEVERITY_LEVELS)} (catch-all). "
                f"Add them to SEVERITY_LEVELS if needed."
            )

    # ------------------------------------------------------------------
    # Split logic
    # ------------------------------------------------------------------

    def _split_records(self, records: list[MetadataRecord]) -> list[MetadataRecord]:
        """
        Dispatch to the appropriate splitting strategy and return only the
        subset of *records* that belongs to ``self.split``.

        Parameters
        ----------
        records : list[MetadataRecord]
            Full validated record list from the metadata JSON.

        Returns
        -------
        list[MetadataRecord]
            Records for the ``"train"`` or ``"dev"`` portion.

        Raises
        ------
        ValueError
            If ``self.strategy`` is not one of the recognised values.
        """
        rng = random.seed(self.seed)  # seeded RNG, isolated from global state

        if self.strategy == "full":
            # Return all records for "train"; return nothing for "dev".
            return records if self.split == "train" else []

        if self.strategy == "random":
            return self._random_split(records, rng)

        if self.strategy == "by_prompt":
            return self._prompt_split(records, rng)

        raise ValueError(
            f"Unknown strategy '{self.strategy}'. "
            "Choose from: 'by_prompt', 'random', 'full'."
        )

    def _random_split(
        self,
        records: list[MetadataRecord],
        rng: random.Random,
    ) -> list[MetadataRecord]:
        """
        Split by shuffling individual samples and cutting at ``train_ratio``.

        This is the simplest strategy but does **not** guarantee prompt-level
        separation — the same lyric may appear in both train and dev under
        different severity conditions, which constitutes information leakage.

        Parameters
        ----------
        records : list[MetadataRecord]
            Full validated record list.
        rng : random.Random
            Seeded random instance for reproducibility.

        Returns
        -------
        list[MetadataRecord]
            Train or dev portion depending on ``self.split``.
        """
        shuffled = records.copy()
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * self.train_ratio)
        return shuffled[:cut] if self.split == "train" else shuffled[cut:]

    def _prompt_split(
        self,
        records: list[MetadataRecord],
        rng: random.Random,
    ) -> list[MetadataRecord]:
        """
        Split by assigning entire prompt groups to train or dev.

        All records that share the same prompt string are kept together,
        so no prompt appears in both splits.  This prevents the model from
        being evaluated on a lyric it has already heard during training (even
        with a different severity level).

        Algorithm
        ---------
        1. Collect unique prompts; sort for determinism; shuffle with *rng*.
        2. Walk the shuffled prompt list, greedily accumulating prompts into
           the train set until the running sample count reaches
           ``floor(total * train_ratio)``.  All remaining prompts go to dev.
        3. Flatten each chosen group back into a record list.

        Using a greedy accumulation (rather than a hard percentage cut on the
        sorted key list) minimises the train/dev size imbalance that arises
        when prompt groups have variable sizes.

        Parameters
        ----------
        records : list[MetadataRecord]
            Full validated record list.
        rng : random.Random
            Seeded random instance for reproducibility.

        Returns
        -------
        list[MetadataRecord]
            Train or dev portion depending on ``self.split``.
        """
        # Build a mapping from prompt string → list of records with that prompt.
        groups: dict[str, list[MetadataRecord]] = defaultdict(list)
        for r in records:
            groups[str(r["prompt"])].append(r)

        # Sort keys before shuffling so the starting order is deterministic
        # regardless of JSON parse order or dict insertion order (Python 3.7+).
        prompt_keys: list[str] = sorted(groups.keys())
        rng.shuffle(prompt_keys)

        total  = len(records)
        target = int(total * self.train_ratio)  # desired train sample count

        train_keys: list[str] = []
        dev_keys:   list[str] = []
        train_count = 0

        for key in prompt_keys:
            if train_count < target:
                train_keys.append(key)
                train_count += len(groups[key])
            else:
                dev_keys.append(key)

        log.debug(
            f"  by_prompt split: "
            f"train={len(train_keys)} prompts / {train_count} samples | "
            f"dev={len(dev_keys)} prompts / {total - train_count} samples"
        )

        # Flatten the chosen prompt groups back into a flat record list.
        chosen_keys = train_keys if self.split == "train" else dev_keys
        return [r for key in chosen_keys for r in groups[key]]

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, signal_id: str, subdir: str) -> np.ndarray:
        """
        Load a single FLAC file, apply channel handling, and resample if necessary.

        Channel handling is controlled by ``self.mono``:

        - ``mono=True``  → average all channels to a single waveform ``(T,)``.
        - ``mono=False`` → keep the original channel layout ``(C, T)``.

        File naming convention
        ----------------------
        - Processed signals:   ``<set_name>/signals/<signal_id>.flac``
        - Clean references:    ``<set_name>/unprocessed/<signal_id>_unproc.flac``

        The ``_unproc`` suffix is appended automatically when *subdir* is
        ``"unprocessed"``.

        Parameters
        ----------
        signal_id : str
            Filename stem (without extension) from the metadata record.
        subdir : str
            Sub-directory within the set folder; either ``"signals"`` or
            ``"unprocessed"``.

        Returns
        -------
        np.ndarray, dtype float32
            - Shape ``(T,)``    when ``self.mono=True``.
            - Shape ``(C, T)``  when ``self.mono=False`` (C = number of channels).

        Raises
        ------
        Exception
            Re-raises any ``soundfile`` read error after logging it.
        """
        # Append the "_unproc" suffix only for the clean reference sub-folder.
        suffix = "_unproc" if subdir == "unprocessed" else ""
        path = self.root_path / self.set_name / subdir / f"{signal_id}{suffix}.flac"

        try:
            # always_2d=True guarantees shape [T, C] even for mono files,
            # which makes the channel-handling logic below uniform.
            waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
            # waveform shape after sf.read: [T, C]
        except Exception as e:
            log.error(f"Failed to load audio file: {path}  —  {e}")
            raise

        if self.mono:
            # Average across the channel axis to produce a mono signal.
            # For a mono file (C=1) this is a no-op beyond removing the channel dim.
            if waveform.shape[1] > 1:
                waveform = waveform.mean(axis=1)   # [T, C] → [T]
            else:
                waveform = waveform[:, 0]          # [T, 1] → [T]

            # Resample only when the file's native rate differs from the target.
            if sr != self.sample_rate:
                waveform = soxr.resample(waveform, sr, self.sample_rate, quality="HQ")

            return waveform  # shape: (T,), dtype: float32

        
        # Preserve channel layout.  Transpose [T, C] → [C, T] so the
        # channel axis is leading, matching the PyTorch convention.
        waveform = waveform.T  # [C, T]

        # Resample each channel independently.  soxr handles 2-D arrays
        # with shape [T, C] (channels-last), so we transpose back temporarily.
        if sr != self.sample_rate:
            waveform = soxr.resample(
                waveform.T, sr, self.sample_rate, quality="HQ"
            ).T  # [T, C] → resample → [T, C] → transpose → [C, T]

        return waveform  # shape: (C, T), dtype: float32

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.records)

    def __getitem__(self, idx: int) -> DatasetItem:
        """
        Return a single sample by index.

        Parameters
        ----------
        idx : int
            Index into ``self.records``.

        Returns
        -------
        audio1 : np.ndarray
            Processed (degraded) waveform from ``signals/``.
            Shape ``(T,)`` when ``mono=True``; ``(C, T)`` when ``mono=False``.
        audio2 : np.ndarray
            Clean reference waveform from ``unprocessed/``.
            Same shape convention as *audio1*.
        severity_string : str
            Raw ``hearing_loss`` label from metadata (e.g. ``"Mild"``).
        severity : int
            Integer encoding of the hearing-loss label.
            Unknown labels are mapped to ``len(SEVERITY_LEVELS)`` (catch-all).
        ground_truth : str
            Normalised lyric / prompt string.
        score : float
            Listener correctness in ``[0, 1]``, or ``float("nan")`` when the
            ``"correctness"`` key is absent from the metadata record (inference
            mode).  Always check ``math.isnan(score)`` before using this value
            for loss computation or metric calculation.
        """
        r = self.records[idx]

        audio1 = self._load_audio(str(r["signal"]), "signals")      # processed signal
        audio2 = self._load_audio(str(r["signal"]), "unprocessed")  # clean reference

        severity_string = str(r["hearing_loss"])
        # Fall back to a catch-all index for any label outside the registry.
        severity: int = SEVERITY_TO_INT.get(severity_string, len(SEVERITY_LEVELS))

        ground_truth = str(r["prompt"])

        # correctness is optional — absent in inference metadata.
        # Return nan as a safe sentinel so callers can detect the missing value
        # without breaking the fixed tuple contract.
        score: float = float(r["correctness"]) if "correctness" in r else float("nan")

        return audio1, audio2, severity_string, severity, ground_truth, score

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def severity_labels(self) -> list[str]:
        """
        Ordered list of severity label strings.

        The position of each string corresponds to its integer encoding,
        i.e. ``severity_labels[i]`` is the human-readable name for integer
        label ``i``.
        """
        return SEVERITY_LEVELS

    @property
    def prompts(self) -> list[str]:
        """
        All prompt strings in this split, in record order.

        May contain duplicates when the same lyric appears under multiple
        severity conditions.  Use ``set(dataset.prompts)`` for unique prompts.
        """
        return [str(r["prompt"]) for r in self.records]

    @property
    def scores(self) -> np.ndarray:
        """
        All listener correctness scores in this split.

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            Values in ``[0, 1]`` for labelled records.  Entries are
            ``nan`` for any record whose metadata lacks a ``"correctness"``
            field (inference mode).
        """
        return np.array(
            [r.get("correctness", float("nan")) for r in self.records],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Collate — pad variable-length waveforms within a batch
# ---------------------------------------------------------------------------

def collate_fn(batch: list[DatasetItem]) -> Batch:
    """
    Collate a list of dataset items into a padded batch.

    Automatically handles both mono and stereo waveforms based on the
    number of dimensions in the first sample's audio array:

    - Mono   ``(T,)``    → pads to ``[B, T_max]``
    - Stereo ``(C, T)``  → pads to ``[B, C, T_max]``

    Non-audio fields (severity strings, ground-truth prompts) are returned
    as plain Python lists; numeric fields (severity ints, scores) are
    returned as numpy arrays.

    Parameters
    ----------
    batch : list[DatasetItem]
        List of ``(audio1, audio2, severity_str, severity, ground_truth, score)``
        tuples as returned by ``LyricIntelligibilityDataset.__getitem__``.

    Returns
    -------
    audio1 : np.ndarray, shape (B, T_max) or (B, C, T_max), dtype float32
        Zero-padded processed waveforms.
    audio2 : np.ndarray, shape (B, T_max) or (B, C, T_max), dtype float32
        Zero-padded clean reference waveforms.
    severity_strings : list[str], length B
        Raw hearing-loss label strings.
    severities : np.ndarray, shape (B,), dtype int64
        Integer-encoded severity labels.
    ground_truths : list[str], length B
        Prompt / lyric strings.
    scores : np.ndarray, shape (B,), dtype float32
        Listener correctness values.
    """
    audio1_list, audio2_list, severity_strings, severities, ground_truths, scores = zip(*batch)

    def pad_waveforms(waveforms: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        Zero-pad a collection of waveforms to a common time length.

        Handles both mono ``(T,)`` and stereo ``(C, T)`` inputs.  Padding is
        always applied along the **time axis** (the last axis), so channel
        count is preserved unchanged.

        Parameters
        ----------
        waveforms : tuple of np.ndarray
            - Mono:   each array has shape ``(T_i,)``
            - Stereo: each array has shape ``(C, T_i)``

        Returns
        -------
        np.ndarray, dtype float32
            - Mono:   shape ``(N, T_max)``
            - Stereo: shape ``(N, C, T_max)``
            Shorter waveforms are right-padded with zeros along the time axis.
        """
        # Time is always the last axis for both mono (T,) and stereo (C, T).
        max_len = max(w.shape[-1] for w in waveforms)
        first   = waveforms[0]

        if first.ndim == 1:
            # Mono: allocate [N, T_max]
            padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
            for i, w in enumerate(waveforms):
                padded[i, :w.shape[0]] = w
        else:
            # Stereo (or any multi-channel): allocate [N, C, T_max]
            C = first.shape[0]
            padded = np.zeros((len(waveforms), C, max_len), dtype=np.float32)
            for i, w in enumerate(waveforms):
                padded[i, :, :w.shape[-1]] = w

        return padded

    return (
        pad_waveforms(audio1_list),              # [B, T_max] or [B, C, T_max]
        pad_waveforms(audio2_list),              # [B, T_max] or [B, C, T_max]
        list(severity_strings),                  # [B]  raw severity strings
        np.array(severities, dtype=np.int64),    # [B]  integer labels
        list(ground_truths),                     # [B]  prompt strings
        np.array(scores, dtype=np.float32),      # [B]  correctness scores
    )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: LyricIntelligibilityDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Wrap a ``LyricIntelligibilityDataset`` in a ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    dataset : LyricIntelligibilityDataset
        Dataset instance to wrap.
    batch_size : int
        Number of samples per batch.  Default: ``8``.
    shuffle : bool
        Whether to reshuffle the data at every epoch.
        Set ``True`` for training, ``False`` for evaluation.  Default: ``False``.
    num_workers : int
        Number of subprocesses for data loading.  Set to ``0`` to load data
        in the main process (useful for debugging).  Default: ``4``.
    pin_memory : bool
        If ``True`` and a CUDA device is available, the dataloader will copy
        tensors into pinned (page-locked) host memory before returning them,
        which can speed up host-to-device transfers.  Default: ``True``.

    Returns
    -------
    DataLoader
        Configured dataloader using the module-level ``collate_fn``.
    """
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        # Only pin memory when a GPU is actually available; avoids a warning otherwise.
        pin_memory  = pin_memory and torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Convenience: build train + dev loaders in one call
# ---------------------------------------------------------------------------

def build_train_dev_loaders(
    root_path: str,
    strategy: Literal["by_prompt", "random", "full"] = "by_prompt",
    train_ratio: float = 0.8,
    seed: int = 42,
    sample_rate: int = 16_000,
    mono: bool = True,
    train_batch_size: int = 8,
    dev_batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and dev ``DataLoader`` objects from a single metadata file.

    Both datasets are constructed with the same *seed*, which guarantees that
    their splits are perfectly complementary (i.e. every record belongs to
    exactly one of train or dev, with no overlap).

    Parameters
    ----------
    root_path : str
        Root directory of the dataset (see module docstring for layout).
    strategy : {"by_prompt", "random", "full"}
        Split strategy passed through to ``LyricIntelligibilityDataset``.
        Default: ``"by_prompt"`` (recommended; prevents prompt-level leakage).
    train_ratio : float
        Fraction of training metadata assigned to the train split.
        Default: ``0.8``.
    seed : int
        Random seed for reproducible splits.  Default: ``42``.
    sample_rate : int
        Target audio sample rate in Hz.  Default: ``16000``.
    mono : bool
        Channel mode passed through to ``LyricIntelligibilityDataset``.
        ``True`` (default) → mono ``[B, T]`` batches.
        ``False``          → stereo ``[B, C, T]`` batches.
    train_batch_size : int
        Batch size for the training dataloader.  Default: ``8``.
    dev_batch_size : int
        Batch size for the dev dataloader.  Default: ``16``.
    num_workers : int
        Number of worker subprocesses for both dataloaders.  Default: ``4``.

    Returns
    -------
    train_loader : DataLoader
        Shuffled dataloader over the training split.
    dev_loader : DataLoader
        Non-shuffled dataloader over the development split.
    """
    # Common kwargs shared by both dataset instantiations.
    shared: dict = dict(
        root_path    = root_path,
        strategy     = strategy,
        train_ratio  = train_ratio,
        seed         = seed,          # identical seed → complementary splits
        sample_rate  = sample_rate,
        mono         = mono,
    )

    # Takes all train samples and split train_ratio for training
    # and 1 - train_ratio for development
    train_ds = LyricIntelligibilityDataset(**shared, split="train")
    dev_ds   = LyricIntelligibilityDataset(**shared, split="dev")

    train_loader = build_dataloader(
        train_ds, batch_size=train_batch_size, shuffle=True,  num_workers=num_workers
    )
    dev_loader = build_dataloader(
        dev_ds,   batch_size=dev_batch_size,   shuffle=False, num_workers=num_workers
    )

    log.info(f"Train samples: {len(train_ds)}  Dev samples: {len(dev_ds)}")
    log.info(f"Train batches: {len(train_loader)}  Dev batches: {len(dev_loader)}")

    return train_loader, dev_loader
