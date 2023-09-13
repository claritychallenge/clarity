import time
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

from clarity.evaluator.ha_metric import pyhaaqi
from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.audiogram import Audiogram
from clarity.utils.signal_processing import compute_rms

audiogram = np.array([45, 45, 35, 45, 60, 65, 70, 65])
audiogram_frequencies = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])


signal, sr = librosa.load(librosa.ex("choice"), duration=15)
signal2, sr2 = librosa.load(librosa.ex("brahms"), duration=15)


signal = signal.astype(np.float32) / 32767.0

for side in ["left", "right"][:1]:
    side_idx = 0 if side == "left" else 1

    # start = time.time()
    # np.random.seed(0)
    # score = compute_haaqi(
    #     processed_signal=signal,
    #     reference_signal=signal,
    #     audiogram=Audiogram(levels=audiogram, frequencies=audiogram_frequencies),
    #     sample_rate=sr,
    #     level1=65 - 20 * np.log10(compute_rms(signal)),
    # )
    # print(f"Old implementation: {time.time() - start} - HAAQI: {score}")

    start = time.time()
    np.random.seed(0)
    score_2 = pyhaaqi.compute_haaqi(
        processed_signal=signal,
        reference_signal=signal2,
        audiogram=audiogram,
        audiogram_frequencies=audiogram_frequencies,
        sample_rate=sr,
        level1=65 - 20 * np.log10(compute_rms(signal)),
    )
    print(f"New implementation: {time.time() - start} - HAAQI: {score_2}")
