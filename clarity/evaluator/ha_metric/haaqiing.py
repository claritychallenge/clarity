import time
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

from clarity.evaluator.haaqi import compute_haaqi
from clarity.evaluator.haspi import eb
from clarity.utils.audiogram import Audiogram
from clarity.utils.signal_processing import compute_rms

audiogram = np.array([45, 45, 35, 45, 60, 65, 70, 65])
audiogram_frequencies = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])


# signal, sr = librosa.load(librosa.ex("choice"), duration=10)
# signal2, sr2 = librosa.load(librosa.ex("brahms"), duration=10)

duration = 15
sr = 24000
signal = np.random.rand((int(sr * duration)))
noise = np.random.rand((int(sr * duration)))
equalisation = 1
# signal = signal.astype(np.float32) / 32767.0

for side in ["left", "right"][:1]:
    side_idx = 0 if side == "left" else 1

    np.random.seed(0)
    start = time.time()
    score = compute_haaqi(
        processed_signal=signal,
        reference_signal=signal + noise,
        audiogram=Audiogram(levels=audiogram, frequencies=audiogram_frequencies),
        processed_sample_rate=sr,
        reference_sample_rate=sr,
        level1=65 - 20 * np.log10(compute_rms(signal)),
    )
    print(f"Old implementation: {time.time() - start} - HAAQI: {score}")


# equalisation = 1
# start = time.time()
# eb_ear_model_output = eb.ear_model(
#         signal, sr, signal, sr,
#         np.array([45, 45, 35, 45, 60, 65]),
#         equalisation,
#         65,
#     )
# print(f"Time: {time.time() - start}")
# print(eb_ear_model_output[0].shape)