import time
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

from clarity.evaluator.ha_metric import pyhaaqi, ear_model
from clarity.evaluator.haaqi import compute_haaqi
from clarity.evaluator.haspi import eb
from clarity.utils.audiogram import Audiogram
from clarity.utils.signal_processing import compute_rms

audiogram = np.array([45, 45, 35, 45, 60, 65, 70, 65])
audiogram_frequencies = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])


signal, sr = librosa.load(librosa.ex("choice"), duration=10)
signal2, sr2 = librosa.load(librosa.ex("brahms"), duration=10)


signal = signal.astype(np.float32) / 32767.0

# for side in ["left", "right"][:1]:
#     side_idx = 0 if side == "left" else 1
#
#     start = time.time()
#     np.random.seed(0)
#     score = compute_haaqi(
#         processed_signal=signal,
#         reference_signal=signal,
#         audiogram=Audiogram(levels=audiogram, frequencies=audiogram_frequencies),
#         sample_rate=sr,
#         level1=65 - 20 * np.log10(compute_rms(signal)),
#     )
#     print(f"Old implementation: {time.time() - start} - HAAQI: {score}")
#
#     start = time.time()
#     np.random.seed(0)
#     score_2 = pyhaaqi.compute_haaqi(
#         processed_signal=signal,
#         reference_signal=signal,
#         audiogram=audiogram,
#         audiogram_frequencies=audiogram_frequencies,
#         sample_rate=sr,
#         level1=65 - 20 * np.log10(compute_rms(signal)),
#     )
#     print(f"New implementation: {time.time() - start} - HAAQI: {score_2}")

equalisation = 1
ear_model_kwards = {"num_bands": 32}
ear = ear_model.EarModel(
    equalisation=equalisation,
    **ear_model_kwards,
)
ear_model_output = ear.compute(
    signal, sr, signal, sr, np.array([45, 45, 35, 45, 60, 65]), 65
)
print(ear_model_output[1])


eb_ear_model_output = eb.ear_model(
        signal, sr, signal, sr,
        np.array([45, 45, 35, 45, 60, 65]),
        equalisation,
        65,
    )
print(eb_ear_model_output[1])