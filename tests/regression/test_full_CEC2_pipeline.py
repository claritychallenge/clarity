"""Regression Tests for CEC2"""
# pylint: disable=too-many-locals invalid-name
from __future__ import annotations

from typing import Final

import numpy as np
from cpuinfo import get_cpu_info
from omegaconf import OmegaConf
from scipy.io import wavfile

from clarity.data.scene_renderer_cec2 import SceneRenderer
from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haspi import haspi_v2_be
from clarity.utils.audiogram import Audiogram, Listener

# Pass some random data through code and compare with reference output
# scene_renderer, enhancer, compressor, haspi


CPUINFO: Final = get_cpu_info()

np.random.seed(0)

# Set up some scene to simulate
# It's been designed to get good code coverage but running quickly
# - Using three maskers - one from each noise type
# - Using a short target with reduce pre and post silence
# - Only generating 2 hearing aid channels
SCENE: Final = {
    "dataset": "train",
    "room": "R06001",
    "scene": "S06001",
    "target": {"name": "T010_G0N_02468", "time_start": 0, "time_end": 115894},
    "duration": 8820,
    "interferers": [
        {
            "position": 1,
            "time_start": 0,
            "time_end": 8820,
            "type": "noise",
            "name": "CIN_fan_014.wav",
            "offset": 5376,
        },
        {
            "position": 2,
            "time_start": 0,
            "time_end": 8820,
            "type": "speech",
            "name": "som_04766_05.wav",
            "offset": 40000,
        },
        {
            "position": 3,
            "time_start": 0,
            "time_end": 8820,
            "type": "music",
            "name": "1111967.low.mp3",
            "offset": 842553,
        },
    ],
    "SNR": 0.0,
    "listener": {
        "rotation": [
            {"sample": 100, "angle": 52.3628},
            {"sample": 400, "angle": 38.5256},
        ],
        "hrir_filename": ["VP_N6-ED", "VP_N6-BTE_fr"],
    },
}

TEST_PATHS: Final = OmegaConf.create(
    {
        "hoairs": "tests/test_data/rooms/HOA_IRs",
        "hrirs": "tests/test_data/hrir/HRIRs_MAT",
        "scenes": "tests/test_data/clarity_data/train/scenes",
        "targets": "tests/test_data/targets",
        "interferers": "tests/test_data/interferers/{type}",
    }
)

TEST_METADATA: Final = OmegaConf.create(
    {
        "room_definitions": "tests/test_data/metadata/rooms.train.json",
        "scene_definitions": "",  # Scene definition file not needed for test
        "hrir_metadata": "tests/test_data/metadata/hrir_data.json",
    }
)

SCENE_RENDERER: Final = SceneRenderer(
    TEST_PATHS,
    TEST_METADATA,
    ambisonic_order=6,
    equalise_loudness=True,
    reference_channel=1,
    channel_norms=[12.0, 3.0],
)


def test_full_cec2_pipeline(
    regtest,
    tmp_path,
    scene: dict | None = None,
    scene_renderer: SceneRenderer = SCENE_RENDERER,
) -> None:
    """Test full CEC2 pipeline"""

    if scene is None:
        scene = SCENE
    target, interferers, anechoic, _head_turn = scene_renderer.generate_hoa_signals(
        scene
    )

    scene_renderer.generate_binaural_signals(
        scene, target, interferers, anechoic, str(tmp_path)
    )
    _, reference = wavfile.read(f"{tmp_path}/S06001_target_anechoic_CH1.wav")
    _, signal = wavfile.read(f"{tmp_path}/S06001_mix_CH1.wav")

    reference = reference.astype(float) / 32768.0
    signal = signal.astype(float) / 32768.0

    # Truncate to 200 ms - i.e. just use part of the signals
    # to speed up the HASPI calculation a little
    signal = signal[:8820, :]
    reference = reference[:8820, :]

    # The data below doesn't really need to be meaningful.
    # The purpose of the test is not to see if the haspi score is reasonable
    # but just to check that the results do not change unexpectedly across releases.

    nalr_cfg = {"nfir": 220, "sample_rate": 44100}
    compressor_cfg = {
        "threshold": 0.35,
        "attenuation": 0.1,
        "attack": 50,
        "release": 1000,
        "rms_buffer_size": 0.064,
    }

    audiogram_l = np.array([45, 50, 60, 65, 60, 65, 70, 80])
    audiogram_r = np.array([45, 45, 60, 70, 60, 60, 80, 80])
    audiogram_cfs = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])

    audiogram_left = Audiogram(levels=audiogram_l, frequencies=audiogram_cfs)
    audiogram_right = Audiogram(levels=audiogram_r, frequencies=audiogram_cfs)

    sample_rate = 44100

    enhancer = NALR(**nalr_cfg)
    compressor = Compressor(**compressor_cfg)  # type: ignore

    nalr_fir, _ = enhancer.build(audiogram_left)
    out_l = enhancer.apply(nalr_fir, signal[:, 0])

    nalr_fir, _ = enhancer.build(audiogram_right)
    out_r = enhancer.apply(nalr_fir, signal[:, 1])

    out_l, _, _ = compressor.process(out_l)
    out_r, _, _ = compressor.process(out_r)

    enhanced_audio = np.stack([out_l, out_r], axis=1)

    enhanced_audio = np.tanh(enhanced_audio)
    listener = Listener(audiogram_left=audiogram_left, audiogram_right=audiogram_right)

    sii_enhanced = haspi_v2_be(
        reference_left=reference[:, 0],
        reference_right=reference[:, 1],
        processed_left=enhanced_audio[:, 0],
        processed_right=enhanced_audio[:, 1],
        sample_rate=sample_rate,
        listener=listener,
    )

    regtest.write(f"Enhanced audio HASPI score is {sii_enhanced:0.7f}\n")

    # Enhanced audio HASPI score is 0.2994066
