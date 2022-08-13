# Regression test
# Pass some random data through code and compare with reference output
# scene_renderer, enhancer, compressor, haspi

import tempfile

import numpy as np
from omegaconf import OmegaConf
from scipy.io import wavfile

from clarity.data.scene_renderer_cec2 import SceneRenderer
from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haspi import haspi_v2_be


def test_full_CEC2_pipeline(regtest):
    np.random.seed(0)

    # Set up some scene to simulate
    # It's been designed to get good code coverage but running quickly
    # - Using three maskers - one from each noise type
    # - Using a short target with reduce pre and post silence
    # - Only generating 2 hearing aid channels
    scene = {
        "dataset": "demo",
        "room": "R06001",
        "scene": "S06001",
        "target": {"name": "T010_G0N_02468", "time_start": 37837, "time_end": 115894},
        "duration": 150000,
        "interferers": [
            {
                "position": 1,
                "time_start": 0,
                "time_end": 150000,
                "type": "noise",
                "name": "CIN_fan_014.wav",
                "offset": 5376,
            },
            {
                "position": 2,
                "time_start": 0,
                "time_end": 150000,
                "type": "speech",
                "name": "som_04766_05.wav",
                "offset": 40000,
            },
            {
                "position": 3,
                "time_start": 0,
                "time_end": 150000,
                "type": "music",
                "name": "1111967.low.mp3",
                "offset": 842553,
            },
        ],
        "SNR": 0.0,
        "listener": {
            "rotation": [
                {"sample": 116192.9795, "angle": 52.3628},
                {"sample": 124829.9795, "angle": 38.5256},
            ],
            "hrir_filename": ["VP_N6-ED", "VP_N6-BTE_fr"],
        },
    }

    demo_paths = OmegaConf.create(
        {
            "hoairs": "tests/test_data/rooms/HOA_IRs",
            "hrirs": "tests/test_data/hrir/HRIRs_MAT",
            "scenes": "tests/test_data/clarity_data/demo/scenes",
            "targets": "tests/test_data/targets",
            "interferers": "tests/test_data/interferers/{type}",
        }
    )

    demo_metadata = OmegaConf.create(
        {
            "room_definitions": "tests/test_data/metadata/rooms.demo.json",
            "scene_definitions": "",  # Scene definition file not needed for test
            "hrir_metadata": "tests/test_data/metadata/hrir_data.json",
        }
    )

    scene_renderer = SceneRenderer(
        demo_paths,
        demo_metadata,
        ambisonic_order=6,
        equalise_loudness=True,
        reference_channel=1,
        channel_norms=[12.0, 3.0],
    )

    output_path = "tmp"

    with tempfile.TemporaryDirectory() as output_path:

        target, interferers, anechoic, _head_turn = scene_renderer.generate_hoa_signals(
            scene
        )

        scene_renderer.generate_binaural_signals(
            scene, target, interferers, anechoic, output_path
        )

        _, reference = wavfile.read(f"{output_path}/S06001_target_anechoic_CH1.wav")
        _, signal = wavfile.read(f"{output_path}/S06001_mix_CH1.wav")

    reference = reference.astype(float) / 32768.0
    signal = signal.astype(float) / 32768.0

    # Truncate to just over 2 seconds - i.e. just use part of the signals to speed up the HASPI calculation a little
    signal = signal[:100000, :]
    reference = reference[:100000, :]

    # The data below doesn't really need to be meaningful.
    # The purpose of the test is not to see if the haspi score is reasonable
    # but just to check that the results do not change unexpectedly across releases.

    nalr_cfg = {"nfir": 220, "fs": 44100}
    compressor_cfg = {
        "threshold": 0.35,
        "attenuation": 0.1,
        "attack": 50,
        "release": 1000,
        "rms_buffer_size": 0.064,
    }

    audiogram_l = [45, 50, 60, 65, 60, 65, 70, 80]
    audiogram_r = [45, 45, 60, 70, 60, 60, 80, 80]
    audiogram_cfs = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]

    fs = 44100

    enhancer = NALR(**nalr_cfg)
    compressor = Compressor(**compressor_cfg)

    nalr_fir, _ = enhancer.build(audiogram_l, audiogram_cfs)
    out_l = enhancer.apply(nalr_fir, signal[:, 0])

    nalr_fir, _ = enhancer.build(audiogram_r, audiogram_cfs)
    out_r = enhancer.apply(nalr_fir, signal[:, 1])

    out_l, _, _ = compressor.process(out_l)
    out_r, _, _ = compressor.process(out_r)

    enhanced_audio = np.stack([out_l, out_r], axis=1)

    enhanced_audio = np.tanh(enhanced_audio)

    sii_enhanced = haspi_v2_be(
        xl=reference[:, 0],
        xr=reference[:, 1],
        yl=enhanced_audio[:, 0],
        yr=enhanced_audio[:, 1],
        fs_signal=fs,
        audiogram_l=audiogram_l,
        audiogram_r=audiogram_r,
        audiogram_cfs=audiogram_cfs,
    )

    print(f"Enhanced audio HASPI score is {sii_enhanced}")

    regtest.write(f"Enhanced audio HASPI score is {sii_enhanced:0.7f}\n")

    # Enhanced audio HASPI score is 0.2994066
