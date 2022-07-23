# Regression test
# Pass some random data through code and compare with reference output
# scene_renderer, enhancer, compressor, haspi

import os

import numpy as np
from omegaconf import OmegaConf
from scipy.io import wavfile
from scipy.signal import unit_impulse

from clarity.data.scene_renderer_cec2 import SceneRenderer
from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.mbstoi.mbstoi import mbstoi
from clarity.evaluator.mbstoi.mbstoi_utils import find_delay_impulse
from clarity.evaluator.msbg.audiogram import Audiogram
from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad


def listen(ear, signal, audiogram_l, audiogram_r):
    """
    Generate MSBG processed signal
    :param ear: MSBG ear
    :param wav: binaural signal
    :return: binaural signal
    """
    ear.set_audiogram(audiogram_l)
    out_l = ear.process(signal[:, 0])

    ear.set_audiogram(audiogram_r)
    out_r = ear.process(signal[:, 1])
    if len(out_l[0]) != len(out_r[0]):
        diff = len(out_l[0]) - len(out_r[0])
        if diff > 0:
            out_r[0] = np.flipud(pad(np.flipud(out_r[0]), len(out_l[0])))
        else:
            out_l[0] = np.flipud(pad(np.flipud(out_l[0]), len(out_r[0])))
    return np.concatenate([out_l, out_r]).T


def test_full_CEC1_pipeline(regtest):
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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    target, interferers, anechoic, head_turn = scene_renderer.generate_hoa_signals(
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

    fs = 44100

    msbg_ear_cfg = {"src_pos": "ff", "fs": fs, "equiv0dBSPL": 100, "ahr": 20}

    audiogram_l = np.array([45, 50, 60, 65, 60, 65, 70, 80])
    audiogram_r = np.array([45, 45, 60, 70, 60, 60, 80, 80])
    audiogram_cfs = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])

    ear = Ear(**msbg_ear_cfg)

    left_audiogram = Audiogram(cfs=audiogram_cfs, levels=audiogram_l)
    right_audiogram = Audiogram(cfs=audiogram_cfs, levels=audiogram_r)

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

    # Create discrete delta function (DDF) signal for time alignment
    ddf_signal = np.zeros((np.shape(signal)))
    ddf_signal[:, 0] = unit_impulse(len(signal), int(MSBG_FS / 2))
    ddf_signal[:, 1] = unit_impulse(len(signal), int(MSBG_FS / 2))

    reference_processed = listen(ear, reference, left_audiogram, right_audiogram)
    signal_processed = listen(ear, signal, left_audiogram, right_audiogram)

    # Calculate channel-specific unit impulse delay due to HL model and audiograms
    delay = find_delay_impulse(ddf_signal, initial_value=int(MSBG_FS / 2))
    max_delay = int(np.max(delay))

    # Allow for value lower than 1000 samples in case of unimpaired hearing
    assert max_delay <= 2000

    # Correct for delays by padding clean signals
    clean_pad = np.zeros((len(reference_processed) + max_delay, 2))
    proc_pad = np.zeros((len(signal_processed) + max_delay, 2))

    assert len(proc_pad) >= len(signal_processed)

    clean_pad[
        int(delay[0]) : int(len(reference_processed) + int(delay[0])), 0
    ] = reference_processed[:, 0]
    clean_pad[
        int(delay[1]) : int(len(reference_processed) + int(delay[1])), 1
    ] = reference_processed[:, 1]
    proc_pad[: len(signal_processed)] = signal_processed

    grid_coarseness = 1
    sii_enhanced = mbstoi(
        clean_pad[:, 0],
        clean_pad[:, 1],
        proc_pad[:, 0],
        proc_pad[:, 1],
        44100,
        grid_coarseness,
    )

    print(f"Enhanced audio MBSTOI score is {sii_enhanced}")

    regtest.write(f"Enhanced audio MBSTOI score is {sii_enhanced:0.7f}\n")

    # Enhanced audio MBSTOI score is 0.2994066
