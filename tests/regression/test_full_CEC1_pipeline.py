"""Tests the full CEC1 baseline pipeline"""
# pylint: disable=too-many-locals invalid-name

# Regression test
# Pass some random data through code and compare with reference output
# CEC1 scene_renderer, enhancer, compressor, MSBG and MBSTOI

import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import unit_impulse

from clarity.data.scene_renderer_cec1 import Renderer
from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.mbstoi.mbstoi import mbstoi
from clarity.evaluator.mbstoi.mbstoi_utils import find_delay_impulse
from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS, pad
from clarity.utils.audiogram import Audiogram


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


def test_full_cec1_pipeline(regtest):
    """Tests the full CEC1 baseline pipeline"""
    np.random.seed(0)

    scene = {
        "room": {"name": "R00001", "dimensions": "5.9x3.4186x2.9"},
        "hrirfilename": "VP_N6-BTE_fr",
        "target": {
            "Positions": [-0.5, 3.4, 1.2],
            "ViewVectors": [0.291, -0.957, 0.0],
            "name": "T010_G0N_02468",
            "nsamples": 109809,
        },
        "listener": {"Positions": [0.2, 1.1, 1.2], "ViewVectors": [-0.414, 0.91, 0.0]},
        "interferer": {
            "Positions": [0.4, 3.2, 1.2],
            "name": "CIN_fan_014",
            "nsamples": 1190700,
            "duration": 27.0,
            "type": "noise",
            "offset": 5376,
        },
        "azimuth_target_listener": -7.54,
        "azimuth_interferer_listener": -29.9,
        "scene": "S06001",
        "dataset": ".",
        "pre_samples": 88200,
        "post_samples": 0,
        "SNR": 0.586,
    }

    with tempfile.TemporaryDirectory() as output_path:
        renderer = Renderer(
            input_path="tests/test_data",
            output_path=output_path,
            num_channels=3,
        )

        renderer.render(
            pre_samples=scene["pre_samples"],
            post_samples=scene["post_samples"],
            dataset=scene["dataset"],
            target_id=scene["target"]["name"],
            noise_type=scene["interferer"]["type"],
            interferer_id=scene["interferer"]["name"],
            room=scene["room"]["name"],
            scene=scene["scene"],
            offset=scene["interferer"]["offset"],
            snr_dB=scene["SNR"],
        )

        _, reference = wavfile.read(f"{output_path}/S06001_target_anechoic.wav")
        _, signal = wavfile.read(f"{output_path}/S06001_mixed_CH1.wav")

    reference = reference.astype(float)  # / 32768.0
    signal = signal.astype(float)  # / 32768.0

    # Truncate to just over 1/2 second - i.e. just use part of the signals to
    # speed up the HASPI calculation a little
    signal = signal[100000:125000, :]
    reference = reference[100000:125000, :]

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

    msbg_ear_cfg = {
        "src_pos": "ff",
        "sample_rate": 44100,
        "equiv_0db_spl": 100,
        "ahr": 20,
    }

    audiogram_l = np.array([45, 20, 30, 35, 30, 45, 50, 50])
    audiogram_r = np.array([45, 25, 30, 40, 30, 40, 50, 50])
    audiogram_cfs = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])

    ear = Ear(**msbg_ear_cfg)

    audiogram_left = Audiogram(frequencies=audiogram_cfs, levels=audiogram_l)
    audiogram_right = Audiogram(frequencies=audiogram_cfs, levels=audiogram_r)

    enhancer = NALR(**nalr_cfg)
    compressor = Compressor(**compressor_cfg)

    nalr_fir, _ = enhancer.build(audiogram_left)
    out_l = enhancer.apply(nalr_fir, signal[:, 0])

    nalr_fir, _ = enhancer.build(audiogram_right)
    out_r = enhancer.apply(nalr_fir, signal[:, 1])

    out_l, _, _ = compressor.process(out_l)
    out_r, _, _ = compressor.process(out_r)

    enhanced_audio = np.stack([out_l, out_r], axis=1)
    enhanced_audio = np.tanh(enhanced_audio) * 100  # * 10000

    # Create discrete delta function (DDF) signal for time alignment
    ddf_signal = np.zeros(np.shape(signal))

    print(ddf_signal.shape)
    ddf_signal[:, 0] = unit_impulse(len(signal), int(MSBG_FS / 2))
    ddf_signal[:, 1] = unit_impulse(len(signal), int(MSBG_FS / 2))

    # Pass through MSBG hearing loss model
    reference_processed = listen(ear, reference, audiogram_left, audiogram_right)
    signal_processed = listen(ear, enhanced_audio, audiogram_left, audiogram_right)

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
