import numpy as np
import pytest

from clarity.evaluator.msbg.msbg_utils import (
    firwin2,
    gen_eh2008_speech_noise,
    gen_tone,
    measure_rms,
    pad,
)

# pylint: disable=W0613
# pylint false positives due to fixtures. pylint-pytest does not seem to work :(


@pytest.fixture(name="use_numpy")
def fixture_use_numpy():
    """Set numpy seed and print options for each test"""
    np.random.seed(0)
    np_print_opts = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=1000)
    yield
    np.set_printoptions(**np_print_opts)


def test_gen_eh2008_speech_noise(regtest, use_numpy):
    signal = gen_eh2008_speech_noise(0.1, 44100.0, 0.0)
    with regtest:
        print(signal[-9:])


def test_gen_tone(regtest, use_numpy):
    signal = gen_tone(500, 0.1, 44100.0, 20.0)
    with regtest:
        print(signal[-9:])


def test_firwin2(regtest, use_numpy):
    params = firwin2(128, [0.0, 0.1, 0.9, 1.0], [0, 1.0, 1.0, 0.0])
    with regtest:
        print(params)


def test_pad(regtest, use_numpy):
    padded = pad(np.array([1.0, 1.0, 1.0, 1.0]), 100)
    with regtest:
        print(padded)


def test_measure_rms(use_numpy, regtest):
    signal = gen_tone(500, 0.05, 44100.0, 20.0)
    noise = gen_eh2008_speech_noise(0.05, 44100.0, 0.0)
    rms, idx, rel_dB_thresh, active = measure_rms(signal + noise, 44100, 0.0, 10.0)
    with regtest:
        print(f"{rms:.7f}, {idx}, {rel_dB_thresh:.7f}, {active:.7f}")
