from clarity.enhancer.compressor import Compressor
import numpy as np

DEFAULT_FS = 44100


def test_compressor_set_attack():
    c = Compressor()
    c.set_attack(1000)

    assert c.attack == 1.0 / DEFAULT_FS


def test_compressor_set_release():
    c = Compressor()
    c.set_release(1000)

    assert c.release == 1.0 / DEFAULT_FS


def test_compressor_process():
    c = Compressor()
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    output, rms, comp_ratios = c.process(signal)

    assert len(output) == len(signal)
    assert np.all(rms >= 0.0)
