"""Test ha_metrics.ear_model module"""
# pylint: disable=import-error
import numpy as np
import pytest

from clarity.evaluator.ha_metric.ear_model import EarModel


def test_ear_model():
    """Test ear model"""
    np.random.seed(0)
    sig_len = 600
    samp_freq = 24000
    out_sig_len = sig_len * 24000 / samp_freq

    ref = np.random.random(size=sig_len)
    proc = np.random.random(size=sig_len)
    ear_model = EarModel(equalisation=0, nchan=10)

    ref_db, ref_bm, proc_db, proc_bm, ref_sl, proc_sl, freq_sample = ear_model.compute(
        reference=ref,
        reference_freq=samp_freq,
        processed=ref + proc,
        processed_freq=samp_freq,
        hearing_loss=np.array([45, 45, 35, 45, 60, 65]),
        level1=65,
        shift=0.0,
    )

    # check shapes
    assert ref_db.shape == (10, out_sig_len)
    assert proc_db.shape == (10, out_sig_len)
    assert ref_bm.shape == (10, out_sig_len)
    assert proc_bm.shape == (10, out_sig_len)
    assert ref_sl.shape == (10,)
    assert proc_sl.shape == (10,)

    # check values
    assert freq_sample == 24000
    assert np.sum(np.abs(ref_db)) == pytest.approx(
        102596.63767028379, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(proc_db)) == pytest.approx(
        4145.3884196835625, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(ref_bm)) == pytest.approx(
        65517.72934742906, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(proc_bm)) == pytest.approx(
        2366.401656815131, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(ref_sl)) == pytest.approx(
        291.3527365691821, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(proc_sl)) == pytest.approx(
        13.655317152968216, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_center_frequency():
    """Test center frequency"""
    ear_model = EarModel(equalisation=1, nchan=10)
    center_freq = ear_model.center_frequency(
        shift=None,
        low_freq=80,
        high_freq=8000,
        min_bw=24.7,
    )
    assert center_freq.shape == (10,)
    assert np.sum(center_freq) == pytest.approx(
        23935.19626226296, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_loss_parameters():
    """Test loss parameters"""

    ear_model = EarModel(equalisation=1, nchan=10)
    (
        attenuated_ohc,
        bandwidth,
        low_knee,
        compression_ratio,
        attenuated_ihc,
    ) = ear_model.loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
        audiometric_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
    )

    # check shapes
    assert attenuated_ohc.shape == (6,)
    assert bandwidth.shape == (6,)
    assert low_knee.shape == (6,)
    assert compression_ratio.shape == (6,)
    assert attenuated_ihc.shape == (6,)
    # check values
    assert np.sum(attenuated_ohc) == pytest.approx(
        220.39149328167292, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(bandwidth) == pytest.approx(
        15.041134665207498, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(low_knee) == pytest.approx(
        400.3914932816729, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(compression_ratio) == pytest.approx(
        6.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(attenuated_ihc) == pytest.approx(
        129.6085067183270, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "reference_freq,expected,expected_len",
    [(12000, 604.1522707137393, 1200), (44100, 162.6502502653759, 326)],
)
def test_resample(reference_freq, expected, expected_len):
    """Test resample"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = np.random.random(size=sig_len)
    ear_model = EarModel(equalisation=1, nchan=10)

    ref_signal_24, freq_sample_hz = ear_model.resample(
        reference_signal, reference_freq, target_sample_rate=24000
    )

    # check shapes
    assert expected_len == int(len(reference_signal) * 24000 / reference_freq)
    # check values
    assert np.sum(np.abs(ref_signal_24)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert freq_sample_hz == 24000


def test_input_align():
    """Test input align"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    ref, proc = ear_model.input_align(reference_signal, processed_signal)

    assert ref.shape == (600,)
    assert proc.shape == (600,)
    assert np.sum(np.abs(ref)) == pytest.approx(
        29892.167176853407, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(proc)) == pytest.approx(
        27199.009291096496, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_middle_ear():
    """Test middle ear"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    filtered_signal = ear_model.middle_ear(reference_signal, 24000)

    assert filtered_signal.shape == (600,)
    assert np.sum(np.abs(filtered_signal)) == pytest.approx(
        9241.220369749171, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "scale, bw_min, bw_max, expected",
    [
        (100.0, 1.0, 2.0, 1.0),
        (1000000.0, 1.0, 2.0, 2.0),
        (1000.0, 1.0, 2.0, 1.22),
    ],
)
def test_bandwidth_adjust(scale, bw_min, bw_max, expected):
    """Test bandwidth adjust"""
    ear_model = EarModel(equalisation=1, nchan=10)
    bw_adjusted = ear_model.bandwidth_adjust(
        control=scale * np.array([1, -1, 1]),
        bandwidth_min=bw_min,
        bandwidth_max=bw_max,
        level1=1,
    )
    assert bw_adjusted == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_env_compress_basilar_membrane():
    """Test env_compress_basilar_membrane"""
    np.random.seed(0)
    sig_len = 600
    env_sig = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001
    control = np.random.random(size=sig_len)

    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    (
        compressed_signal,
        compressed_basilar_membrane,
    ) = ear_model.env_compress_basilar_membrane(
        env_sig,
        bm,  # pylint: disable=invalid-name
        control,
        attn_ohc=0.01,
        threshold_low=70.0,
        compression_ratio=0.1,
        fsamp=24000,
        level1=140,
        threshold_high=100,
    )
    # check shapes
    assert compressed_signal.shape == (600,)
    assert compressed_basilar_membrane.shape == (600,)
    # check values
    assert np.mean(np.abs(compressed_signal)) == pytest.approx(
        15486012153068.807, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.mean(np.abs(compressed_basilar_membrane)) == pytest.approx(
        15415471156.59357, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_envelope_align():
    """Test envelope align"""

    np.random.seed(0)
    sig_len = 600
    scale = 1.1
    reference = np.random.random(size=sig_len)

    # Make output look like a shifted copy of the reference
    output = reference.copy()
    output[50:] = output[:-50]
    output[0:50] = 0
    output *= scale

    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    aligned_output = ear_model.envelope_align(
        reference, output, freq_sample=24000, corr_range=100
    )

    # check shapes and values
    assert aligned_output.shape == (600,)
    assert np.sum(np.abs(aligned_output)) == pytest.approx(
        299.1891022020615, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    # Check output is now aligned with the reference
    assert aligned_output[100] == pytest.approx(
        scale * reference[100], rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_envelope_sl():
    """Test envelope sl"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001
    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    reference, basilar_membrane = ear_model.envelope_sl(
        reference=ref,
        basilar_membrane=bm,
        attenuated_ihc=40.0,
        level1=120,
    )

    assert reference.shape == (600,)
    assert basilar_membrane.shape == (600,)
    assert np.sum(np.abs(reference)) == pytest.approx(
        42746.12859151134, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(basilar_membrane)) == pytest.approx(
        98.97646233693762, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_inner_hair_cell_adaptation():
    """Test inner hair cell adaptation"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001
    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    output_db, output_basilar_membrane = ear_model.inner_hair_cell_adaptation(
        reference_db=ref, reference_basilar_membrane=bm, delta=1.00, freq_sample=24000
    )

    # check shapes and values
    assert output_db.shape == (600,)
    assert output_basilar_membrane.shape == (600,)
    assert np.sum(np.abs(output_db)) == pytest.approx(
        298.9359292744365, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(output_basilar_membrane)) == pytest.approx(
        0.2963082865723811, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_basilar_membrane_add_noise():
    """Test basilar membrane add noise"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)

    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    noisy_reference = ear_model.basilar_membrane_add_noise(
        reference=ref, threshold=40, level1=120
    )

    # check shapes and values and that signal has changed
    assert noisy_reference.shape == (600,)
    assert np.sum(np.abs(noisy_reference)) == pytest.approx(
        298.919051930547, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert not np.allclose(noisy_reference, ref)

    # Check that adding on nearly 0 noise (-100 db) doesn't change the signal
    noisy_reference = ear_model.basilar_membrane_add_noise(
        reference=ref, threshold=-100, level1=120
    )
    assert np.allclose(noisy_reference, ref)


def test_group_delay_compensate():
    """Test group delay compensate"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))

    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    processed = ear_model.group_delay_compensate(
        reference=reference,
        bandwidths=np.array([30, 60, 90, 120]),
        center_freq=np.array([100, 200, 300, 400]),
        freq_sample=24000,
        min_bandwidth=24.7,
    )

    # check shapes and values
    assert processed.shape == (4, 600)
    assert np.sum(np.abs(processed)) == pytest.approx(
        1193.8088344682358, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_convert_rms_to_sl():
    """Test convert rms to sl"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=sig_len)
    control = np.random.random(size=sig_len)

    ear_model = EarModel(
        equalisation=1,
        nchan=10,
    )
    ref_db = ear_model.convert_rms_to_sl(
        reference=reference,
        control=control,
        attenuated_ohc=0.1,
        threshold_low=40,
        compression_ratio=10,
        attenuated_ihc=0.1,
        level1=120,
        threshold_high=100,
    )

    # check shapes and values
    assert ref_db.shape == (600,)
    assert np.sum(np.abs(ref_db)) == pytest.approx(
        34746.74406262155, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
