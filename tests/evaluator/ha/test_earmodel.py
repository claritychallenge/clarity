"""Test the Ear class in the clarity.evaluator.ha module"""

import numpy as np
import pytest

from clarity.evaluator.ha import Ear
from clarity.utils.audiogram import Audiogram


@pytest.fixture
def audiogram():
    """Return an instance of the Audiogram class for testing."""
    audiogram_levels = np.array([30, 40, 40, 65, 70, 65])
    audiogram_frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    return Audiogram(
        levels=audiogram_levels,
        frequencies=audiogram_frequencies,
    )


@pytest.fixture()
def ear_instance():
    """
    Return a function that returns an instance of the Ear class
    with the specified parameters.
    """

    def _ear_instance(
        same_size=True,
        equalisation=0,
        num_bands=32,
        m_delay=1,
    ):
        return Ear(
            signals_same_size=same_size,
            equalisation=equalisation,
            num_bands=num_bands,
            m_delay=m_delay,
        )

    return _ear_instance


@pytest.mark.parametrize(
    "same_size, equalisation, num_bands, delay",
    ([(True, 0, 32, 1), (True, 1, 10, 0), (False, 0, 10, 0), (False, 1, 32, 1)]),
)
def test_initialization(ear_instance, same_size, equalisation, num_bands, delay):
    """Test the initialization of the Ear class."""
    ear_instance = ear_instance(same_size, equalisation, num_bands, delay)
    assert ear_instance.equalisation == equalisation
    assert ear_instance.num_bands == num_bands
    assert ear_instance.m_delay == delay
    assert ear_instance.signals_same_size is same_size


def test_parameters_computation(ear_instance):
    """Test the computation of parameters in the Ear class."""
    ear_instance = ear_instance()
    assert len(ear_instance.center_freq) == ear_instance.num_bands
    assert len(ear_instance.center_freq_control) == ear_instance.num_bands
    assert ear_instance.bandwidth_1 is not None
    assert isinstance(ear_instance.compress_basilar_membrane_coef, dict)
    assert isinstance(ear_instance.middle_ear_coef, dict)


@pytest.mark.parametrize("same_size", ([True, False]))
def test_set_audiogram(ear_instance, audiogram, same_size):
    """Test the set_audiogram method of the Ear class."""
    ear_instance = ear_instance(same_size=same_size)
    # Set audiogram
    ear_instance.set_audiogram(audiogram)

    assert ear_instance.audiogram == audiogram
    assert ear_instance.reference_computed is False
    assert len(ear_instance.reference_cochlear_compression) == 0
    assert len(ear_instance.temp_reference_b) == 0

    if not ear_instance.signals_same_size:
        assert len(ear_instance.sincf) == 0
        assert len(ear_instance.coscf) == 0
        assert len(ear_instance.sincf_control) == 0
        assert len(ear_instance.coscf_control) == 0


def test_process_reference(ear_instance, audiogram):
    """Test the process_reference method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()

    # Create a dummy signal for testing
    num_samples = 1000
    signal = np.random.randn(num_samples)

    # Ensure audiogram is set before calling process_reference
    ear_instance.set_audiogram(audiogram)

    # Call process_reference
    (
        reference_db,
        reference_basilar_membrane,
        reference_sl,
    ) = ear_instance.process_reference(signal)

    # Add assertions based on expected behavior of process_reference
    assert len(reference_db) == ear_instance.num_bands
    assert len(reference_basilar_membrane) == ear_instance.num_bands
    assert len(reference_sl) == ear_instance.num_bands
    assert ear_instance.reference_computed is True

    assert np.sum(np.abs(reference_db)) == pytest.approx(
        900827.5913333015, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(reference_basilar_membrane)) == pytest.approx(
        573309.2052175319, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(reference_sl)) == pytest.approx(
        1074.0358414117686, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(ear_instance.coscf) == pytest.approx(
        71.72892210848208, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(ear_instance.sincf) == pytest.approx(
        -278.7015186488348, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(ear_instance.coscf_control) == pytest.approx(
        71.72892210848208, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(ear_instance.sincf_control) == pytest.approx(
        -278.7015186488348, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_process_reference_before_audiogram(ear_instance):
    """
    Test the process_reference method of the Ear class before setting the audiogram.
    """
    ear_instance = ear_instance()

    # Create a dummy signal for testing
    num_samples = 1000
    signal = np.random.randn(num_samples)

    # Call process_reference before setting the audiogram
    with pytest.raises(ValueError):
        ear_instance.process_reference(signal)


def test_process_enhanced(ear_instance, audiogram):
    """Test the process_enhanced method of the Ear class."""
    ear_instance = ear_instance()

    # Create a dummy signal for testing
    num_samples = 1000
    reference = np.random.randn(num_samples)
    enhanced = np.random.randn(num_samples)

    # Set the Audiogram
    ear_instance.set_audiogram(audiogram)

    # Call process_reference
    ear_instance.process_reference(reference)

    # Call process_enhanced
    enhanced_db, enhanced_basilar_membrane, enhanced_sl = ear_instance.process_enhanced(
        enhanced
    )

    # Add assertions based on expected behavior of process_enhanced
    assert len(enhanced_db) == ear_instance.num_bands
    assert len(enhanced_basilar_membrane) == ear_instance.num_bands
    assert len(enhanced_sl) == ear_instance.num_bands


def test_process_enhanced_before_reference(ear_instance, audiogram):
    """
    Test the process_enhanced method of the Ear class before calling process_reference.
    """
    ear_instance = ear_instance()

    # Create a dummy signal for testing
    num_samples = 1000
    enhanced = np.random.randn(num_samples)

    # Set the Audiogram
    ear_instance.set_audiogram(audiogram)

    # Call process_enhanced before process_reference
    with pytest.raises(ValueError):
        ear_instance.process_enhanced(enhanced)


def test_process_enhanced_before_audiogram(ear_instance):
    """
    Test the process_enhanced method of the Ear class before setting the audiogram.
    """
    ear_instance = ear_instance()

    # Create a dummy signal for testing
    num_samples = 1000
    enhanced = np.random.randn(num_samples)

    # Call process_enhanced before setting the audiogram
    with pytest.raises(ValueError):
        ear_instance.process_enhanced(enhanced)


def test_process_common(ear_instance, audiogram):
    """Test the process_common method of the Ear class."""
    ear_instance = ear_instance()

    # Create a dummy signal for testing
    num_samples = 1000
    signal = np.random.randn(num_samples)

    # Set audiogram
    ear_instance.set_audiogram(audiogram)

    # Ensure reference is computed before calling process_common
    ear_instance.process_reference(signal)

    # Call process_common
    signal_db, signal_basilar_membrane, signal_sl = ear_instance.process_common(signal)

    # Add assertions based on expected behavior of process_common
    assert len(signal_db) == ear_instance.num_bands
    assert len(signal_basilar_membrane) == ear_instance.num_bands
    assert len(signal_sl) == ear_instance.num_bands


def test_center_frequency(ear_instance):
    """Test method center frequency"""
    ear_instance = ear_instance(num_bands=10)

    center_freq = ear_instance.center_frequency(
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )
    assert center_freq.shape == (10,)
    assert np.sum(center_freq) == pytest.approx(
        23935.19626226296, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_center_frequency_shift(ear_instance):
    """Test method center frequency"""
    ear_instance = ear_instance(num_bands=10)

    center_freq = ear_instance.center_frequency(
        shift=0.1,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )
    assert center_freq.shape == (10,)
    assert np.sum(center_freq) == pytest.approx(
        33526.27322859052, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "audiometric_freq", (np.array([250, 500, 1000, 2000, 4000, 6000]), None)
)
def test_loss_parameters(ear_instance, audiometric_freq):
    """Test loss parameters"""

    ear_instance = ear_instance()

    (
        attenuated_ohc,
        bandwidth,
        low_knee,
        compression_ratio,
        attenuated_ihc,
    ) = ear_instance.loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
        audiometric_freq=audiometric_freq,
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


def test_middle_ear(ear_instance):
    """Test middle ear"""
    ear_instance = ear_instance()

    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    filtered_signal = ear_instance.middle_ear(reference_signal)

    assert filtered_signal.shape == (600,)
    assert np.sum(np.abs(filtered_signal)) == pytest.approx(
        9241.220369749171, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_gammatone_basilar_membrane(ear_instance):
    """Test gammatone_basilar_membrane"""
    np.random.seed(0)

    ear_instance = ear_instance()

    # Create a dummy signal for testing
    num_samples = 1000
    signal = np.random.randn(num_samples)

    # Create dummy cosine and sine for center frequency
    coscf = np.random.randn(num_samples)
    sincf = np.random.randn(num_samples)

    # Call gammatone_basilar_membrane
    envelope, basilar_membrane = ear_instance.gammatone_basilar_membrane(
        signal=signal,
        bandwidth=100,  # Example bandwidth
        center_freq=1000,  # Example center frequency
        coscf=coscf,
        sincf=sincf,
    )

    # Add assertions based on expected behavior of gammatone_basilar_membrane
    assert len(envelope) == num_samples
    assert len(basilar_membrane) == num_samples

    assert np.sum(np.abs(envelope)) == pytest.approx(
        1605.0323490431642, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(basilar_membrane)) == pytest.approx(
        2311.7749724177334, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_bandwidth_adjust(ear_instance):
    """Test bandwidth_adjust method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()
    # Create dummy control signal, min bandwidth, and max bandwidth for testing
    control = np.random.rand(1000)
    bandwidth_min = 20
    bandwidth_max = 200

    # Call bandwidth_adjust
    adjusted_bandwidth = ear_instance.bandwidth_adjust(
        control, bandwidth_min, bandwidth_max
    )

    # Add assertions based on expected behavior of bandwidth_adjust
    assert isinstance(adjusted_bandwidth, float)
    assert np.sum(adjusted_bandwidth) == pytest.approx(
        56.68477203105376, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_bandwidth_adjust_control_db_100(ear_instance):
    """Test bandwidth_adjust method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()
    # Create dummy control signal, min bandwidth, and max bandwidth for testing
    control = np.random.rand(1000) * 200
    bandwidth_min = 20
    bandwidth_max = 200

    # Call bandwidth_adjust
    adjusted_bandwidth = ear_instance.bandwidth_adjust(
        control, bandwidth_min, bandwidth_max
    )

    # Add assertions based on expected behavior of bandwidth_adjust
    assert isinstance(adjusted_bandwidth, float)
    assert np.sum(adjusted_bandwidth) == pytest.approx(
        bandwidth_max, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_bandwidth_adjust_control_db_50(ear_instance):
    """Test bandwidth_adjust method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()
    # Create dummy control signal, min bandwidth, and max bandwidth for testing
    control = np.random.rand(1000) * 0.1
    bandwidth_min = 20
    bandwidth_max = 200

    # Call bandwidth_adjust
    adjusted_bandwidth = ear_instance.bandwidth_adjust(
        control, bandwidth_min, bandwidth_max
    )

    # Add assertions based on expected behavior of bandwidth_adjust
    assert isinstance(adjusted_bandwidth, float)
    assert np.sum(adjusted_bandwidth) == pytest.approx(
        bandwidth_min, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_env_compress_basilar_membrane(ear_instance):
    """Test env_compress_basilar_membrane method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()

    # Create dummy inputs for testing
    num_samples = 1000
    envsig = np.random.rand(num_samples)
    bm = np.random.rand(num_samples)
    control = np.random.rand(num_samples)
    attn_ohc = 0.5
    threshold_low = 20
    compression_ratio = 2
    threshold_high = 100

    # Call env_compress_basilar_membrane
    (
        compressed_signal,
        compressed_basilar_membrane,
    ) = ear_instance.env_compress_basilar_membrane(
        envsig=envsig,
        bm=bm,
        control=control,
        attn_ohc=attn_ohc,
        threshold_low=threshold_low,
        compression_ratio=compression_ratio,
        threshold_high=threshold_high,
    )

    # Add assertions based on expected behavior of env_compress_basilar_membrane
    assert len(compressed_signal) == num_samples
    assert len(compressed_basilar_membrane) == num_samples
    assert np.sum(compressed_signal) == pytest.approx(
        66.18340924327552, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(compressed_basilar_membrane) == pytest.approx(
        68.14948989999604, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_envelope_sl(ear_instance):
    """Test envelope_sl method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()

    # Create dummy inputs for testing
    num_samples = 1000
    envelope = np.random.rand(num_samples)
    basilar_membrane = np.random.rand(num_samples)
    attenuated_ihc = 0.5

    # Call envelope_sl
    _envelope, _basilar_membrane = ear_instance.envelope_sl(
        envelope=envelope,
        basilar_membrane=basilar_membrane,
        attenuated_ihc=attenuated_ihc,
    )

    # Add assertions based on expected behavior of envelope_sl
    assert len(_envelope) == num_samples
    assert len(_basilar_membrane) == num_samples

    assert np.sum(_envelope) == pytest.approx(
        55770.75943438894, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(_basilar_membrane) == pytest.approx(
        120016.14816597928, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_basilar_membrane_add_noise(ear_instance):
    """Test basilar_membrane_add_noise method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()

    # Create dummy inputs for testing
    num_samples = 600
    ref = np.random.rand(num_samples)

    # Call basilar_membrane_add_noise
    noisy_reference = ear_instance.basilar_membrane_add_noise(
        signal=ref, threshold=10, level1=120
    )

    # check shapes and values and that signal has changed
    assert noisy_reference.shape == (600,)
    assert np.sum(np.abs(noisy_reference)) == pytest.approx(
        298.9215889219827, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert not noisy_reference == pytest.approx(
        ref, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    # Check that adding on nearly 0 noise (-100 db) doesn't change the signal
    noisy_reference = ear_instance.basilar_membrane_add_noise(
        signal=ref, threshold=-100, level1=120
    )
    assert noisy_reference == pytest.approx(
        ref, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_group_delay_compensate(ear_instance):
    """Test group_delay_compensate method of the Ear class."""
    ear_instance = ear_instance(num_bands=4, m_delay=1)

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))

    processed = ear_instance.group_delay_compensate(
        input_signal=reference,
        bandwidths=np.array([30, 60, 90, 120]),
        center_freq=np.array([100, 200, 300, 400]),
        ear_q=9.26449,
        min_bandwidth=24.7,
    )

    # check shapes and values
    assert processed.shape == (4, 600)
    assert np.sum(np.abs(processed)) == pytest.approx(
        1193.8088344682358, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_convert_rms_to_sl(ear_instance):
    """Test convert_rms_to_sl method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()
    ear_instance.level1 = 120

    sig_len = 600
    reference = np.random.random(size=sig_len)
    control = np.random.random(size=sig_len)

    ref_db = ear_instance.convert_rms_to_sl(
        reference=reference,
        control=control,
        attenuated_ohc=0.1,
        threshold_low=40,
        compression_ratio=10,
        attenuated_ihc=0.1,
    )

    # check shapes and values
    assert ref_db.shape == (600,)
    assert np.sum(np.abs(ref_db)) == pytest.approx(
        34746.74406262155, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_envelope_align(ear_instance):
    """Test envelope_align method of the Ear class."""

    ear_instance = ear_instance()
    np.random.seed(0)
    sig_len = 600
    scale = 1.1
    reference = np.random.random(size=sig_len)

    # Make output look like a shifted copy of the reference
    output = reference.copy()
    output[50:] = output[:-50]
    output[0:50] = 0
    output *= scale

    aligned_output = ear_instance.envelope_align(reference, output, corr_range=100)

    # check shapes and values
    assert aligned_output.shape == (600,)
    assert np.sum(np.abs(aligned_output)) == pytest.approx(
        299.1891022020615, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    # Check output is now aligned with the reference
    assert aligned_output[100] == pytest.approx(
        scale * reference[100], rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_envelope_align_long(ear_instance):
    """Test envelope_align method of the Ear class."""

    ear_instance = ear_instance()
    np.random.seed(0)
    sig_len = 2400
    scale = 1.1
    reference = np.random.random(size=sig_len)

    # Make output look like a shifted copy of the reference
    output = reference.copy()
    output[50:] = output[:-50]
    output[0:50] = 0
    output *= scale

    aligned_output = ear_instance.envelope_align(reference, output, corr_range=100)

    # check shapes and values
    assert aligned_output.shape == (sig_len,)
    assert np.sum(np.abs(aligned_output)) == pytest.approx(
        1304.2657154489764, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    # Check output is now aligned with the reference
    assert aligned_output[100] == pytest.approx(
        scale * reference[100], rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_find_noiseless_boundaries(ear_instance):
    """Test find_noiseless_boundaries method of the Ear class."""
    np.random.seed(0)

    ear_instance = ear_instance()

    # Create a dummy signal with some silence
    num_samples = 1000
    signal = np.zeros(num_samples)
    silence_start = 100
    silence_end = 200
    signal[silence_start:silence_end] = 0.1 * np.random.randn(
        silence_end - silence_start
    )

    # Call the find_noiseless_boundaries method
    start, end = ear_instance.find_noiseless_boundaries(signal)

    # Assertions based on expected behavior
    assert start == silence_start
    assert end == silence_end - 1


def test_gammatone_bandwidth_demodulation(ear_instance):
    """Test gammatone_bandwidth_demodulation method of the Ear class."""
    ear_instance = ear_instance()

    centre_freq_sin, centre_freq_cos = ear_instance.gammatone_bandwidth_demodulation(
        npts=100,
        tpt=0.001,
        center_freq=1000,
    )
    assert centre_freq_sin.shape == (100,)
    assert centre_freq_cos.shape == (100,)
    assert np.sum(centre_freq_sin) == pytest.approx(
        -0.3791946274493412, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(centre_freq_cos) == pytest.approx(
        -0.39460748051808026, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_inner_hair_cell_adaptation(ear_instance):
    """Test inner hair cell adaptation method of the Ear class."""
    ear_instance = ear_instance()

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001

    output_db, output_basilar_membrane = ear_instance.inner_hair_cell_adaptation(
        signal_db=ref, basilar_membrane=bm, delta=1.00, freq_sample=24000
    )

    # check shapes and values
    assert output_db.shape == (600,)
    assert output_basilar_membrane.shape == (600,)
    assert np.sum(np.abs(output_db)) == pytest.approx(
        298.9359292744365, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(output_basilar_membrane)) == pytest.approx(
        0.2963294865299153, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_input_align(ear_instance):
    """Test input align method of the Ear class."""
    ear_instance = ear_instance()

    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    ear_instance.reference_align = reference_signal

    proc = ear_instance.input_align(processed_signal)

    assert proc.shape == (600,)
    assert np.sum(np.abs(proc)) == pytest.approx(
        27199.009291096496, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_str_representation(ear_instance):
    # Create an instance of HAAQI_V1
    ear_instance = ear_instance()

    # Assert that calling str() on the instance returns the expected string
    assert str(ear_instance) == "HearingAid Model for HAAQI, HASQI and HAPPI."
