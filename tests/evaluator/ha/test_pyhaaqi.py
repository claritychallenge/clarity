"""Test the pyhaaqi module."""

import numpy as np
import pytest

from clarity.evaluator.ha import HaaqiV1
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
def haaqi_instance():
    """Return an instance of the HAAQI_V1 class for testing"""

    def _haaqi_instance(
        equalisation=1,
        num_bands=32,
        silence_threshold=2.5,
        add_noise=0.0,
        segment_size=8,
        n_cepstral_coef=6,
        segment_covariance=16,
        ear_model_kwargs=None,
    ):
        return HaaqiV1(
            equalisation=equalisation,
            num_bands=num_bands,
            silence_threshold=silence_threshold,
            add_noise=add_noise,
            segment_size=segment_size,
            n_cepstral_coef=n_cepstral_coef,
            segment_covariance=segment_covariance,
            ear_model_kwargs=ear_model_kwargs,
        )

    return _haaqi_instance


def test_initialization(haaqi_instance):
    """Test the initialization of the HAAQI_V1 class."""
    instance = haaqi_instance()

    assert instance.num_bands == 32
    assert instance.silence_threshold == 2.5
    assert instance.add_noise == 0.0
    assert instance.segment_size == 8
    assert instance.n_cepstral_coef == 6
    assert instance.segment_covariance == 16
    assert instance.audiogram is None
    assert instance.level1 == 65.0
    assert instance.ear_model is not None
    # Add more assertions as needed for other attributes


def test_set_audiogram(haaqi_instance, audiogram):
    """Test the set_audiogram method of the HAAQI_V1 class."""
    instance = haaqi_instance()
    instance.set_audiogram(audiogram)
    assert instance.audiogram == audiogram
    assert instance.ear_model.audiogram == audiogram


# Add more test cases to cover other methods and functionalities of the HAAQI_V1 class


def test_process(haaqi_instance, audiogram):
    """Test the process method of the HAAQI_V1 class."""
    # Initialize HAAQI_V1 instance

    np.random.seed(42)

    haqqi_instance = haaqi_instance()
    haqqi_instance.set_audiogram(audiogram)

    # Generate reference and enhanced signals (example)
    reference_signal = np.random.randn(600)
    enhanced_signal = np.random.randn(600)

    # Process the signals
    score = haqqi_instance.process(reference_signal, 24000, enhanced_signal, 24000)

    # Perform assertions
    assert isinstance(score, float)
    # Add more specific assertions based on your expectations for the processed score
    assert score == pytest.approx(
        0.441399286739319, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_set_reference(haaqi_instance, audiogram):
    """Test the set_reference method of the HAAQI_V1 class."""
    np.random.seed(42)

    # Initialize HAAQI_V1 instance
    haqqi_instance = haaqi_instance()
    haqqi_instance.set_audiogram(audiogram)

    # Generate reference signal (example)
    reference_signal = np.random.randn(600)

    # Set the reference signal
    haqqi_instance.set_reference(reference_signal, 24000)

    # Perform assertions
    assert haqqi_instance.audiogram is not None
    assert haqqi_instance.reference_basilar_membrane is not None
    assert haqqi_instance.reference_sl is not None
    assert haqqi_instance.reference_smooth is not None
    assert haqqi_instance.reference_linear_magnitude is not None
    assert haqqi_instance.segments_above_threshold >= 0
    assert isinstance(haqqi_instance.index_above_threshold, np.ndarray)
    # Add more specific assertions based on your expectations for the updated attributes
    assert np.sum(haqqi_instance.reference_basilar_membrane) == pytest.approx(
        385.8043465957131, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(haqqi_instance.reference_sl) == pytest.approx(
        526.453739008096, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(haqqi_instance.reference_smooth) == pytest.approx(
        2140.4223790398237, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(haqqi_instance.reference_linear_magnitude) == pytest.approx(
        1.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_score(haaqi_instance, audiogram):
    """Test the score method of the HAAQI_V1 class."""

    # Initialize HAAQI_V1 instance
    haqqi_instance = haaqi_instance()
    haqqi_instance.set_audiogram(audiogram)

    # Generate reference and enhanced signals (example)
    reference_signal = np.random.randn(600)
    enhanced_signal = np.random.randn(600)

    # Set the reference signal
    haqqi_instance.set_reference(reference_signal, 24000)

    # Call the score method with the enhanced signal
    combined_model, nonlinear_model, linear_model, raw_data = haqqi_instance.score(
        enhanced_signal, 24000
    )

    # Perform assertions
    assert isinstance(combined_model, float)
    assert isinstance(nonlinear_model, float)
    assert isinstance(linear_model, float)
    assert isinstance(raw_data, np.ndarray)
    # Add more specific assertions based on your expectations for the returned values

    assert combined_model == pytest.approx(
        0.4550928987217293, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert nonlinear_model == pytest.approx(
        0.6076413195145778, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert linear_model == pytest.approx(
        0.6368854502282981, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(raw_data) == pytest.approx(
        2.648552522125631, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_linear_model(haaqi_instance, audiogram):
    """Test the linear_model method of the HAAQI_V1 class."""
    # Initialize HAAQI_V1 instance
    haqqi_instance = haaqi_instance()
    haqqi_instance.set_audiogram(audiogram)

    # Generate simulated enhanced signal spectrum in dB SL
    enhanced_sl = np.random.standard_normal(100)
    haqqi_instance.reference_linear_magnitude = np.random.standard_normal(100)

    # Call the linear_model method with the enhanced signal spectrum
    linear_model, d_loud, d_norm = haqqi_instance.linear_model(enhanced_sl)

    # Perform assertions
    assert isinstance(linear_model, float)
    assert isinstance(d_loud, float)
    assert isinstance(d_norm, float)
    # Add more specific assertions based on your expectations for the returned values
    assert linear_model == pytest.approx(
        0.28020388909904365, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert d_loud == pytest.approx(
        0.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert d_norm == pytest.approx(
        0.4175914889702588, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_non_linear_model(haaqi_instance, audiogram):
    """Test the non_linear_model method of the HAAQI_V1 class."""
    np.random.seed(42)
    # Initialize HAAQI_V1 instance
    haaqi_instance = haaqi_instance(num_bands=10)
    haaqi_instance.set_audiogram(audiogram)

    # Generate simulated enhanced signal spectrum in dB SL and basilar membrane movement
    haaqi_instance.set_reference(np.random.standard_normal(600), 24000)

    enhanced_db = np.random.standard_normal((10, 600))
    enhanced_basilar_membrane = np.random.standard_normal((10, 600))

    # Call the non_linear_model method with the simulated data
    (
        nonlinear_model,
        mel_cepstral_high,
        basilar_membrane_sync5,
    ) = haaqi_instance.non_linear_model(enhanced_db, enhanced_basilar_membrane)

    # Perform assertions
    assert isinstance(nonlinear_model, float)
    assert isinstance(mel_cepstral_high, float)
    assert isinstance(basilar_membrane_sync5, float)
    # Add more specific assertions based on your expectations for the returned values

    assert nonlinear_model == pytest.approx(
        0.4855342000239626, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cepstral_high == pytest.approx(
        0.8399715403081676, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert basilar_membrane_sync5 == pytest.approx(
        0.15723904904739136, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_env_smooth(haaqi_instance):
    """Test the env_smooth method of the HAAQI_V1 class."""

    np.random.seed(0)

    sig_len = 600
    segment_size = 1  # ms
    envelopes = np.random.random(size=(4, sig_len))
    sample_rate = 24000

    haaqi_instance = haaqi_instance(num_bands=4, segment_size=segment_size)

    smooth = haaqi_instance.env_smooth(envelopes=envelopes)

    # check shapes and values
    expected_length = 2 * sig_len / (sample_rate * segment_size / 1000)
    assert smooth.shape == (4, expected_length)
    assert np.sum(np.abs(smooth)) == pytest.approx(
        100.7397658862719, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_melcor9(haaqi_instance, audiogram):
    """Test the melcor9 method of the HAAQI_V1 class."""
    haaqi_instance = haaqi_instance(
        num_bands=4,
        segment_size=4,
        n_cepstral_coef=6,
        silence_threshold=12,
        add_noise=0.00,
    )
    haaqi_instance.set_audiogram(audiogram)

    np.random.seed(0)
    sig_len = 6000
    reference = np.random.random(size=sig_len)
    distorted = 20 * np.random.random(size=(4, sig_len))  # noqa: F841

    haaqi_instance.set_reference(reference, 24000)

    mel_cep_ave, mel_cep_low, mel_cep_high, mel_cep_mod = haaqi_instance.melcor9(
        signal=distorted,
    )

    assert mel_cep_mod.shape == (8,)

    assert mel_cep_ave == pytest.approx(
        0.8193269402657135, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_low == pytest.approx(
        0.997632785696832, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_high == pytest.approx(
        0.6410210948345949, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    print(mel_cep_mod)
    assert mel_cep_mod == pytest.approx(
        np.array(
            [
                0.99997632,
                0.99976785,
                0.99866212,
                0.99212485,
                0.95449984,
                0.66563292,
                0.80983534,
                0.13411628,
            ]
        ),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_melcor9_equal_input(haaqi_instance, audiogram):
    """Test the melcor9 method of the HAAQI_V1 class with equal inputs"""

    haaqi_instance = haaqi_instance(
        num_bands=4,
        segment_size=4,
        n_cepstral_coef=6,
        silence_threshold=12,
        add_noise=0.00,
    )
    haaqi_instance.set_audiogram(audiogram)

    np.random.seed(0)
    sig_len = 6000
    reference = np.random.random(size=sig_len)
    haaqi_instance.set_reference(reference, 24000)

    mel_cep_ave, mel_cep_low, mel_cep_high, mel_cep_mod = haaqi_instance.melcor9(
        signal=haaqi_instance.reference_smooth,
    )

    assert mel_cep_mod.shape == (8,)

    assert mel_cep_ave == pytest.approx(
        1.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_low == pytest.approx(
        1.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_high == pytest.approx(
        1.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_mod == pytest.approx(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_melcor9_crosscovmatrix(haaqi_instance, audiogram):
    """Test melcor9 crosscovmatrix"""
    haaqi_instance = haaqi_instance(
        num_bands=6,
        segment_size=4,
        n_cepstral_coef=6,
        silence_threshold=12,
        add_noise=0.00,
    )
    haaqi_instance.set_audiogram(audiogram)

    np.random.seed(0)
    sig_len = 600
    n_modulations = 4
    n_basis = 6
    reference = np.random.random(size=(n_basis, sig_len))
    processed = np.random.random(size=(n_basis, sig_len))
    basilar_membrane = np.random.random(size=(n_modulations, sig_len))

    cross_cov_matrix = haaqi_instance.melcor9_crosscovmatrix(
        b=basilar_membrane,
        nmod=n_modulations,  # n modulation channels
        nbasis=n_basis,  # n cepstral coefficient
        nsamp=sig_len,
        nfir=32,
        reference_cep=reference,
        processed_cep=processed,
    )

    assert cross_cov_matrix.shape == (n_modulations, n_basis)
    assert np.mean(cross_cov_matrix) == pytest.approx(
        0.9981200876213588, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_spectrum_diff(haaqi_instance, audiogram):
    """Test spectrum diff"""
    haaqi_instance = haaqi_instance(
        num_bands=4,
        segment_size=4,
        n_cepstral_coef=6,
        silence_threshold=12,
        add_noise=0.00,
    )
    haaqi_instance.set_audiogram(audiogram)

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=sig_len)
    processed = np.random.random(size=4)

    haaqi_instance.set_reference(reference, 24000)
    dloud, dnorm, dslope = haaqi_instance.spectrum_diff(processed_sl=processed)

    # Check shapes
    assert dloud.shape == (3,)
    assert dnorm.shape == (3,)
    assert dslope.shape == (3,)

    # Check values
    assert dloud == pytest.approx(
        np.array([0.568892340367931, 0.6658124414755029, 0.2844461701839655]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )
    assert dnorm == pytest.approx(
        np.array([1.0951514258294566, 1.1133879280049916, 0.3935881735702844]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )
    assert dslope == pytest.approx(
        np.array([0.7840940506954968, 1.2714387556494948, 0.4234018079877836]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_bm_covary_ok(haaqi_instance, audiogram):
    """Test bm covary"""
    haaqi_instance = haaqi_instance(num_bands=4, segment_covariance=4)
    haaqi_instance.set_audiogram(audiogram)

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    signal_cross_cov, ref_mean_square, proc_mean_square = haaqi_instance.bm_covary(
        reference_basilar_membrane=reference,
        processed_basilar_membrane=reference + 0.4 * processed,
    )

    # Check shapes
    assert signal_cross_cov.shape == (4, 12)
    assert ref_mean_square.shape == (4, 12)
    assert proc_mean_square.shape == (4, 12)

    # Check values
    assert np.sum(np.abs(signal_cross_cov)) == pytest.approx(
        46.43343502040397, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(ref_mean_square)) == pytest.approx(
        16.65809872708418, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(proc_mean_square)) == pytest.approx(
        26.110255782462076, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_bm_covary_error(haaqi_instance, audiogram):
    """Test bm covary fails when segment size too small"""
    haaqi_instance = haaqi_instance(
        num_bands=4,
        segment_covariance=2,
    )
    haaqi_instance.set_audiogram(audiogram)

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    with pytest.raises(ValueError):
        (
            _signal_cross_cov,
            _ref_mean_square,
            _proc_mean_square,
        ) = haaqi_instance.bm_covary(
            reference_basilar_membrane=reference,
            processed_basilar_membrane=reference + 0.4 * processed,
        )


def test_ave_covary2(haaqi_instance, audiogram):
    """Test ave covary2 method of the HAAQI_V1 class."""
    haaqi_instance = haaqi_instance(
        num_bands=4,
        silence_threshold=0.6,
    )
    haaqi_instance.set_audiogram(audiogram)

    np.random.seed(0)
    sig_len = 600

    signal_cross_cov = np.random.random(size=(4, sig_len))
    ref_mean_square = np.random.random(size=(4, sig_len))

    ave_covariance, ihc_sync_covariance = haaqi_instance.ave_covary2(
        signal_cross_covariance=signal_cross_cov,
        reference_signal_mean_square=ref_mean_square,
        lp_filter_order=np.array([1, 3, 5, 5, 5, 5]),
        freq_cutoff=1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    )

    assert len(ihc_sync_covariance) == 6

    assert ave_covariance == pytest.approx(
        0.5129961720524688, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(ihc_sync_covariance) == pytest.approx(
        3.057984614887033, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    assert ihc_sync_covariance == pytest.approx(
        [
            0.5072428879657319,
            0.5051636721505166,
            0.5087880598896098,
            0.511743935480639,
            0.5124431322756856,
            0.5126029271248499,
        ],
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )
