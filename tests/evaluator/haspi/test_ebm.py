"""Tests for ebm module"""
import numpy as np
import pytest

from clarity.evaluator.haspi.ebm import (
    add_noise,
    cepstral_correlation_coef,
    env_filter,
    fir_modulation_filter,
    modulation_cross_correlation,
)


def test_env_filter_ok() -> None:
    """Test for env_filter function"""
    reference_env, processed_env = env_filter(
        reference_db=np.array([[80, 80, 80, 80]]),
        processed_db=np.array([[40, 50, 60, 70]]),
        filter_cutoff=1000,
        freq_sub_sample=4000,
        freq_samp=16000,
    )
    assert reference_env == pytest.approx(
        48.84481105, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert processed_env == pytest.approx(
        32.32800458, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "ref_db, proc_db, filter_cutoff, freq_sub_sample, freq_sample, expected",
    [
        (
            np.array([[80, 80, 80, 80]]),
            np.array([[40, 50, 60, 70]]),
            4000,
            1000,
            16000,
            ValueError,
        ),
        (
            np.array([[80, 80, 80, 80]]),
            np.array([[40, 50, 60, 70]]),
            4000,
            32000,
            16000,
            ValueError,
        ),
        (
            np.array([[80, 80, 80, 80]]),
            np.array([40, 50, 60, 70]),  # Wrong shape
            1000,
            4000,
            16000,
            ValueError,
        ),
    ],
)
def test_env_filter_error(
    ref_db, proc_db, filter_cutoff, freq_sub_sample, freq_sample, expected
) -> None:
    """Test for env_filter function"""
    with pytest.raises(expected):
        _reference_env, _processed_env = env_filter(
            reference_db=ref_db,
            processed_db=proc_db,
            filter_cutoff=filter_cutoff,
            freq_sub_sample=freq_sub_sample,
            freq_samp=freq_sample,
        )


def test_cepstral_correlation_coef_ok() -> None:
    """Test for cepstral_correlation_coef function"""
    np.random.seed(0)
    n_basis = 4
    sig_len = 100
    n_bands = 2
    random_sig1 = 100 * np.random.normal(size=(sig_len, n_bands))
    random_sig2 = 100 * np.random.normal(size=(sig_len, n_bands))
    ref_cep, proc_cep = cepstral_correlation_coef(
        reference_db=random_sig1,
        processed_db=random_sig1 + random_sig2,
        thresh_cep=10.0,
        thresh_nerve=10.0,
        nbasis=n_basis,
    )
    assert ref_cep.shape == (72, n_basis)
    assert proc_cep.shape == (72, n_basis)
    assert np.mean(np.abs(ref_cep)) == pytest.approx(
        82.4395877477838, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.mean(np.abs(proc_cep)) == pytest.approx(
        110.74820232861624, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_cepstral_correlation_coef_error() -> None:
    """Test for cepstral_correlation_coef function"""
    np.random.seed(0)
    n_basis = 4
    sig_len = 100
    n_bands = 2
    random_sig1 = 100 * np.random.normal(size=(sig_len, n_bands))
    random_sig2 = 100 * np.random.normal(size=(sig_len, n_bands))
    with pytest.raises(ValueError):
        _ref_cep, _proc_cep = cepstral_correlation_coef(
            reference_db=random_sig1,
            processed_db=random_sig1 + random_sig2,
            thresh_cep=1000.0,  # Crazy high threshold to test that error is raised
            thresh_nerve=1000.0,
            nbasis=n_basis,
        )


def test_add_noise() -> None:
    """Test for add_noise function"""
    np.random.seed(0)

    reference = 100 * np.random.normal(size=(100, 4))
    noisy_reference = add_noise(reference, 100.0)
    assert np.mean(np.abs(noisy_reference)) == pytest.approx(109.84025018392005)
    assert noisy_reference.shape == reference.shape

    # Add 0 dB noise should not change the signal
    reference = 100 * np.random.normal(size=(100, 4))
    noisy_reference = add_noise(reference, 0)
    assert noisy_reference == pytest.approx(
        reference, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_fir_modulation_filter() -> None:
    """Test for fir_modulation_filter function"""
    np.random.seed(0)

    in_center_freqs = np.array([2, 6, 10, 16, 25, 40, 64, 100])

    ref_mod, proc_mod, out_center_freqs = fir_modulation_filter(
        reference_envelope=100 * np.random.normal(size=(100, 4)),
        processed_envelope=100 * np.random.normal(size=(100, 4)),
        freq_sub_sampling=1000,
        # Adding a couple of invalid center freqs to check that they are removed
        center_frequencies=np.append(in_center_freqs, [500, 2000]),
    )

    # Check return value have expected shape
    assert ref_mod.shape == (4, len(in_center_freqs), 100)
    assert proc_mod.shape == (4, len(in_center_freqs), 100)
    assert out_center_freqs.shape == in_center_freqs.shape
    # Check return value have expected values
    assert in_center_freqs == pytest.approx(
        out_center_freqs, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.mean(np.abs(ref_mod)) == pytest.approx(
        14.477262943878237, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.mean(np.abs(proc_mod)) == pytest.approx(
        13.85669526424176, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_modulation_cross_correlation() -> None:
    """Test for modulation_cross_correlation function"""
    np.random.seed(0)

    n_cepstral_bands = 5
    n_mod_filters = 6
    signal_length = 100
    signal_shape = (n_cepstral_bands, n_mod_filters, signal_length)
    ref_mod = 100 * np.random.normal(size=signal_shape)
    proc_mod = 100 * np.random.normal(size=signal_shape)

    # All channels should be perfectly correlated with themselves
    cross_corr = modulation_cross_correlation(
        reference_modulation=ref_mod, processed_modulation=ref_mod
    )
    assert cross_corr.shape == (n_mod_filters,)
    assert cross_corr == pytest.approx(
        1.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    # Check an expected value for a pair of random signals
    cross_corr = modulation_cross_correlation(
        reference_modulation=ref_mod, processed_modulation=proc_mod
    )
    assert cross_corr.shape == (n_mod_filters,)
    assert np.sum(cross_corr) == pytest.approx(
        0.4414165879317118, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
