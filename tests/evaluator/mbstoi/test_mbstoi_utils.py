"""Tests for mbstoi_utils module"""
import numpy as np
import pytest

from clarity.evaluator.mbstoi.mbstoi_utils import (
    _firstpartfunc,
    _fourthpartfunc,
    _secondpartfunc,
    _thirdpartfunc,
    equalisation_cancellation,
    find_delay_impulse,
    remove_silent_frames,
    stft,
    thirdoct,
)


def test_equalisation_cancellation() -> None:
    """Test for equalisation_cancellation function"""

    np.random.seed(0)
    n_freq_channels = 5
    n_samples = 50
    n_frames = 30
    n_taus = 20
    n_gammas = 40
    centre_frequencies = np.array([940.0, 1190.0, 1500.0])

    left_ear_clean_hat = np.random.random(size=(n_freq_channels, n_samples))
    right_ear_clean_hat = np.random.random(size=(n_freq_channels, n_samples))
    left_ear_noisy_hat = np.random.random(size=(n_freq_channels, n_samples))
    right_ear_noisy_hat = np.random.random(size=(n_freq_channels, n_samples))
    frequency_band_edges_indices = np.array([[1, 2], [2, 1], [2, 1]])
    n_gamma_bands = len(centre_frequencies)
    taus = np.random.random(size=n_taus)
    gammas = np.random.random(size=n_gammas)
    intell_grid = np.random.random(size=(n_taus, n_gammas))
    p_ec_max = np.random.random(size=(n_gamma_bands, n_gammas))
    sigma_epsilon = np.random.random(size=n_gammas)
    sigma_delta = np.random.random(size=n_taus)

    intermediate_grid, p_ec_m = equalisation_cancellation(
        left_ear_clean_hat=left_ear_clean_hat,
        right_ear_clean_hat=right_ear_clean_hat,
        left_ear_noisy_hat=left_ear_noisy_hat,
        right_ear_noisy_hat=right_ear_noisy_hat,
        n_third_octave_bands=n_gamma_bands,
        n_frames=n_frames,
        frequency_band_edges_indices=frequency_band_edges_indices,
        centre_frequencies=centre_frequencies,
        taus=taus,
        n_taus=n_taus,
        gammas=gammas,
        n_gammas=n_gammas,
        intermediate_intelligibility_measure_grid=intell_grid,
        p_ec_max=p_ec_max,
        sigma_epsilon=sigma_epsilon,
        sigma_delta=sigma_delta,
    )

    # check shapes
    assert intermediate_grid.shape == (n_taus, n_gammas)
    assert p_ec_m.shape == (n_gamma_bands, n_gammas)

    # check values
    assert np.sum(intermediate_grid) == pytest.approx(
        277.84700307853785, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(p_ec_m) == pytest.approx(
        73.14293826967146, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_firstpartfunc() -> None:
    """Test for firstpartfunc function"""

    np.random.seed(0)
    n_taus = 100
    n_gammas = 40
    result = _firstpartfunc(
        L1=np.random.random(size=(1, 30)),
        L2=np.random.random(size=(1, 30)),
        R1=np.random.random(size=(1, 30)),
        R2=np.random.random(size=(1, 30)),
        n_taus=n_taus,
        gammas=np.random.random(size=(1, n_gammas)),
        epsexp=np.random.random(size=(1, n_gammas)),
    )
    assert result.shape == (n_taus, n_gammas)
    assert np.sum(result) == pytest.approx(
        405590.0256254587, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_secondpartfunc() -> None:
    """Test for secndpartfunc function"""

    np.random.seed(0)
    n_taus = 100
    n_gammas = 40

    result = _secondpartfunc(
        L1=np.random.random(size=(1, 30)),
        L2=np.random.random(size=(1, 30)),
        rho1=np.random.random(size=(1, 30)),
        rho2=np.random.random(size=(1, 30)),
        tauexp=np.random.random(size=(1, n_taus)),
        epsdelexp=np.random.random(size=(n_taus, n_gammas)),
        gammas=np.random.random(size=(1, n_gammas)),
    )
    assert result.shape == (n_taus, n_gammas)
    assert np.sum(result) == pytest.approx(
        127394.91897484378, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_thirdpartfunc() -> None:
    """Test for thirdpartfunc function"""

    np.random.seed(0)
    n_taus = 100
    n_gammas = 40

    result = _thirdpartfunc(
        R1=np.random.random(size=(1, 30)),
        R2=np.random.random(size=(1, 30)),
        rho1=np.random.random(size=(1, 30)),
        rho2=np.random.random(size=(1, 30)),
        tauexp=np.random.random(size=(1, n_taus)),
        epsdelexp=np.random.random(size=(n_taus, n_gammas)),
        gammas=np.random.random(size=(1, n_gammas)),
    )
    assert result.shape == (n_taus, n_gammas)
    assert np.sum(result) == pytest.approx(
        9483.482148264484, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_fourthpartfunc() -> None:
    """Test for fourthpartfunc function"""

    np.random.seed(0)
    n_taus = 100
    n_gammas = 40

    result = _fourthpartfunc(
        rho1=np.random.random(size=(1, 30)),
        rho2=np.random.random(size=(1, 30)),
        tauexp2=np.random.random(size=(1, n_taus)),
        n_gammas=n_gammas,
        deltexp=np.random.random(size=(1, n_taus)),
    )
    assert result.shape == (n_taus, n_gammas)
    assert np.sum(result) == pytest.approx(
        75883.96658627654, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_stft() -> None:
    """Test for stft function"""
    np.random.seed(0)

    fft_size = 32
    signal = np.random.random(size=600)
    stft_out = stft(signal, win_size=128, fft_size=fft_size)

    assert stft_out.shape == (8, fft_size)
    assert np.sum(stft_out) == pytest.approx(
        0.07977179832430759, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_remove_silent_frames() -> None:
    """Test for remove_silent_frames function"""

    np.random.seed(0)
    n_frames = 600

    left_ear_clean = np.random.random(n_frames)
    right_ear_clean = np.random.random(n_frames)
    left_ear_clean[0:300] = 0
    left_ear_noisy = left_ear_clean + np.random.random(n_frames)
    right_ear_noisy = right_ear_clean + np.random.random(n_frames)

    (xl_sil, xr_sil, yl_sil, yr_sil) = remove_silent_frames(
        left_ear_clean=left_ear_clean,
        right_ear_clean=right_ear_clean,
        left_ear_noisy=left_ear_noisy,
        right_ear_noisy=right_ear_noisy,
        dynamic_range=40,
        frame_length=64,
        hop=32,
    )

    assert np.sum(xl_sil[0:300]) == pytest.approx(
        0.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(xl_sil) == pytest.approx(
        128.15949932062742, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(xr_sil) == pytest.approx(
        272.06229083012374, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(yl_sil) == pytest.approx(
        413.65823132789836, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(yr_sil) == pytest.approx(
        551.1857227061379, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_thirdoct() -> None:
    """Test for thirdoct function"""
    (
        octave_band_matrix,
        centre_frequencies,
        frequency_band_edges_indices,
        freq_low,
        freq_high,
    ) = thirdoct(sample_rate=8000, nfft=512, num_bands=3, min_freq=100)

    assert freq_low == pytest.approx(
        [93.75, 109.375, 140.625], rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert freq_high == pytest.approx(
        [109.375, 140.625, 171.875], rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert centre_frequencies == pytest.approx(
        np.array([[100.0, 125.99210499, 158.7401052]]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )
    assert octave_band_matrix.shape == (3, 257)
    assert np.sum(octave_band_matrix) == pytest.approx(
        5.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert frequency_band_edges_indices == pytest.approx(
        np.array([[7, 7], [8, 9], [10, 11]]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


@pytest.mark.parametrize(
    "lag, n_samples, initial_value",
    [
        (20, 1000, 0),
        (20, 1000, 10),
        (10, 500, 0),
        (10, 500, 10),
        (10, 500, -10),
        (0, 1000, 10),
    ],
)
def test_find_delay_impulse(lag, n_samples, initial_value) -> None:
    """Test for find_delay_impulse function with lag applied in both directions"""
    np.random.seed(0)

    # Make channel 0 lagged relative to channel 1 and check lag is recovered
    ddf = np.random.random(size=(n_samples, 4))
    if lag != 0:
        ddf[:-lag, 1] = ddf[lag:, 0]
    else:
        ddf[:, 1] = ddf[:, 0]
    delay = find_delay_impulse(ddf=ddf, initial_value=initial_value)

    assert delay.shape == (2, 1)
    assert delay[0, 0] - delay[1, 0] == lag

    # Make channel 1 lagged relative to channel 0 and check lag is recovered
    ddf = np.random.random(size=(n_samples, 4))
    if lag != 0:
        ddf[:-lag, 0] = ddf[lag:, 1]
    else:
        ddf[:, 0] = ddf[:, 1]
    delay = find_delay_impulse(ddf=ddf, initial_value=initial_value)

    assert delay.shape == (2, 1)
    assert delay[1, 0] - delay[0, 0] == lag
