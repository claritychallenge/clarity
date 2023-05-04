import numpy as np

from clarity.evaluator.ha_metric.pyhaaqi import HAAQI
from clarity.evaluator.haspi.eb import (
    env_smooth,
    melcor9_crosscovmatrix,
    melcor9,
    spectrum_diff,
    bm_covary,
    ave_covary2,
)


def compare_env_smooth():
    """Test env smooth"""

    np.random.seed(0)
    sig_len = 600
    segment_size = 1  # ms
    envelopes = np.random.random(size=(4, sig_len))
    sample_rate = 24000

    eb_smooth = env_smooth(
        envelopes=envelopes, segment_size=segment_size, sample_rate=sample_rate
    )

    haaqi = HAAQI(segment_size=segment_size, sample_rate=sample_rate)
    haaqi_smooth = haaqi.env_smooth(envelopes=envelopes)
    print(f"Env Smooth {np.sum(np.abs(eb_smooth)) - np.sum(np.abs(haaqi_smooth))}")


def compare_melcor9_crosscovmatrix():
    """Test melcor9 crosscovmatrix"""

    np.random.seed(0)
    sig_len = 600
    n_modulations = 4
    n_basis = 6
    reference = np.random.random(size=(n_basis, sig_len))
    processed = np.random.random(size=(n_basis, sig_len))
    basilar_membrane = np.random.random(size=(n_modulations, sig_len))

    cross_cov_matrix = melcor9_crosscovmatrix(
        b=basilar_membrane,
        nmod=n_modulations,  # n modulation channels
        nbasis=n_basis,  # n cepstral coefficient
        nsamp=sig_len,
        nfir=32,
        reference_cep=reference,
        processed_cep=processed,
    )

    eb_cross_cov_matrix = np.mean(cross_cov_matrix)
    haaqi = HAAQI(segment_size=8, sample_rate=24000)
    haaqi_smooth = haaqi.melcor9_crosscovmatrix(
        b=basilar_membrane,
        nmod=n_modulations,  # n modulation channels
        nbasis=n_basis,  # n cepstral coefficient
        nsamp=sig_len,
        nfir=32,
        reference_cep=reference,
        processed_cep=processed,
    )

    print(
        f"Melcor9 Crosscovmatrix {np.sum(np.abs(eb_cross_cov_matrix)) - np.sum(np.abs(np.mean(haaqi_smooth)))}"
    )


def compare_melcor9():
    """Test melcor9"""

    np.random.seed(0)
    sig_len = 6000
    reference = 20 * np.random.random(size=(4, sig_len))
    distorted = 20 * np.random.random(size=(4, sig_len))  # noqa: F841

    # TODO: This is always returning 0's :-()
    mel_cep_ave, mel_cep_low, mel_cep_high, mel_cep_mod = melcor9(
        reference=reference,
        distorted=distorted,
        threshold=12,
        add_noise=0.00,
        segment_size=4,  # ms
        n_cepstral_coef=6,
    )

    haaqi = HAAQI(segment_size=4, sample_rate=24000, silence_threshold=12)
    (
        haaqi_mel_cep_ave,
        haaqi_mel_cep_low,
        haaqi_mel_cep_high,
        haaqi_mel_cep_mod,
    ) = haaqi.melcor9(
        reference=reference,
        distorted=distorted,
        n_cepstral_coef=6,
    )

    print(f"Melcor9 {mel_cep_ave - haaqi_mel_cep_ave}")
    print(f"Melcor9 {mel_cep_low - haaqi_mel_cep_low}")
    print(f"Melcor9 {mel_cep_high - haaqi_mel_cep_high}")
    print(f"Melcor9 {np.sum(np.abs(mel_cep_mod)) - np.sum(np.abs(haaqi_mel_cep_mod))}")


def compare_bm_covary_ok():
    """Test bm covary"""

    np.random.seed(0)
    sig_len = 600
    sample_rate = 24000
    segment_size = 4
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    signal_cross_cov, ref_mean_square, proc_mean_square = bm_covary(
        reference_basilar_membrane=reference,
        processed_basilar_membrane=reference + 0.4 * processed,
        segment_size=segment_size,
        sample_rate=sample_rate,
    )

    haaqi = HAAQI(
        segment_size=segment_size, sample_rate=sample_rate, silence_threshold=12
    )
    (
        haaqi_signal_cross_cov,
        haaqi_ref_mean_square,
        haaqi_proc_mean_square,
    ) = haaqi.bm_covary(
        reference_basilar_membrane=reference,
        processed_basilar_membrane=reference + 0.4 * processed,
    )

    print(
        f"BM Covary - cross_cov {np.sum(np.abs(signal_cross_cov)) - np.sum(np.abs(haaqi_signal_cross_cov))}"
    )
    print(
        f"BM Covary - ref_mean_square {np.sum(np.abs(ref_mean_square)) - np.sum(np.abs(haaqi_ref_mean_square))}"
    )
    print(
        f"BM Covary - proc_mean_square {np.sum(np.abs(proc_mean_square)) - np.sum(np.abs(haaqi_proc_mean_square))}"
    )


def compare_spectrum_diff():
    """Test spectrum diff"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    dloud, dnorm, dslope = spectrum_diff(reference_sl=reference, processed_sl=processed)

    # Check values
    haaqi = HAAQI(segment_size=4, sample_rate=24000, silence_threshold=12)
    haaqi_dloud, haaqi_dnorm, haaqi_dslope = haaqi.spectrum_diff(
        reference_sl=reference,
        processed_sl=processed,
    )

    print(f"Spectrum Diff {dloud - haaqi_dloud}")
    print(f"Spectrum Diff {dnorm - haaqi_dnorm}")
    print(f"Spectrum Diff {dslope - haaqi_dslope}")


def compare_ave_covary2():
    """Test ave covary2"""
    np.random.seed(0)
    sig_len = 600

    signal_cross_cov = np.random.random(size=(32, sig_len))
    ref_mean_square = np.random.random(size=(32, sig_len))

    ave_covariance, ihc_sync_covariance = ave_covary2(
        signal_cross_covariance=signal_cross_cov,
        reference_signal_mean_square=ref_mean_square,
        threshold_db=0.6,
        lp_filter_order=np.array([1, 3, 5, 5, 5, 5]),
        freq_cutoff=1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    )
    np.random.seed(0)
    sig_len = 600

    signal_cross_cov = np.random.random(size=(32, sig_len))
    ref_mean_square = np.random.random(size=(32, sig_len))
    haaqi = HAAQI(segment_size=4, sample_rate=24000, silence_threshold=0.6)
    haaqi_ave_covariance, haaqi_ihc_sync_covariance = haaqi.ave_covary2(
        signal_cross_covariance=signal_cross_cov,
        reference_signal_mean_square=ref_mean_square,
        lp_filter_order=np.array([1, 3, 5, 5, 5, 5]),
        freq_cutoff=1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    )

    print(
        f"Ave Covary2 {np.sum(np.abs(ave_covariance)) - np.sum(np.abs(haaqi_ave_covariance))}"
    )
    print(
        f"Ave Covary2 {np.sum(np.abs(ihc_sync_covariance)) - np.sum(np.abs(haaqi_ihc_sync_covariance))}"
    )


if __name__ == "__main__":
    compare_env_smooth()
    compare_melcor9_crosscovmatrix()
    compare_melcor9()
    compare_spectrum_diff()
    compare_bm_covary_ok()
    compare_ave_covary2()
