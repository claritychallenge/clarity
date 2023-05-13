"""Test ha_metrics.ear_model module"""
# pylint: disable=import-error
import numpy as np
import pytest

from clarity.evaluator.ha_metric.pyhaaqi import HAAQI, compute_haaqi


def test_env_smooth():
    """Test env smooth"""

    np.random.seed(0)
    sig_len = 600
    segment_size = 1  # ms
    envelopes = np.random.random(size=(4, sig_len))
    sample_rate = 24000

    haaqi = HAAQI(
        segment_size=segment_size,
        signal_sample_rate=24000.0,
        ear_model_sample_rate=24000.0,
    )
    smooth = haaqi.env_smooth(envelopes=envelopes)

    # check shapes and values
    expected_length = 2 * sig_len / (sample_rate * segment_size / 1000)
    assert smooth.shape == (4, expected_length)
    assert np.sum(np.abs(smooth)) == pytest.approx(
        100.7397658862719, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_melcor9_crosscovmatrix():
    """Test melcor9 crosscovmatrix"""

    np.random.seed(0)
    sig_len = 600
    n_modulations = 4
    n_basis = 6
    reference = np.random.random(size=(n_basis, sig_len))
    processed = np.random.random(size=(n_basis, sig_len))

    basilar_membrane = np.random.random(size=(n_modulations, sig_len))
    haaqi = HAAQI(
        segment_size=8,
        signal_sample_rate=24000.0,
        ear_model_sample_rate=24000.0,
    )
    cross_cov_matrix = haaqi.melcor9_crosscovmatrix(
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


def test_melcor9():
    """Test melcor9"""

    np.random.seed(0)
    sig_len = 6000
    reference = 20 * np.random.random(size=(4, sig_len))
    distorted = 20 * np.random.random(size=(4, sig_len))  # noqa: F841

    # TODO: This is always returning 0's :-()
    haaqi = HAAQI(
        segment_size=4,
        signal_sample_rate=24000.0,
        ear_model_sample_rate=24000.0,
        silence_threshold=12,
    )
    mel_cep_ave, mel_cep_low, mel_cep_high, mel_cep_mod = haaqi.melcor9(
        reference=reference,
        distorted=distorted,
        n_cepstral_coef=6,
    )

    assert mel_cep_mod.shape == (8,)

    assert mel_cep_ave == pytest.approx(
        0.07847183114015181, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_low == pytest.approx(
        0.08244466051783299, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_high == pytest.approx(
        0.07449900176247062, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert mel_cep_mod == pytest.approx(
        np.array(
            [
                0.08074834547870297,
                0.035224248042312015,
                0.10987800587454441,
                0.10392804267577263,
                0.12997273983987667,
                0.09738887779419407,
                0.06127608591547042,
                0.009358303500341338,
            ]
        ),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_melcor9_equal_input():
    """Test melcor9"""

    np.random.seed(0)
    sig_len = 6000
    reference = 20 * np.random.random(size=(4, sig_len))

    haaqi = HAAQI(
        segment_size=4,
        signal_sample_rate=24000.0,
        ear_model_sample_rate=24000.0,
        silence_threshold=12,
    )
    mel_cep_ave, mel_cep_low, mel_cep_high, mel_cep_mod = haaqi.melcor9(
        reference=reference,
        distorted=reference,
        n_cepstral_coef=6,
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


def test_spectrum_diff():
    """Test spectrum diff"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    haaqi = HAAQI(
        segment_size=4,
        signal_sample_rate=24000.0,
        ear_model_sample_rate=24000.0,
        silence_threshold=12,
    )
    dloud, dnorm, dslope = haaqi.spectrum_diff(
        reference_sl=reference, processed_sl=processed
    )

    # Check shapes
    assert dloud.shape == (3,)
    assert dnorm.shape == (3,)
    assert dslope.shape == (3,)

    # Check values
    assert dloud == pytest.approx(
        np.array([0.037373053253378426, 7.671407491067586e-05, 4.762048775653672e-05]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )
    assert dnorm == pytest.approx(
        np.array([44.84779838405483, 0.09203844008264382, 0.0570982338215042]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )
    assert dslope == pytest.approx(
        np.array([0.03918063966807504, 0.00010878329523853699, 8.006096597686129e-05]),
        rel=pytest.rel_tolerance,
        abs=pytest.abs_tolerance,
    )


def test_bm_covary_ok():
    """Test bm covary"""

    np.random.seed(0)
    sig_len = 600
    sample_rate = 24000
    segment_size = 4
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    haaqi = HAAQI(
        segment_size=segment_size,
        segment_covariance=segment_size,
        signal_sample_rate=sample_rate,
        ear_model_sample_rate=sample_rate,
    )
    signal_cross_cov, ref_mean_square, proc_mean_square = haaqi.bm_covary(
        reference_basilar_membrane=reference,
        processed_basilar_membrane=reference + 0.4 * processed,
    )

    # Check shapes
    assert signal_cross_cov.shape == (4, 12)
    assert ref_mean_square.shape == (4, 12)
    assert proc_mean_square.shape == (4, 12)

    # Check values
    assert np.sum(np.abs(signal_cross_cov)) == pytest.approx(
        46.43935095481214, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(ref_mean_square)) == pytest.approx(
        16.65809872708418, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(np.abs(proc_mean_square)) == pytest.approx(
        26.110255782462076, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_bm_covary_error():
    """Test bm covary fails when segment size too small"""

    np.random.seed(0)
    sig_len = 600
    sample_rate = 24000
    segment_size = 2  # needs to be a little over 2 ms
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    haaqi = HAAQI(
        segment_covariance=segment_size,
        signal_sample_rate=sample_rate,
        ear_model_sample_rate=sample_rate,
        silence_threshold=12,
    )
    with pytest.raises(ValueError):
        _signal_cross_cov, _ref_mean_square, _proc_mean_square = haaqi.bm_covary(
            reference_basilar_membrane=reference,
            processed_basilar_membrane=reference + 0.4 * processed,
        )


def test_ave_covary2():
    """Test ave covary2"""
    np.random.seed(0)
    sig_len = 600

    signal_cross_cov = np.random.random(size=(4, sig_len))
    ref_mean_square = np.random.random(size=(4, sig_len))

    haaqi = HAAQI(
        segment_size=4,
        signal_sample_rate=24000.0,
        ear_model_sample_rate=24000.0,
        silence_threshold=0.6,
        ear_model_kwards={"nchan": 4},
    )
    print(1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0]))
    ave_covariance, ihc_sync_covariance = haaqi.ave_covary2(
        signal_cross_covariance=signal_cross_cov,
        reference_signal_mean_square=ref_mean_square,
        lp_filter_order=np.array([1, 3, 5, 5, 5, 5]),
        freq_cutoff=np.array([1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0]),
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


def test_haaqi_v1() -> None:
    """Test for haaqi_v1 index"""
    np.random.seed(0)
    sample_rate = 16000.0
    x = np.random.uniform(-1, 1, int(sample_rate * 0.5))  # i.e. 500 ms of audio
    y = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    hearing_loss = np.array([45, 45, 35, 45, 60, 65])
    equalisation_mode = 1
    level1 = 65

    haaqi = HAAQI(
        signal_sample_rate=sample_rate,
        ear_model_sample_rate=24000.0,
        equalisation=equalisation_mode,
        level1=level1,
    )

    score, _, _, _ = haaqi.compute(reference=x, processed=y, hearing_loss=hearing_loss)

    assert score == pytest.approx(
        0.111290948, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "scale_reference,equalisation,expected_result",
    [(False, 1, 0.113759275), (True, 1, 0.114157435), (True, 0, 0.098472862)],
)
def test_compute_haaqi(scale_reference, equalisation, expected_result):
    """Test for compute_haaqi function"""
    np.random.seed(42)

    sample_rate = 16000
    enh_signal = np.random.uniform(-1, 1, int(sample_rate * 0.5))
    ref_signal = np.random.uniform(-1, 1, int(sample_rate * 0.5))

    audiogram = np.array([10, 20, 30, 40, 50, 60])
    audiogram_frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Compute HAAQI score
    score = compute_haaqi(
        processed_signal=enh_signal,
        reference_signal=ref_signal,
        audiogram=audiogram,
        audiogram_frequencies=audiogram_frequencies,
        sample_rate=sample_rate,
        scale_reference=scale_reference,
        equalisation=equalisation,
    )

    # Check that the score is a float between 0 and 1
    assert score == pytest.approx(
        expected_result, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
