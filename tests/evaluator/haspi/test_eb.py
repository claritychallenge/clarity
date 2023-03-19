"""Tests for eb module"""

import numpy as np
import pytest

from clarity.evaluator.haspi.eb import (
    ave_covary2,
    bandwidth_adjust,
    basilar_membrane_add_noise,
    bm_covary,
    center_frequency,
    convert_rms_to_sl,
    ear_model,
    env_compress_basilar_membrane,
    env_smooth,
    envelope_align,
    envelope_sl,
    gammatone_bandwidth_demodulation,
    gammatone_basilar_membrane,
    group_delay_compensate,
    inner_hair_cell_adaptation,
    input_align,
    loss_parameters,
    mel_cepstrum_correlation,
    melcor9,
    melcor9_crosscovmatrix,
    middle_ear,
    resample_24khz,
    spectrum_diff,
)

# ear_model


def test_ear_model():
    """Test ear model"""
    np.random.seed(0)
    sig_len = 600
    samp_freq = 24000
    out_sig_len = sig_len * 24000 / samp_freq

    ref = np.random.random(size=sig_len)
    proc = np.random.random(size=sig_len)
    ref_db, ref_bm, proc_db, proc_bm, ref_sl, proc_sl, freq_sample = ear_model(
        reference=ref,
        reference_freq=samp_freq,
        processed=ref + proc,
        processed_freq=samp_freq,
        hearing_loss=np.array([45, 45, 35, 45, 60, 65]),
        itype=0,
        level1=65,
        nchan=10,
        m_delay=1,
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
    assert np.sum(np.abs(ref_db)) == pytest.approx(102596.63767028379)
    assert np.sum(np.abs(proc_db)) == pytest.approx(4145.3884196835625)
    assert np.sum(np.abs(ref_bm)) == pytest.approx(65517.72934742906)
    assert np.sum(np.abs(proc_bm)) == pytest.approx(2366.401656815131)
    assert np.sum(np.abs(ref_sl)) == pytest.approx(291.3527365691821)
    assert np.sum(np.abs(proc_sl)) == pytest.approx(13.655317152968216)


def test_center_frequency():
    """Test center frequency"""

    center_freq = center_frequency(
        nchan=10,
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )
    assert center_freq.shape == (10,)
    assert np.sum(center_freq) == pytest.approx(23935.19626226296)


def test_loss_parameters():
    """Test loss parameters"""

    (
        attenuated_ohc,
        bandwith,
        low_knee,
        compression_ratio,
        annenuated_ihc,
    ) = loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
        audiometric_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
    )

    # check shapes
    assert attenuated_ohc.shape == (6,)
    assert bandwith.shape == (6,)
    assert low_knee.shape == (6,)
    assert compression_ratio.shape == (6,)
    assert annenuated_ihc.shape == (6,)
    # check values
    assert np.sum(attenuated_ohc) == pytest.approx(220.39149328167292)
    assert np.sum(bandwith) == pytest.approx(15.041134665207498)
    assert np.sum(low_knee) == pytest.approx(400.3914932816729)
    assert np.sum(compression_ratio) == pytest.approx(6.0)
    assert np.sum(annenuated_ihc) == pytest.approx(129.6085067183270)


def test_resample():
    """Test resample"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = np.random.random(size=sig_len)
    reference_freq = 12000
    ref_signal_24, freq_sample_hz = resample_24khz(
        reference_signal, reference_freq, freq_sample_hz=24000
    )

    # check shapes
    assert len(ref_signal_24) == len(reference_signal) * 24000 / reference_freq
    # check values
    assert np.sum(np.abs(ref_signal_24)) == pytest.approx(604.1522707137393)
    assert freq_sample_hz == 24000


def test_input_align():
    """Test input align"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    ref, proc = input_align(reference_signal, processed_signal)

    assert ref.shape == (600,)
    assert proc.shape == (600,)
    assert np.sum(np.abs(ref)) == pytest.approx(29892.167176853407)
    assert np.sum(np.abs(proc)) == pytest.approx(27199.009291096496)


def test_middle_ear():
    """Test middle ear"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    filtered_signal = middle_ear(reference_signal, 24000)

    assert filtered_signal.shape == (600,)
    assert np.sum(np.abs(filtered_signal)) == pytest.approx(9241.220369749171)


def test_gammatone_basilar_membrane():
    """Test gammatone basilar membrane"""
    np.random.seed(0)
    sig_len = 600
    ref = 100 * np.random.random(size=sig_len)
    proc = ref + 10 * np.random.random(size=sig_len)

    (
        reference_envelope,
        reference_basilar_membrane,
        processed_envelope,
        processed_basilar_membrane,
    ) = gammatone_basilar_membrane(
        reference=ref,
        reference_bandwidth=1.4,
        processed=proc,
        processed_bandwidth=2.0,
        freq_sample=24000,
        center_freq=1000,
        ear_q=9.26449,
        min_bandwidth=24.7,
    )

    # check shapes
    assert reference_envelope.shape == (600,)
    assert reference_basilar_membrane.shape == (600,)
    assert processed_envelope.shape == (600,)
    assert processed_basilar_membrane.shape == (600,)
    # check values
    assert np.sum(np.abs(reference_envelope)) == pytest.approx(3605.427313705984)
    assert np.sum(np.abs(reference_basilar_membrane)) == pytest.approx(2288.3557465)
    assert np.sum(np.abs(processed_envelope)) == pytest.approx(4426.111706599469)
    assert np.sum(np.abs(processed_basilar_membrane)) == pytest.approx(2804.93743475)


def test_gammatone_bandwidth_demodulation():
    """Test gammatone bandwidth demodulation"""
    centre_freq_sin, centre_freq_cos = gammatone_bandwidth_demodulation(
        npts=100,
        tpt=0.001,
        center_freq=1000,
        center_freq_cos=np.zeros(100),
        center_freq_sin=np.zeros(100),
    )
    assert centre_freq_sin.shape == (100,)
    assert centre_freq_cos.shape == (100,)
    assert np.sum(centre_freq_sin) == pytest.approx(-0.3791946274493412)
    assert np.sum(centre_freq_cos) == pytest.approx(-0.39460748051808026)


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
    bw_adjusted = bandwidth_adjust(
        control=scale * np.array([1, -1, 1]),
        bandwidth_min=bw_min,
        bandwidth_max=bw_max,
        level1=1,
    )
    assert bw_adjusted == pytest.approx(expected)


def test_env_compress_basilar_membrane():
    """Test env_compress_basilar_membrane"""
    np.random.seed(0)
    sig_len = 600
    env_sig = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001
    control = np.random.random(size=sig_len)
    compressed_signal, compressed_basilar_membrane = env_compress_basilar_membrane(
        env_sig,
        bm,  # pylint: disable=invalid-name
        control,
        attn_ohc=0.01,
        threshold_low=70.0,
        compression_ratio=0.1,
        fsamp=24000,
        level1=140,
        small=1e-30,
        threshold_high=120,
    )
    # check shapes
    assert compressed_signal.shape == (600,)
    assert compressed_basilar_membrane.shape == (600,)
    # check values
    assert np.mean(np.abs(compressed_signal)) == pytest.approx(15486012153068.807)
    assert np.mean(np.abs(compressed_basilar_membrane)) == pytest.approx(
        15415471156.59357
    )


# envelope_align
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

    aligned_output = envelope_align(
        reference, output, freq_sample=24000, corr_range=100
    )

    # check shapes and values
    assert aligned_output.shape == (600,)
    assert np.sum(np.abs(aligned_output)) == pytest.approx(299.1891022020615)
    # Check output is now aligned with the reference
    assert aligned_output[100] == pytest.approx(scale * reference[100])


# envelope_sl
def test_envelope_sl():
    """Test envelope sl"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001

    reference, basilar_membrane = envelope_sl(
        reference=ref,
        basilar_membrane=bm,
        attnenuated_ihc=40.0,
        level1=120,
        small=1e-30,
    )

    assert reference.shape == (600,)
    assert basilar_membrane.shape == (600,)
    assert np.sum(np.abs(reference)) == pytest.approx(42746.12859151134)
    assert np.sum(np.abs(basilar_membrane)) == pytest.approx(98.97646233693762)


# inner_hair_cell_adaptation
def test_inner_hair_cell_adaptation():
    """Test inner hair cell adaptation"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001

    output_db, output_basilar_membrane = inner_hair_cell_adaptation(
        reference_db=ref, reference_basilar_membrane=bm, delta=1.00, freq_sample=24000
    )

    # check shapes and values
    assert output_db.shape == (600,)
    assert output_basilar_membrane.shape == (600,)
    assert np.sum(np.abs(output_db)) == pytest.approx(298.9359292744365)
    assert np.sum(np.abs(output_basilar_membrane)) == pytest.approx(0.2963082865723811)


# basilar_membrane_add_noise
def test_basilar_membrane_add_noise():
    """Test basilar membrane add noise"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)

    noisy_reference = basilar_membrane_add_noise(
        reference=ref, threshold=40, level1=120
    )

    # check shapes and values and that signal has changed
    assert noisy_reference.shape == (600,)
    assert np.sum(np.abs(noisy_reference)) == pytest.approx(298.919051930547)
    assert not np.allclose(noisy_reference, ref)

    # Check that adding on nearly 0 noise (-100 db) doesn't change the signal
    noisy_reference = basilar_membrane_add_noise(
        reference=ref, threshold=-100, level1=120
    )
    assert np.allclose(noisy_reference, ref)


# group_delay_compensate
def test_group_delay_compensate():
    """Test group delay compensate"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))

    processed = group_delay_compensate(
        reference=reference,
        bandwidths=np.array([30, 60, 90, 120]),
        center_freq=np.array([100, 200, 300, 400]),
        freq_sample=24000,
        ear_q=9.26449,
        min_bandwidth=24.7,
    )

    # check shapes and values
    assert processed.shape == (4, 600)
    assert np.sum(np.abs(processed)) == pytest.approx(1193.8088344682358)


def test_convert_rms_to_sl():
    """Test convert rms to sl"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=sig_len)
    control = np.random.random(size=sig_len)

    ref_db = convert_rms_to_sl(
        reference=reference,
        control=control,
        attnenuated_ohc=0.1,
        threshold_low=40,
        compression_ratio=10,
        attnenuated_ihc=0.1,
        level1=120,
        threshold_high=100,
        small=1e-30,
    )

    # check shapes and values
    assert ref_db.shape == (600,)
    assert np.sum(np.abs(ref_db)) == pytest.approx(34746.74406262155)


def test_env_smooth():
    """Test env smooth"""

    np.random.seed(0)
    sig_len = 600
    segment_size = 1  # ms
    envelopes = np.random.random(size=(4, sig_len))
    sample_freq = 24000

    smooth = env_smooth(
        envelopes=envelopes, segment_size=segment_size, freq_sample=sample_freq
    )

    # check shapes and values
    expected_length = 2 * sig_len / (sample_freq * segment_size / 1000)
    assert smooth.shape == (4, expected_length)
    assert np.sum(np.abs(smooth)) == pytest.approx(100.7397658862719)


def test_mel_cepstrum_correlation():
    """Test mel cepstrum correlation"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))
    distorted = np.random.random(size=(4, sig_len))
    addnoise = 0.0001

    # self correlation should produce scores of 1.0
    (
        ave_cepstral_correlation,
        individual_cepstral_correlation,
    ) = mel_cepstrum_correlation(
        reference=reference, distorted=reference, threshold=-20, addnoise=addnoise
    )

    # check shapes and values
    assert ave_cepstral_correlation == pytest.approx(1.0)
    assert individual_cepstral_correlation == pytest.approx(1.0)

    # correlation between two random signals should be low
    (
        ave_cepstral_correlation,
        individual_cepstral_correlation,
    ) = mel_cepstrum_correlation(
        reference=reference, distorted=distorted, threshold=-20, addnoise=addnoise
    )

    # check shapes and values
    assert np.array(individual_cepstral_correlation).shape == (6,)
    assert ave_cepstral_correlation == pytest.approx(0.04195483905166838)
    assert individual_cepstral_correlation == pytest.approx(
        np.array(
            [0.04582154, 0.04306849, 0.02364514, 0.07634693, 0.02364514, 0.04306849]
        )
    )


def test_melcor9():
    """Test melcor9"""

    np.random.seed(0)
    sig_len = 6000
    reference = 100 * np.random.random(size=sig_len)
    _distorted = 100 * np.random.random(size=sig_len)  # noqa: F841

    # TODO: This is always returning 0's :-()
    mel_cep_ave, mel_cep_low, mel_cep_high, mel_cep_mod = melcor9(
        reference=reference,
        distorted=reference,
        threshold=20,
        add_noise=0.00,
        segment_size=2,  # ms
        n_cepstral_coef=6,
    )

    assert mel_cep_ave == pytest.approx(0)
    assert mel_cep_low == pytest.approx(0)
    assert mel_cep_high == pytest.approx(0)
    assert mel_cep_mod == pytest.approx(0)


def test_melcor9_crosscovmatrix():
    """Test melcor9 crosscovmatrix"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(6, sig_len))
    processed = np.random.random(size=(6, sig_len))
    bm = np.random.random(size=sig_len)

    # TODO: undocumented function - unsure what inputs should look like
    # Need to find correct inputs so that it doesn't throw an error
    with pytest.raises(ValueError):
        _cross_cov_matrix = melcor9_crosscovmatrix(  # noqa: F841
            b=bm,
            nmod=10,  # n modulation channels
            nbasis=6,  # n cepstral coefficients
            nsamp=600,
            nfir=128,
            reference_cep=reference,
            processed_cep=processed,
        )


def test_spectrum_diff():
    """Test spectrum diff"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    dloud, dnorm, dslope = spectrum_diff(reference_sl=reference, processed_sl=processed)

    # Check shapes
    assert dloud.shape == (3,)
    assert dnorm.shape == (3,)
    assert dslope.shape == (3,)

    # Check values
    assert dloud == pytest.approx(
        np.array([0.037373053253378426, 7.671407491067586e-05, 4.762048775653672e-05])
    )
    assert dnorm == pytest.approx(
        np.array([44.84779838405483, 0.09203844008264382, 0.0570982338215042])
    )
    assert dslope == pytest.approx(
        np.array([0.03918063966807504, 0.00010878329523853699, 8.006096597686129e-05])
    )


def test_bm_covary():
    """Test bm covary"""

    np.random.seed(0)
    sig_len = 600
    sample_freq = 24000
    segment_size = 1
    reference = np.random.random(size=(4, sig_len))
    processed = np.random.random(size=(4, sig_len))

    signal_cross_cov, ref_mean_square, proc_mean_square = bm_covary(
        reference_basilar_membrane=reference,
        processed_basilar_membrane=reference + 0.1 * processed,
        segment_size=segment_size,
        freq_sample=sample_freq,
    )

    # Check shapes
    expected_length = 2 * sig_len / (sample_freq * segment_size / 1000)

    assert signal_cross_cov.shape == (4, expected_length)
    assert ref_mean_square.shape == (4, expected_length)
    assert proc_mean_square.shape == (4, expected_length)

    # Check values
    assert np.sum(np.abs(signal_cross_cov)) == pytest.approx(200)
    assert np.sum(np.abs(ref_mean_square)) == pytest.approx(70.1701757426345)
    assert np.sum(np.abs(proc_mean_square)) == pytest.approx(78.38279741287329)


def test_ave_covary2():
    """Test ave covary2"""
    np.random.seed(0)
    sig_len = 600

    signal_cross_cov = np.random.random(size=(4, sig_len))
    ref_mean_square = np.random.random(size=(4, sig_len))

    ave_covariance, ihc_sync_covariance = ave_covary2(
        signal_cross_covariance=signal_cross_cov,
        reference_signal_mean_square=ref_mean_square,
        threshold_db=20,
        lp_filter_order=np.array([1, 3, 5, 5, 5, 5]),
        freq_cutoff=1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    )

    # TODO: outputting all 0's -- need better input

    assert len(ihc_sync_covariance) == 6

    assert ave_covariance == pytest.approx(0)
    assert ihc_sync_covariance == pytest.approx([0, 0, 0, 0, 0, 0])
