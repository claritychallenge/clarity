"""Tests for eb module"""

import numpy as np
import pytest

from clarity.evaluator.haspi.eb import (
    center_frequency,
    ear_model,
    loss_parameters,
    resample_24khz,
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


# resample_24khz
def test_resample():
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


# input_align

# middle_ear

# gammatone_bandwith_demodulation

# env_compress_basilar_membrane

# envelope_align

# envelope_sl

# inner_hair_cell_adaptation

# basilar_membrane_add_noise

# group_delay_compensate

# convert_rms_to_sl

# env_smooth

# mel_cepstrum_correlation

# melcor9

# melcor9_crosscovmatrix

# spectrum_diff

# bm_covary

# ave_covary2
