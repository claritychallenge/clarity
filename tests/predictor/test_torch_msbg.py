"""Module for testing torch_msbg.py"""
import numpy as np
import pytest
import torch

from clarity.predictor.torch_msbg import (  # MSBGHearingModel,
    audfilt,
    makesmearmat3,
    torchloudnorm,
)

# *NB*: Test below identical to the teset in evaluator/msbg, i.e. this
# version of audfilt is behaving identically and can be replaced


def test_audfilt():
    """Test the auditory filter function"""
    sample_freq = 44100
    r_lower = 0.5
    r_upper = 1.5
    n_taps = 128
    filter_params = audfilt(rl=r_lower, ru=r_upper, sr=sample_freq, size=n_taps)
    assert filter_params.shape == (n_taps, n_taps)
    assert np.sum(np.abs(filter_params)) == pytest.approx(19.879915844855944)


# *NB*: Test below identical to the teset in evaluator/msbg, i.e. this
# version of makesmearmat is behaving identically and can be replaced


def test_make_smear_mat3_valid_input():
    """Tests that make_smear_mat3 returns matrix with the correct dimensions"""
    r_lower = 0.5
    r_upper = 1.5
    sample_freq = 44100
    f_smear = makesmearmat3(rl=r_lower, ru=r_upper, sr=sample_freq)
    assert f_smear.shape == (256, 256)
    assert np.sum(np.abs(f_smear)) == pytest.approx(2273.976168294156)


#  MSBGHearingModel (class)
#  MSBGHearingModel.measure_rms
#  MSBGHearingModel.calibrate_spl
#  MSBGHearingModel.smear
#  MSBGHearingModel.recruitment
#  MSBGHearingModel.recruitment_fir
#  MSBGHearingModel.forward


# torchloudnorm (class)
# torchloudnorm.apply_filter
# torchloudnorm.integrated_loudness
# torchloudnorm.normalise_loudness
# torchloudnorm.forward


def test_torchloudnorm_apply_filter():
    """Test torchloudnorm apply filter function"""
    torch.manual_seed(0)
    torch.set_num_threads(1)

    x = torch.randn(2, 1, 40000)
    x = x.cpu()
    loud_norm = torchloudnorm()

    y_torch = loud_norm.apply_filter(x)
    y = y_torch.cpu().detach().numpy()

    assert y.shape == (2, 1, 40000)
    assert np.sum(np.abs(y)) == pytest.approx(98578.4921875)


def test_torchloudnorm_integrated_loudness():
    """Test torchloudnorm integrated loundness function"""
    torch.manual_seed(0)
    torch.set_num_threads(1)

    x = torch.randn(2, 1, 20000)
    x = x.cpu()
    loud_norm = torchloudnorm()

    y_torch = loud_norm.integrated_loudness(x)
    y = y_torch.cpu().detach().numpy()

    assert y.shape == (2, 1)
    assert np.sum(np.abs(y)) == pytest.approx(6.166536331176758)


def test_torchloudnorm_normalise_loudness():
    """Test torchloudnorm normalise loudness function"""
    torch.manual_seed(0)
    torch.set_num_threads(1)

    lufs = torch.FloatTensor([[-30.0], [-40.0]]).cpu()
    x = torch.randn(2, 20000)
    x = x.cpu()
    loud_norm = torchloudnorm()
    y_torch = loud_norm.normalize_loudness(x, lufs=lufs)
    y = y_torch.cpu().detach().numpy()

    assert y.shape == (2, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(33388.645)
    assert True


def test_torchloudnorm_forward():
    """Test torchloudnorm forward function"""
    torch.manual_seed(0)
    torch.set_num_threads(1)

    x = torch.randn(2, 20000)
    x = x.cpu()
    loud_norm = torchloudnorm()
    y_torch = loud_norm.forward(x)  # i.e. sane as `y_torch = loud_norm(x)`
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(355.77362060546875)
