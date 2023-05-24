"""Module for testing torch_msbg.py"""
import numpy as np
import pytest
import torch

from clarity.predictor.torch_msbg import MSBGHearingModel, torchloudnorm
from clarity.utils.audiogram import (
    AUDIOGRAM_MILD,
    AUDIOGRAM_MODERATE,
    AUDIOGRAM_MODERATE_SEVERE,
)


@pytest.fixture
def use_torch():
    """Fixture to ensure torch is used"""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    torch.set_default_tensor_type(torch.FloatTensor)


@pytest.fixture
def msbg_model():
    """MSBG model fixture"""
    model = MSBGHearingModel(
        audiogram=AUDIOGRAM_MODERATE.levels.tolist(),
        audiometric=AUDIOGRAM_MODERATE.frequencies.tolist(),
        device="cpu",
    )
    return model


@pytest.fixture
def msbg_model_quick():
    """MSBG model fixture with smaller kernel for quick testing"""
    model = MSBGHearingModel(
        audiogram=AUDIOGRAM_MODERATE.levels.tolist(),
        audiometric=AUDIOGRAM_MODERATE.frequencies.tolist(),
        kernel_size=129,
        device="cpu",
    )
    return model


# Tests for MSBGHearingModel class


def test_msbg_hearing_model_init(use_torch):
    """Test the MSBGHearingModel class init function"""
    model = MSBGHearingModel(
        audiogram=AUDIOGRAM_MILD.levels.tolist(),
        audiometric=AUDIOGRAM_MILD.frequencies.tolist(),
        device="cpu",
    )
    assert model.win_len == 441
    model = MSBGHearingModel(
        audiogram=AUDIOGRAM_MODERATE.levels.tolist(),
        audiometric=AUDIOGRAM_MODERATE.frequencies.tolist(),
        device="cpu",
    )
    assert model.win_len == 441
    model = MSBGHearingModel(
        audiogram=AUDIOGRAM_MODERATE_SEVERE.levels.tolist(),
        audiometric=AUDIOGRAM_MODERATE_SEVERE.frequencies.tolist(),
        device="cpu",
    )
    assert model.win_len == 441
    model = MSBGHearingModel(
        audiogram=AUDIOGRAM_MODERATE.levels.tolist(),
        audiometric=AUDIOGRAM_MODERATE.frequencies.tolist(),
        device="cpu",
    )
    assert model.win_len == 441


def test_msbg_hearing_model_measure_rms(use_torch, msbg_model):
    """Test the measure_rms function"""
    x = torch.randn(2, 20000)
    x = x.cpu()
    y_torch = msbg_model.measure_rms(x)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 1)
    assert y[0, 0] == pytest.approx(
        1.0040439, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert y[1, 0] == pytest.approx(
        0.99920964, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_msbg_hearing_model_calibrate_spl(use_torch, msbg_model):
    """Test the calibrate_spl function"""
    x = torch.randn(2, 20000)
    x = x.cpu()
    y_torch = msbg_model.calibrate_spl(x)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(
        32026.688, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_msbg_hearing_model_calibrate_spl_null(use_torch):
    """Test the calibrate_spl function does nothing is calibration is disabled"""
    this_msbg_model = MSBGHearingModel(
        audiogram=AUDIOGRAM_MODERATE.levels.tolist(),
        audiometric=AUDIOGRAM_MODERATE.frequencies.tolist(),
        spl_cali=False,  # <--- Disable calibration
    )
    x = torch.randn(2, 20000)
    x = x.cpu()
    initial_sum = np.sum(np.abs(x.detach().numpy()))
    y_torch = this_msbg_model.calibrate_spl(x)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(
        initial_sum, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )  # <--- No change


def test_msbg_hearing_model_src_to_cochlea_filt(use_torch, msbg_model):
    """Test the src_to_cochlea_filt function"""
    x = torch.randn(1, 20000)
    x = x.cpu()
    y_torch = msbg_model.src_to_cochlea_filt(x, msbg_model.cochlea_filter_forward)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (1, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(
        14804.21875, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_msbg_hearing_model_smear(use_torch, msbg_model):
    """Test the smear function"""
    x = torch.randn(2, 1, 20000)
    x = x.cpu()
    y_torch = msbg_model.smear(x)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 1, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(
        29985.154, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_msbg_hearing_model_recruitment(use_torch, msbg_model_quick):
    """Test the recruitment function"""
    x = torch.randn(2, 1, 10000)
    x = x.cpu()
    y_torch = msbg_model_quick.recruitment(x)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 1, 10000)
    assert np.sum(np.abs(y)) == pytest.approx(
        13046.6328125, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_msbg_hearing_model_recruitment_fir(use_torch, msbg_model_quick):
    """Test the recruitment_fir function"""
    x = torch.randn(1, 10000)
    x = x.cpu()
    y_torch = msbg_model_quick.recruitment_fir(x)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (1, 1, 10000)
    assert np.sum(np.abs(y)) == pytest.approx(
        5229.41162109375, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_msbg_hearing_model_forward(use_torch, msbg_model_quick):
    """Test the forward function"""
    x = torch.randn(2, 10000)
    x = x.cpu()
    y_torch = msbg_model_quick.forward(x)
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 10000)
    # Tolerances below slightly relaxed from the 1e-7 default
    assert np.sum(np.abs(y)) == pytest.approx(8778.4501953125, rel=1e-6, abs=1e-6)


# Tests for torchloudnorm class


def test_torchloudnorm_apply_filter(use_torch):
    """Test torchloudnorm apply filter function"""

    x = torch.randn(2, 1, 40000)
    x = x.cpu()
    loud_norm = torchloudnorm(
        device="cpu",
    )

    y_torch = loud_norm.apply_filter(x)
    y = y_torch.cpu().detach().numpy()

    assert y.shape == (2, 1, 40000)
    assert np.sum(np.abs(y)) == pytest.approx(
        98578.4921875, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_torchloudnorm_integrated_loudness(use_torch):
    """Test torchloudnorm integrated loundness function"""

    x = torch.randn(2, 1, 20000)
    x = x.cpu()
    loud_norm = torchloudnorm(
        device="cpu",
    )

    y_torch = loud_norm.integrated_loudness(x)
    y = y_torch.cpu().detach().numpy()

    assert y.shape == (2, 1)
    assert np.sum(np.abs(y)) == pytest.approx(
        6.166536331176758, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_torchloudnorm_normalise_loudness(use_torch):
    """Test torchloudnorm normalise loudness function"""

    lufs = torch.FloatTensor([[-30.0], [-40.0]]).cpu()
    x = torch.randn(2, 20000)
    x = x.cpu()
    loud_norm = torchloudnorm(
        device="cpu",
    )
    y_torch = loud_norm.normalize_loudness(x, lufs=lufs)
    y = y_torch.cpu().detach().numpy()

    assert y.shape == (2, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(
        33388.645, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert True


def test_torchloudnorm_forward(use_torch):
    """Test torchloudnorm forward function"""

    x = torch.randn(2, 20000)
    x = x.cpu()
    loud_norm = torchloudnorm(
        device="cpu",
    )
    y_torch = loud_norm.forward(x)  # i.e. sane as `y_torch = loud_norm(x)`
    y = y_torch.cpu().detach().numpy()
    assert y.shape == (2, 20000)
    assert np.sum(np.abs(y)) == pytest.approx(
        355.77362060546875, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
