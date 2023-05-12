"""Module for testing torch_stoi.py"""
import numpy as np
import pytest
import torch
from pystoi.stoi import FS
from torch.testing import assert_close

from clarity.predictor.torch_stoi import (
    NegSTOILoss,
    masked_mean,
    masked_norm,
    meanvar_norm,
)


@pytest.fixture
def use_torch():
    """Fixture to ensure torch is used"""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_printoptions(precision=10)


def test_negstoi_loss_init(use_torch):
    """test negstoi loss constructed with various switch settings"""
    model = NegSTOILoss(sample_rate=FS * 2)
    assert model.sample_rate == FS * 2
    model = NegSTOILoss(sample_rate=FS * 2, use_vad=False)
    assert model.sample_rate == FS * 2
    model = NegSTOILoss(sample_rate=FS * 2, extended=True)
    assert model.sample_rate == FS * 2
    model = NegSTOILoss(sample_rate=FS * 2, do_resample=True)
    assert model.sample_rate == FS * 2


@pytest.mark.parametrize(
    "seed1, seed2, use_vad, extended, expected",
    [
        (0, 0, True, False, -1.0),  # <- score -1 when targets and estimates are equal
        (0, 0, False, False, -1.0),
        (0, 0, False, True, -1.0),
        (0, 1, True, False, -0.0182733461),
        (0, 1, False, False, -0.0182733461),
        (0, 1, False, True, -0.0279680751),
    ],
)
def test_negstoi_loss_forward(use_torch, seed1, seed2, use_vad, extended, expected):
    """test negstoi loss forward"""
    torch.manual_seed(0)
    model = NegSTOILoss(sample_rate=16000, use_vad=use_vad, extended=extended)
    torch.manual_seed(seed1)
    estimated_targets = torch.rand(1, 16000)
    torch.manual_seed(seed2)
    targets = torch.rand(1, 16000)
    loss = model(estimated_targets, targets)
    assert loss.shape == (1,)
    assert_close(loss, torch.tensor([expected]))


def test_negstoi_loss_forward_2d(use_torch):
    """test negstoi loss forward with 2d targets"""
    torch.manual_seed(0)
    model = NegSTOILoss(sample_rate=16000)
    estimated_targets = torch.rand(2, 16000)
    targets = torch.rand(2, 16000)
    loss = model(estimated_targets, targets)
    assert loss.shape == torch.Size([2])
    assert_close(loss, torch.tensor([0.0437693633, 0.0397059321]))


def test_negstoi_loss_detect_silent_frame(use_torch):
    """test negstoi loss detect silent frame"""
    torch.manual_seed(0)
    signal = torch.rand(1, 16000)
    dyn_range = 1.0
    framelen = 64
    hop = 32
    mask = NegSTOILoss.detect_silent_frames(
        x=signal,
        dyn_range=dyn_range,
        framelen=framelen,
        hop=hop,
    )
    assert mask.shape == torch.Size([1, 1, 498])
    assert mask.cpu().detach().numpy().sum() == 155


def test_negstoi_loss_stft(use_torch):
    """test negstoi loss stft"""
    torch.manual_seed(0)
    fft_size = 16
    win = torch.hann_window(fft_size)
    signal = torch.rand(1, 16000)
    stft = NegSTOILoss.stft(x=signal, win=win, fft_size=fft_size, overlap=4)
    assert stft.shape == torch.Size([1, 9, 3998, 2])
    assert stft.cpu().detach().numpy().sum() == pytest.approx(
        8025.33251953125, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_negstoi_loss_rowcolnorn(use_torch):
    """test negstoi loss rowcolnorn"""
    torch.manual_seed(0)
    x_size = [3, 100, 100]
    x = torch.rand(x_size)
    y = NegSTOILoss.rowcol_norm(x)
    assert y.shape == torch.Size(x_size)
    y_array = y.cpu().detach().numpy()
    assert np.sum(np.abs(y_array)) == pytest.approx(
        2589.76416015625, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_meanvar_norm(use_torch):
    """test meanvar norm"""
    torch.manual_seed(0)
    x = torch.rand(1, 100)
    y = meanvar_norm(x)
    assert_close(y.abs().sum(), torch.tensor(8.8223075867))
    # Masking values should change the result
    y_mask = meanvar_norm(x, mask=x > 0.5)
    assert_close(y_mask.abs().sum(), torch.tensor(32.58656692504883))
    # Mask above some very low value, i.e use all x, so same as no mask
    y_all = meanvar_norm(x, mask=x > -500)
    assert_close(y_all, y)
    # Mask above some high low value, i.e use no value, gives very large number
    # from division by (zero + EPS)
    y_none = meanvar_norm(x, mask=x > 500)
    assert_close(y_none.abs().sum(), torch.tensor(4884195328.0))


def test_masked_mean(use_torch):
    """test masked mean"""
    torch.manual_seed(0)
    x = torch.rand(1, 100)
    y = masked_mean(x)
    assert_close(y.abs().sum(), torch.tensor(0.4884195327758789))
    # Masking values should change the result
    y_mask = masked_mean(x, mask=x > 0.5)
    assert_close(y_mask.abs().sum(), torch.tensor(0.7467631697654724))
    # Mask above some very low value, i.e use all x, so same as no mask
    y_all = masked_mean(x, mask=x > -500)
    assert_close(y_all, y)
    # Mask above some high low value, i.e use no value, should give 0
    y_none = masked_mean(x, mask=x > 500)
    assert_close(y_none.abs().sum(), torch.tensor(0.0))


def test_masked_norm(use_torch):
    """test masked norm"""
    torch.manual_seed(0)
    x = torch.rand(1, 100)
    y = masked_norm(x)
    assert_close(y.abs().sum(), torch.tensor(5.6958699226379395))
    # Masking values should change the result
    y_mask = masked_norm(x, mask=x > 0.5)
    assert_close(y_mask.abs().sum(), torch.tensor(5.369431972503662))
    # Mask above some very low value, i.e use all x, so same as no mask
    y_all = masked_norm(x, mask=x > -500)
    assert_close(y_all, y)
    # Mask above some high low value, i.e use no value, should give 0
    y_none = masked_norm(x, mask=x > 500)
    assert_close(y_none.abs().sum(), torch.tensor(0.0))


def test_masked_mean_keep_dims(use_torch):
    """test masked norm"""
    torch.manual_seed(0)
    x = torch.rand(2, 100, 100)
    # with keepdim=False the final dimension is dropped...
    y1 = masked_mean(x, keepdim=False)
    assert y1.shape == torch.Size([2, 100])
    # ... with True it is kept but will have a size of 1
    y2 = masked_mean(x, keepdim=True)
    assert y2.shape == torch.Size([2, 100, 1])
    # ... shapes different but the sums should be the same
    assert_close(y1.abs().sum(), y2.abs().sum())
    # ... and the values should be the same
    assert_close(y1, y2.squeeze(-1))
