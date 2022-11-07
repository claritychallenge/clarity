# Regression test
import pytest
import torch

from clarity.predictor.torch_msbg import MSBGHearingModel, torchloudnorm
from clarity.predictor.torch_stoi import NegSTOILoss


def test_torch_msbg_stoi():

    torch.manual_seed(0)
    torch.set_num_threads(1)
    stoi_loss = NegSTOILoss(sample_rate=44100)
    estoi_loss = NegSTOILoss(sample_rate=44100, extended=True)

    audiogram = [45, 35, 30, 45, 50, 50]
    audiometric = [250, 500, 1000, 2000, 4000, 6000]
    msbg = MSBGHearingModel(audiogram=audiogram, audiometric=audiometric)

    x = torch.randn(2, 44100)
    if torch.cuda.is_available():
        x = x.cuda()
    y = msbg(x)
    stoi_loss = stoi_loss.forward(x.cpu(), y.cpu()).mean()
    estoi_loss = estoi_loss.forward(x.cpu(), y.cpu()).mean()

    assert stoi_loss == pytest.approx(-0.46197575330734253)
    assert estoi_loss == pytest.approx(-0.3299955725669861)


def test_torchloudnorm():

    torch.manual_seed(0)
    torch.set_num_threads(1)
    ln = torchloudnorm()

    x = torch.randn(2, 44100)
    if torch.cuda.is_available():
        x = x.cuda()
    y = ln(x)
    div = (y / (x + 1e-8)).cpu().mean()

    assert div == pytest.approx(0.011114642024040222)
