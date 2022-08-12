# Regression test

import torch

from clarity.predictor.torch_msbg import MSBGHearingModel, torchloudnorm
from clarity.predictor.torch_stoi import NegSTOILoss


def test_torch_msbg_stoi(regtest):

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

    regtest.write(
        f"Torch MSBG STOILoss {stoi_loss:0.5f}, ESTOILoss {estoi_loss:0.5f}\n"
    )


def test_torchloudnorm(regtest):

    torch.manual_seed(0)
    torch.set_num_threads(1)
    ln = torchloudnorm()

    x = torch.randn(2, 44100)
    if torch.cuda.is_available():
        x = x.cuda()
    y = ln(x)
    div = (y / (x + 1e-8)).cpu().mean()

    regtest.write(f"Torch Loudnorm div is {div:0.4f}\n")
