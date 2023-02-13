# Regression test

import torch

from clarity.engine.losses import SISNRLoss, SNRLoss, STOILevelLoss, STOILoss


def test_sisnr_loss(regtest):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    si_snr_loss = SISNRLoss()
    x = torch.randn(10, 1000)
    y = torch.randn(10, 1000)
    loss = si_snr_loss.forward(x, y)

    regtest.write(f"SISNR loss {loss:0.4f}\n")


def test_snr_loss(regtest):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    snr_loss = SNRLoss()
    x = torch.randn(10, 1000)
    y = torch.randn(10, 1000)
    loss = snr_loss.forward(x, y)

    regtest.write(f"SNR loss {loss:0.6f}\n")


def test_stoi_loss(regtest):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    stoi_loss = STOILoss(sr=16000)
    x = torch.randn(2, 16000)
    y = torch.randn(2, 16000)
    loss = stoi_loss.forward(x, y)

    regtest.write(f"STOI loss {loss:0.7f}\n")


def test_stoi_level_loss(regtest):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    stoi_level_loss = STOILevelLoss(sr=16000, alpha=0.5)
    x = torch.randn(2, 16000)
    y = torch.randn(2, 16000)
    loss = stoi_level_loss.forward(x, y)

    regtest.write(f"STOI level loss {loss:0.7f}\n")
