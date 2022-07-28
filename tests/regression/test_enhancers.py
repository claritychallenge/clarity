import numpy as np
import torch

from clarity.enhancer.dsp import filter
from clarity.enhancer.gha.audiogram import Audiogram


def test_dsp_filter(regtest):
    amfir = filter.AudiometricFIR(sr=44100, nfir=220)
    torch.manual_seed(0)
    signal = torch.rand(440, dtype=torch.float)
    signal = torch.reshape(signal, (1, 1, -1))
    signal.to(amfir.device)
    output = amfir(signal)
    output = np.round(output.detach().numpy(), 6)
    regtest.write(f"signal output: \n{output}\n")


def test_gha_audiogram(regtest):

    cfs = np.array([250, 500, 1000, 2000, 4000, 8000])
    for i in [1, 10, 30, 40]:
        np.random.seed(0)
        levels_l = np.round(np.log10(np.random.rand(6) / 20) * (-i), 0)
        levels_r = np.round(np.log10(np.random.rand(6) / 20) * (-i), 0)
        ag = Audiogram(levels_l, levels_r, cfs)
        regtest.write(
            f"Audiogram original: \n{ag.cfs}\n{ag.levels_l}\n{ag.levels_r}\n{ag.severity}\n"
        )
        ag = ag.select_subset_of_cfs(np.array([500, 1000, 2000]))
        regtest.write(
            f"Audiogram new: \n{ag.cfs}\n{ag.levels_l}\n{ag.levels_r}\n{ag.severity}\n"
        )
