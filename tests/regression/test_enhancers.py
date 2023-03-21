import json
import tempfile

import numpy as np
import torch

from clarity.enhancer.dsp import filter  # pylint: disable=redefined-builtin
from clarity.enhancer.gha.audiogram import Audiogram
from clarity.enhancer.gha.gha_interface import GHAHearingAid as gha
from clarity.enhancer.gha.gha_utils import format_gaintable, get_gaintable

gha_params = {  # hyperparameters for GHA Hearing Aid, BE CAREFUL if making changes
    "fs": 44100,
    "ahr": 20,
    "audf": None,
    "cfg_file": "prerelease_combination3_smooth",
    "noisegatelevels": None,
    "noisegateslope": 0,
    "cr_level": 0,
    "max_output_level": 100,
    "equiv0dBSPL": 100,
    "test_nbits": 16,
}


def test_dsp_filter(regtest):
    amfir = filter.AudiometricFIR(sr=44100, nfir=220, device="cpu")
    torch.manual_seed(0)
    signal = torch.rand(10, dtype=torch.float)
    signal = torch.reshape(signal, (1, 1, -1))
    output = amfir(signal.to(amfir.device))
    output = np.round(output.detach().cpu().numpy(), 4)
    regtest.write(f"signal output: \n{output}\n")


def test_gha_audiogram(regtest):
    cfs = np.array([250, 500, 1000, 2000, 4000, 8000])
    for i in [1, 10, 30, 40]:
        np.random.seed(0)
        levels_l = np.round(np.log10(np.random.rand(6) / 20) * (-i), 0)
        levels_r = np.round(np.log10(np.random.rand(6) / 20) * (-i), 0)
        ag = Audiogram(levels_l, levels_r, cfs)
        regtest.write(
            "Audiogram original: \n"
            f"{ag.cfs}\n{ag.levels_l}\n{ag.levels_r}\n{ag.severity}\n"
        )
        ag = ag.select_subset_of_cfs(np.array([500, 1000, 2000]))
        regtest.write(
            f"Audiogram new: \n{ag.cfs}\n{ag.levels_l}\n{ag.levels_r}\n{ag.severity}\n"
        )


def test_GHA_inputs(regtest):
    enhancer = gha(**gha_params)

    infile_names = [
        f"tests/test_data/scenes/S06001_mixed_CH{ch}.wav" for ch in range(1, 4)
    ]
    _fd_merged, merged_filename = tempfile.mkstemp(
        prefix="clarity-merged-", suffix=".wav"
    )
    enhancer.create_HA_inputs(infile_names, merged_filename)

    signal = enhancer.read_signal(merged_filename)
    np.set_printoptions(threshold=100)
    regtest.write(f"signal output: \n{signal[:,0]}\n")


def test_GHA_config(regtest):
    enhancer = gha(**gha_params)
    with open("tests/test_data/metadata/listeners.json", encoding="utf-8") as fp:
        listeners = json.load(fp)
    listener = listeners["L0001"]
    cfs = np.array(listener["audiogram_cfs"], dtype="int")
    audiogram_left = np.array(listener["audiogram_levels_l"])
    audiogram_right = np.array(listener["audiogram_levels_r"])
    audiogram = Audiogram(cfs=cfs, levels_l=audiogram_left, levels_r=audiogram_right)

    cfg_template = f"tests/test_data/openMHA/{enhancer.cfg_file}_template.cfg"

    gaintable = get_gaintable(
        audiogram,
        enhancer.noisegatelevels,
        enhancer.noisegateslope,
        enhancer.cr_level,
        enhancer.max_output_level,
    )
    formatted_sGt = format_gaintable(gaintable, noisegate_corr=True)

    config = enhancer.create_configured_cfgfile(
        "input", "output", formatted_sGt, cfg_template
    )
    regtest.write(f"config output: \n{config}\n")
