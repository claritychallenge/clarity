import torch
from torchaudio.utils import download_asset

from clarity.utils import separate_hdemucs


def test_separate_hdemucs():
    SAMPLE_SONG = download_asset("tutorial-assets/hdemucs_mix.wav")
    y, sr = separate_hdemucs(
        audio_track=SAMPLE_SONG,
        segment=5.0,
        overlap=0.1,
        device=torch.device("cpu"),
    )
    assert sr == 44100
    assert y["drums"].shape == (2, 7560512)
    assert y["bass"].shape == (2, 7560512)
    assert y["other"].shape == (2, 7560512)
    assert y["vocals"].shape == (2, 7560512)
