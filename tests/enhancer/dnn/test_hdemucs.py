import torch
import torchaudio
from torchaudio.utils import download_asset

from clarity.enhancer.dnn.hdemucs import apply_separation, separate_hdemucs


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


def test_apply_separation():
    sample_rate = 44100
    signal = torch.rand(sample_rate * 10)

    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()

    device = torch.device("cpu")
    model.to(device)

    y = apply_separation(
        signal, model, sample_rate, segment=5.0, overlap=0.1, device=device
    )
    assert y.shape == (1, 4, 2, 7560512)
    signal = torch.rand(sample_rate * 10, 2)
    y = apply_separation(
        signal, model, sample_rate, segment=5.0, overlap=0.1, device=device
    )
    assert y.shape == (1, 4, 2, 7560512)
    signal = signal.unsqueeze(0)
    y = apply_separation(
        signal, model, sample_rate, segment=5.0, overlap=0.1, device=device
    )
    assert y.shape == (1, 4, 2, 7560512)
