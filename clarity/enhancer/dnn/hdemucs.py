import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade


def apply_separation(mix, model, sample_rate, segment=10.0, overlap=0.1, device=None):
    """Separate audio into sources using the model.
    Args:
        mix (torch.Tensor): Audio tensor of shape (batch, channels, time).
        model (torch.nn.Module): Model to separate audio.
        sample_rate (int): Sample rate of audio.
        segment (float): Length of audio segments in seconds.
        overlap (float): Overlap between segments in seconds.
        device (torch.device): Device to use for inference.
    Returns:
        torch.Tensor: Separated audio tensor of shape (batch, sources, channels, time).
    """
    # Ensure shape (batch, channels, length)
    if mix.ndim == 1:
        # one track and mono audio
        mix = mix.unsqueeze(0)
    elif mix.ndim == 2:
        # one track and stereo audio
        mix = mix.unsqueeze(0)

    # If track is mono, repeat it to match the number of channels
    if mix.shape[1] == 1:
        mix = mix.repeat(1, 2, 1)

    batch, channels, length = mix.shape

    # Move model and audio to device
    mix = mix.to(device)
    model = model.to(device)

    # Normalize audio
    mix = mix / torch.max(torch.abs(mix))

    # Split audio into segments
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    separated_audio = torch.zeros(
        batch, len(model.sources), channels, length, device=device
    )

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        separated_audio[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    # Move audio back to CPU
    separated_audio = separated_audio.cpu()

    return separated_audio


def separate_hdemucs(audio_track=None, segment=10.0, overlap=0.1, device=None):
    """Separate audio track using HDemucs.
    Args:
        audio_track (str): Paths of audio track.
        segment (float): Length of audio segments in seconds.
        overlap (float): Overlap between segments in seconds.
        device (torch.device): Device to use for inference.
    Returns:
        torch.Tensor: Separated audio tensor of shape (sources, channels, time).
    """
    # Load model
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    sources_list = model.sources

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_track)

    sources = apply_separation(
        waveform, model, sample_rate, segment=segment, overlap=overlap, device=device
    ).squeeze(0)

    return dict(zip(sources_list, sources)), sample_rate
