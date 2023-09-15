"""Module that contains functions for source separation."""
from __future__ import annotations

# pylint: disable=import-error
import torch
from torchaudio.transforms import Fade


def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.1,
    device: torch.device | str | None = None,
):
    """
    Apply model to a given mixture.
    Use fade, and add segments together in order to add model segment by segment.

    Args:
        model (torch.nn.Module): model to use for separation
        mix (torch.Tensor): mixture to separate, shape (batch, channels, time)
        sample_rate (int): sampling rate of the mixture
        segment (float): segment length in seconds
        overlap (float): overlap between segments, between 0 and 1
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.

    Returns:
        torch.Tensor: estimated sources

    Based on https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html
    """
    device = mix.device if device is None else torch.device(device)
    mix = torch.as_tensor(mix, device=device)

    if mix.ndim == 2:
        # one track and stereo audio
        mix = mix.unsqueeze(0)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, 4, channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    return final.cpu().detach().numpy()


def get_device(device: str) -> tuple:
    """Get the Torch device.

    Args:
        device (str): device type, e.g. "cpu", "gpu0", "gpu1", etc.

    Returns:
        torch.device: torch.device() appropiate to the hardware available.
        str: device type selected, e.g. "cpu", "cuda".
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    if device.startswith("gpu"):
        device_index = int(device.replace("gpu", ""))
        if device_index > torch.cuda.device_count():
            raise ValueError(f"GPU device index {device_index} is not available.")
        return torch.device(f"cuda:{device_index}"), "cuda"

    if device == "cpu":
        return torch.device("cpu"), "cpu"

    raise ValueError(f"Unsupported device type: {device}")
