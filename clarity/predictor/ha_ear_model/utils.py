"""Torch utils for computing HAAQI score"""

import torch


def full_correlation(x, y):
    # Compute lengths of input tensors
    x_len = x.shape[-1]
    y_len = y.shape[-1]
    out_len = x_len + y_len - 1

    # Pad tensors with zeros
    x_padded = torch.nn.functional.pad(x, (y_len - 1, 0))
    y_padded = torch.nn.functional.pad(y, (x_len - 1, 0))

    # Compute convolution of padded tensors
    out = torch.nn.functional.conv1d(
        x_padded.unsqueeze(0).unsqueeze(0),
        y_padded.unsqueeze(0).unsqueeze(0),
        padding="same",
    )

    # Crop output tensor to original size
    out = out[..., :out_len]

    return out.squeeze().squeeze()
