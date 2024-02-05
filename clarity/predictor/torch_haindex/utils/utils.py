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

    # Compute correlation for each pair of samples independently
    out = torch.nn.functional.conv1d(
        x_padded.unsqueeze(1),
        y_padded.unsqueeze(1),
        padding="same",
    )
    out = out[..., :out_len]

    mask = torch.arange(x.shape[0])
    out = out[mask, mask, :]
    return out


# Coefficients for filter in env_compress_basilar_membrane for Ear Model
# Coefficients are the output of:
#    fsamp = 24000
#    flp = 800
#    b, a = butter(1, flp / (0.5 * fsamp))
COMPRESS_BASILAR_MEMBRANE_COEFS = {
    "24000": {
        "b": [0.09510798340249643, 0.09510798340249643],
        "a": [1.0, -0.8097840331950071],
    }
}
# Coefficients for low pass butterworth filter in Middle ear for Ear Model
# Coefficients are output of:
#    freq_sample = 24000
#    butterworth_low_pass, low_pass = butter(1, 5000 / (0.5 * freq_sample))
#    butterworth_high_pass, high_pass = butter(2, 350 / (0.5 * freq_sample), "high")
MIDDLE_EAR_COEF = {
    "24000": {
        "butterworth_low_pass": [0.4341737512063021, 0.4341737512063021],
        "low_pass": [1.0, -0.13165249758739583],
        "butterworth_high_pass": [
            0.9372603902698923,
            -1.8745207805397845,
            0.9372603902698923,
        ],
        "high_pass": [1.0, -1.8705806407352794, 0.8784609203442912],
    }
}

#  Coefficients for group_delay_compensate

CORRECT_DELAY_COEFS = [
    0,
    50,
    92,
    127,
    157,
    183,
    205,
    225,
    242,
    256,
    267,
    275,
    283,
    291,
    299,
    305,
    311,
    316,
    320,
    325,
    329,
    332,
    335,
    338,
    340,
    341,
    342,
    344,
    344,
    345,
    346,
    347,
]
