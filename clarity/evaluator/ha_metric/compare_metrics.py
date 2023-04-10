import numpy as np
import torch

from clarity.evaluator.ha_metric.ear_model import EarModel
from clarity.evaluator.haspi.eb import (
    center_frequency,
    convert_rms_to_sl,
    loss_parameters,
    resample_24khz,
)


def eb_center_frequency():
    """Test center frequency"""

    center_freq = center_frequency(
        nchan=10,
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )
    return np.sum(center_freq)


def eb_loss_parameters():
    """Test loss parameters"""

    (
        attenuated_ohc,
        bandwith,
        low_knee,
        compression_ratio,
        annenuated_ihc,
    ) = loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
        audiometric_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
    )

    return {
        "attenuated_ohc": np.sum(attenuated_ohc),
        "bandwith": np.sum(bandwith),
        "low_knee": np.sum(low_knee),
        "compression_ratio": np.sum(compression_ratio),
        "annenuated_ihc": np.sum(annenuated_ihc),
    }


def eb_resample():
    """Test resample"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = np.random.random(size=sig_len)
    reference_freq = 12000
    ref_signal_24, freq_sample_hz = resample_24khz(
        reference_signal, reference_freq, freq_sample_hz=24000
    )

    # check values
    return np.sum(np.abs(ref_signal_24))


def eb_convert_rms_to_sl():
    """Test convert rms to sl"""

    np.random.seed(0)
    sig_len = 600
    reference = np.random.random(size=sig_len)
    control = np.random.random(size=sig_len)

    ref_db = convert_rms_to_sl(
        reference=reference,
        control=control,
        attnenuated_ohc=0.1,
        threshold_low=40,
        compression_ratio=10,
        attnenuated_ihc=0.1,
        level1=120,
        threshold_high=100,
        small=1e-30,
    )

    return np.sum(np.abs(ref_db))


def torch_center_frequency():
    """Test center frequency"""

    ear_model = EarModel(nchan=10)
    center_freq = ear_model.center_frequency(
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )
    return torch.sum(center_freq).detach().numpy()


def torch_loss_parameters():
    """Test loss parameters"""

    ear_model = EarModel(nchan=10)
    (
        attenuated_ohc,
        bandwith,
        low_knee,
        compression_ratio,
        annenuated_ihc,
    ) = ear_model.loss_parameters(
        hearing_loss=torch.Tensor([45, 45, 50, 60, 70, 80]),
        center_freq=torch.Tensor([250, 500, 1000, 2000, 4000, 6000]),
        audiometric_freq=torch.Tensor([250, 500, 1000, 2000, 4000, 6000]),
    )

    return {
        "attenuated_ohc": torch.sum(attenuated_ohc).detach().numpy(),
        "bandwith": torch.sum(bandwith).detach().numpy(),
        "low_knee": torch.sum(low_knee).detach().numpy(),
        "compression_ratio": torch.sum(compression_ratio).detach().numpy(),
        "annenuated_ihc": torch.sum(annenuated_ihc).detach().numpy(),
    }


def torch_resample():
    """Test resample"""
    ear_model = EarModel(nchan=10)

    # Generate signal with numpy to ensure same signal as eb
    np.random.seed(0)
    sig_len = 600
    reference_signal = np.random.random(size=sig_len)

    reference_freq = 12000
    ref_signal_24, freq_sample_hz = ear_model.resample(
        torch.Tensor(reference_signal), reference_freq, 24000
    )

    # check values
    return torch.sum(torch.abs(ref_signal_24)).detach().numpy()


def torch_convert_rms_to_sl():
    """Test convert rms to sl"""

    ear_model = EarModel(nchan=10)
    np.random.seed(0)
    sig_len = 600
    reference = torch.Tensor(np.random.random(size=sig_len))
    control = torch.Tensor(np.random.random(size=sig_len))

    ref_db = ear_model.convert_rms_to_sl(
        reference=reference,
        control=control,
        attnenuated_ohc=0.1,
        threshold_low=40,
        compression_ratio=10,
        attnenuated_ihc=0.1,
        level1=120,
        threshold_high=100,
        small=1e-30,
    )

    return torch.sum(np.abs(ref_db)).detach().numpy()


if __name__ == "__main__":
    print(
        f"Center frequency: difference {eb_center_frequency() - torch_center_frequency()}"
    )
    for key, value in eb_loss_parameters().items():
        print(
            f"Loss Parameters - {key}: difference {value - torch_loss_parameters()[key]}"
        )
    print(f"Resample: difference {eb_resample() - torch_resample()}")
    print(
        f"Convert RMS to SL: difference {eb_convert_rms_to_sl() - torch_convert_rms_to_sl()}"
    )
