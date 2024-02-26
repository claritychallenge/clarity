import numpy as np
import torch

from clarity.evaluator.haspi.eb import center_frequency
from clarity.evaluator.haspi.eb import ear_model as eb_ear_model
from clarity.evaluator.haspi.eb import (
    gammatone_bandwidth_demodulation,
    gammatone_basilar_membrane,
    input_align,
    loss_parameters,
    middle_ear,
)
from clarity.predictor.torch_ha.torch_earmodel import EarModel
from clarity.predictor.torch_ha.torch_earmodel import (
    center_frequency as torch_center_frequency,
)
from clarity.predictor.torch_ha.torch_earmodel import gammatone_bandwidth_demodulation


def check_ear_model():
    """Test ear model"""
    np.random.seed(0)
    sig_len = 24000
    samp_freq = 24000

    ref = np.random.random(size=sig_len)
    proc = np.random.random(size=sig_len)
    ref2 = np.random.random(size=sig_len)
    proc2 = np.random.random(size=sig_len)
    ref3 = np.random.random(size=sig_len)
    proc3 = np.random.random(size=sig_len)

    ear_model = EarModel(
        num_bands=32,
        equalisation=0,
        signal_length=sig_len,
        batch_size=3,
    ).to("cuda")
    (
        t_ref_db,
        t_ref_bm,
        t_proc_db,
        t_proc_bm,
        t_ref_sl,
        t_proc_sl,
    ) = ear_model.forward(
        reference=torch.tensor(np.array([ref, ref2, ref3])).to("cuda"),
        processed=torch.tensor(np.array([ref + proc, ref2 + proc2, ref3 + proc3])).to(
            "cuda"
        ),
        hearing_loss=torch.tensor(
            np.array(
                [
                    [45, 45, 35, 45, 60, 65],
                    [20, 20, 20, 25, 25, 25],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
        ).to("cuda"),
        level1=torch.tensor(
            np.array(
                [
                    [65],
                    [50],
                    [44],
                ]
            )
        ).to("cuda"),
    )

    print(t_ref_db.shape)

    print("Done!")


def check_center_frequency():
    """Test center frequency"""

    center_freq = center_frequency(
        nchan=10,
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )

    t_center_freq = torch_center_frequency(
        nchan=10,
        low_freq=80,
        high_freq=8000,
        shift=None,
        ear_q=9.26449,
        min_bw=24.7,
    )

    print(
        "Center Frequency :",
        np.sum(center_freq) - np.sum(t_center_freq),
    )


def check_loss_parameters():
    """Test loss parameters"""

    (
        attenuated_ohc,
        bandwidth,
        low_knee,
        compression_ratio,
        attenuated_ihc,
    ) = loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
        audiometric_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
    )

    ear_model = EarModel(num_bands=6, device="cpu")

    (
        attenuated_ohc_t,
        bandwidth_t,
        low_knee_t,
        compression_ratio_t,
        attenuated_ihc_t,
    ) = ear_model.loss_parameters(
        hearing_loss=torch.tensor(
            np.array([[45, 45, 50, 60, 70, 80]]), dtype=torch.float64
        ),
        center_freq=torch.tensor(np.array([[250, 500, 1000, 2000, 4000, 6000]])),
    )

    print("Loss Parameters")
    print(
        "Attenuated_ohc  :",
        np.sum(attenuated_ohc) - np.sum(attenuated_ohc_t.detach().numpy()),
    )
    print("Bandwidth  :", np.sum(bandwidth) - np.sum(bandwidth_t.detach().numpy()))
    print("low_knee  :", np.sum(low_knee) - np.sum(low_knee_t.detach().numpy()))
    print(
        "Compression_ratio  :",
        np.sum(compression_ratio) - np.sum(compression_ratio_t.detach().numpy()),
    )
    print(
        "Attenuated_ihc  :",
        np.sum(attenuated_ihc) - np.sum(attenuated_ihc_t.detach().numpy()),
    )


def check_input_align():
    """Test input align"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    ref, proc = input_align(reference_signal, processed_signal)

    ear_model = EarModel(num_bands=6, device="cpu")
    t_ref, t_proc = ear_model.input_align(
        torch.tensor(reference_signal).unsqueeze(0),
        torch.tensor(processed_signal).unsqueeze(0),
    )

    print("Input Align")
    print(
        "Reference  :",
        np.sum(ref) - np.sum(t_ref.detach().numpy()),
    )
    print(
        "Processed  :",
        np.sum(proc) - np.sum(t_proc.detach().numpy()),
    )


def check_middle_ear():
    """Test middle ear"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    filtered_signal = middle_ear(reference_signal, 24000)

    ear_model = EarModel(device="cpu")
    t_filtered_signal = ear_model.middle_ear(
        torch.tensor(reference_signal).unsqueeze(0)
    )

    print("Middle Ear")
    print(
        "Filtered Signal  :",
        np.sum(filtered_signal) - np.sum(t_filtered_signal.detach().numpy()),
    )


def check_gammatone_basilar_membrane():
    """Test gammatone basilar membrane"""
    np.random.seed(0)
    sig_len = 600
    ref = 100 * np.random.random(size=sig_len)
    proc = ref + 10 * np.random.random(size=sig_len)

    (
        reference_envelope,
        reference_basilar_membrane,
        processed_envelope,
        processed_basilar_membrane,
    ) = gammatone_basilar_membrane(
        reference=ref,
        reference_bandwidth=1.4,
        processed=proc,
        processed_bandwidth=2.0,
        freq_sample=24000,
        center_freq=1000,
        ear_q=9.26449,
        min_bandwidth=24.7,
    )

    ear_model = EarModel(device="cpu")
    (
        t_reference_envelope,
        t_reference_basilar_membrane,
        t_processed_envelope,
        t_processed_basilar_membrane,
    ) = ear_model.gammatone_basilar_membrane(
        torch.tensor(ref).unsqueeze(0),
        torch.tensor(1.4).unsqueeze(0),
        torch.tensor(proc).unsqueeze(0),
        torch.tensor(2.0).unsqueeze(0),
        torch.tensor(1000).unsqueeze(0),
        torch.tensor(9.26449).unsqueeze(0),
    )

    print("Gammatone Basilar Membrane")
    print(
        "Reference Envelope  :",
        np.sum(reference_envelope) - np.sum(t_reference_envelope.detach().numpy()),
    )
    print(
        "Reference Basilar Membrane  :",
        np.sum(reference_basilar_membrane)
        - np.sum(t_reference_basilar_membrane.detach().numpy()),
    )
    print(
        "Processed Envelope  :",
        np.sum(processed_envelope) - np.sum(t_processed_envelope.detach().numpy()),
    )
    print(
        "Processed Basilar Membrane  :",
        np.sum(processed_basilar_membrane)
        - np.sum(t_processed_basilar_membrane.detach().numpy()),
    )


def check_gammatone_bandwidth_demodulation():
    """Test gammatone bandwidth demodulation"""
    centre_freq_sin, centre_freq_cos = gammatone_bandwidth_demodulation(
        npts=100,
        tpt=0.001,
        center_freq=1000,
        center_freq_cos=np.zeros(100),
        center_freq_sin=np.zeros(100),
    )

    ear_model = EarModel()
    (
        t_centre_freq_sin,
        t_centre_freq_cos,
    ) = ear_model.gammatone_bandwidth_demodulation(
        torch.tensor(100),
        torch.tensor(0.001),
        torch.tensor(1000),
    )

    print("Gammatone Bandwidth Demodulation")
    print(
        "Centre Freq Sin  :",
        np.sum(centre_freq_sin) - np.sum(t_centre_freq_sin.detach().numpy()),
    )
    print(
        "Centre Freq Cos  :",
        np.sum(centre_freq_cos) - np.sum(t_centre_freq_cos.detach().numpy()),
    )


if __name__ == "__main__":
    check_center_frequency()
    check_loss_parameters()
    # check_input_align()
    check_middle_ear()
    # check_gammatone_basilar_membrane()
    # check_gammatone_bandwidth_demodulation()
    check_ear_model()
