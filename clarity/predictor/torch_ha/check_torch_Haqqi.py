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


def check_ear_model():
    """Test ear model"""
    np.random.seed(0)
    samp_freq = 24000
    sig_len = 24000
    for sig_len in [24000]:
        for center_freq in [80, 114.49661]:
            tpt = 0.0002617993877991494
            sincf, coscf = gammatone_bandwidth_demodulation(
                sig_len, tpt, center_freq, np.zeros(sig_len), np.zeros(sig_len)
            )
            a = np.stack((sincf, coscf), axis=1)
            np.savetxt(f"a_{sig_len}_{center_freq}.csv", a, delimiter=",")

    ref = np.random.random(size=sig_len)
    proc = np.random.random(size=sig_len)
    ref_db, ref_bm, proc_db, proc_bm, ref_sl, proc_sl, freq_sample = eb_ear_model(
        reference=ref,
        reference_freq=samp_freq,
        processed=ref + proc,
        processed_freq=samp_freq,
        hearing_loss=np.array([45, 45, 35, 45, 60, 65]),
        itype=0,
        level1=65,
        nchan=32,
        m_delay=1,
        shift=0.0,
    )

    ear_model = EarModel(nchan=10)
    (
        t_ref_db,
        t_ref_bm,
        t_proc_db,
        t_proc_bm,
        t_ref_sl,
        t_proc_sl,
    ) = ear_model.forward(
        reference=torch.tensor(ref),
        processed=torch.tensor(ref + proc),
        hearing_loss=torch.tensor([45, 45, 35, 45, 60, 65]),
        equalisation=0,
        level1=65,
    )

    print(np.sum(np.abs(ref_db)) - np.sum(np.abs(t_ref_db.detach().numpy())))
    print(np.sum(np.abs(ref_bm)) - np.sum(np.abs(t_ref_bm.detach().numpy())))
    print(np.sum(np.abs(proc_db)) - np.sum(np.abs(t_proc_db.detach().numpy())))
    print(np.sum(np.abs(proc_bm)) - np.sum(np.abs(t_proc_bm.detach().numpy())))
    print(np.sum(np.abs(ref_sl)) - np.sum(np.abs(t_ref_sl.detach().numpy())))
    print(np.sum(np.abs(proc_sl)) - np.sum(np.abs(t_proc_sl.detach().numpy())))

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
    ear_model = EarModel(
        nchan=10,
    )
    t_center_freq = ear_model.center_frequency(
        low_freq=80,
        high_freq=8000,
        shift=None,
        ear_q=9.26449,
        min_bw=24.7,
    )

    print(
        "Center Frequency :",
        np.sum(center_freq) - np.sum(t_center_freq.detach().numpy()),
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

    ear_model = EarModel(nchan=6)

    (
        attenuated_ohc_t,
        bandwidth_t,
        low_knee_t,
        compression_ratio_t,
        attenuated_ihc_t,
    ) = ear_model.loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
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

    ear_model = EarModel(nchan=6)
    t_ref, t_proc = ear_model.input_align(
        torch.tensor(reference_signal),
        torch.tensor(processed_signal),
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

    ear_model = EarModel()
    t_filtered_signal = ear_model.middle_ear(torch.tensor(reference_signal))

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

    ear_model = EarModel()
    (
        t_reference_envelope,
        t_reference_basilar_membrane,
        t_processed_envelope,
        t_processed_basilar_membrane,
    ) = ear_model.gammatone_basilar_membrane(
        torch.tensor(ref),
        torch.tensor(1.4),
        torch.tensor(proc),
        torch.tensor(2.0),
        torch.tensor(1000),
        torch.tensor(9.26449),
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
    # check_center_frequency()
    # check_loss_parameters()
    # check_input_align()
    # check_middle_ear()
    # check_gammatone_basilar_membrane()
    # check_gammatone_bandwidth_demodulation()
    check_ear_model()
