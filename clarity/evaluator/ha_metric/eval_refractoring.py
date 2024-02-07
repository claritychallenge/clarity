import numpy as np

from clarity.evaluator.haspi.eb import (
    center_frequency,
    loss_parameters,
    input_align,
    middle_ear,
    gammatone_basilar_membrane,
)

from clarity.evaluator.ha_metric import ear_model
from clarity.evaluator.ha_metric import gammatone_filter


def eval_center_frequency():
    """Test center frequency"""

    center_freq = center_frequency(
        nchan=10,
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )

    ear = ear_model.EarModel(
        equalisation=1,
        num_bands=10,
        ear_q=9.26449,
    )
    ear_model_cf = ear.center_frequency(
        low_freq=80, high_freq=8000, min_bw=24.7, shift=None
    )

    print(f"Centre Frequency difference: {np.sum(center_freq) - np.sum(ear_model_cf)}")


def eval_loss_parameters():
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

    ear = ear_model.EarModel(
        equalisation=1,
        num_bands=10,
        ear_q=9.26449,
    )
    ear_model_cf = ear.loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
        audiometric_freq=np.array([250, 500, 1000, 2000, 4000, 6000]),
    )

    print("\nLoss Parameters difference: ")
    print(f"- Attenuated OHC: {np.sum(attenuated_ohc) - np.sum(ear_model_cf[0])}")
    print(f"- Bandwidth: {np.sum(bandwidth) - np.sum(ear_model_cf[1])}")
    print(f"- Low Knee: {np.sum(low_knee) - np.sum(ear_model_cf[2])}")
    print(f"- Compression Ratio: {np.sum(compression_ratio) - np.sum(ear_model_cf[3])}")
    print(f"- Attenuated IHC: {np.sum(attenuated_ihc) - np.sum(ear_model_cf[4])}")


def eval_input_align():
    """Test input align"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    ref, proc = input_align(reference_signal, processed_signal)

    ear = ear_model.EarModel(
        equalisation=1,
        num_bands=10,
        ear_q=9.26449,
    )
    ref_ha, proc_ha = ear.input_align(reference_signal, processed_signal)

    print("\nInput Align difference: ")
    print(f"- Reference: {np.sum(ref) - np.sum(ref_ha)}")
    print(f"- Processed: {np.sum(proc) - np.sum(proc_ha)}")


def eval_middle_ear():
    """Test middle ear"""
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    filtered_signal = middle_ear(reference_signal, 24000)

    ear = ear_model.EarModel(
        equalisation=1,
        num_bands=10,
        ear_q=9.26449,
    )
    ear_model_filtered = ear.middle_ear(reference_signal, 24000)

    print(
        f"\nMiddle Ear difference: {np.sum(filtered_signal) - np.sum(ear_model_filtered)}"
    )


def evalt_gammatone_basilar_membrane():
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


if __name__ == "__main__":
    eval_center_frequency()
    eval_loss_parameters()
    eval_input_align()
    eval_middle_ear()
