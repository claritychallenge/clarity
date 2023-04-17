import numpy as np
import torch

from clarity.evaluator.ha_metric.ear_model import EarModel
from clarity.evaluator.haspi.eb import (
    bandwidth_adjust,
    center_frequency,
    convert_rms_to_sl,
    env_compress_basilar_membrane,
    envelope_align,
    envelope_sl,
    gammatone_bandwidth_demodulation,
    gammatone_basilar_membrane,
    inner_hair_cell_adaptation,
    input_align,
    loss_parameters,
    middle_ear,
    resample_24khz,
)


def compare_center_frequency():
    """Test center frequency"""

    eb_center_freq = center_frequency(
        nchan=10,
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    torch_center_freq = ear_model.center_frequency(
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )

    print(
        f"Center frequency - Difference {np.sum(eb_center_freq) - np.sum(torch_center_freq.detach().numpy())}"
    )


def compare_loss_parameters():
    """Test loss parameters"""

    center_freq = center_frequency(
        nchan=10,
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )

    (
        attenuated_ohc,
        bandwith,
        low_knee,
        compression_ratio,
        annenuated_ihc,
    ) = loss_parameters(
        hearing_loss=np.array([45, 45, 50, 60, 70, 80]),
        center_freq=center_freq,
    )

    eb_loss_parameters = {
        "attenuated_ohc": np.sum(attenuated_ohc),
        "bandwith": np.sum(bandwith),
        "low_knee": np.sum(low_knee),
        "compression_ratio": np.sum(compression_ratio),
        "annenuated_ihc": np.sum(annenuated_ihc),
    }

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    center_freq = ear_model.center_frequency(
        shift=None,
        low_freq=80,
        high_freq=8000,
        ear_q=9.26449,
        min_bw=24.7,
    )
    (
        attenuated_ohc,
        bandwith,
        low_knee,
        compression_ratio,
        annenuated_ihc,
    ) = ear_model.loss_parameters(
        hearing_loss=torch.Tensor([45, 45, 50, 60, 70, 80]),
        center_freq=center_freq,
    )

    torch_loss_parameters = {
        "attenuated_ohc": torch.sum(attenuated_ohc).detach().numpy(),
        "bandwith": torch.sum(bandwith).detach().numpy(),
        "low_knee": torch.sum(low_knee).detach().numpy(),
        "compression_ratio": torch.sum(compression_ratio).detach().numpy(),
        "annenuated_ihc": torch.sum(annenuated_ihc).detach().numpy(),
    }

    for key, value in eb_loss_parameters.items():
        print(
            f"Loss Parameters - {key}: difference {value - torch_loss_parameters[key]}"
        )


def compare_resample():
    """Test resample"""
    for reference_freq in [16000, 24000, 44100]:
        np.random.seed(0)
        sig_len = 600
        reference_signal = np.random.random(size=sig_len)

        ref_signal_24, freq_sample_hz = resample_24khz(
            reference_signal, reference_freq, freq_sample_hz=24000
        )
        eb_resample = np.sum(np.abs(ref_signal_24))

        ear_model = EarModel(itype=1, nchan=10, shift=None)

        # Generate signal with numpy to ensure same signal as eb
        np.random.seed(0)
        sig_len = 600
        reference_signal = np.random.random(size=sig_len)

        ref_signal_24, freq_sample_hz = ear_model.resample(
            torch.Tensor(reference_signal), reference_freq
        )

        # check values
        torch_resample = torch.sum(torch.abs(ref_signal_24)).detach().numpy()

        print(
            f"Resample - From {reference_freq} to 24000: difference {eb_resample - torch_resample}"
        )


def compare_input_align():
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    processed_signal = reference_signal.copy()
    processed_signal[50:] = processed_signal[:-50]
    processed_signal[0:50] = 0

    ref, proc = input_align(reference_signal, processed_signal)

    eb_align_ref = np.sum(np.abs(ref))
    eb_align_proc = np.sum(np.abs(proc))

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    torch_ref, torch_proc = ear_model.input_align(
        torch.Tensor(reference_signal), torch.Tensor(processed_signal)
    )

    print(
        f"Input align - Ref difference {eb_align_ref - torch.sum(torch.abs(torch_ref)).detach().numpy()}"
    )
    print(
        f"Input align - Proc difference {eb_align_proc - torch.sum(torch.abs(torch_proc)).detach().numpy()}"
    )


def compare_middle_ear():
    np.random.seed(0)
    sig_len = 600
    reference_signal = 100 * np.random.random(size=sig_len)
    filtered_signal = middle_ear(reference_signal, 24000)

    eb_middle_ear = np.sum(np.abs(filtered_signal))

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    torch_filtered_signal = ear_model.middle_ear(torch.Tensor(reference_signal))
    torch_middle_ear = torch.sum(torch.abs(torch_filtered_signal)).detach().numpy()

    print(f"Middle ear - Difference {eb_middle_ear - torch_middle_ear}")


def compare_gammatone_bandwidth_demodulation():
    centre_freq_sin, centre_freq_cos = gammatone_bandwidth_demodulation(
        npts=100,
        tpt=0.001,
        center_freq=1000,
        center_freq_cos=np.zeros(100),
        center_freq_sin=np.zeros(100),
    )
    centre_freq_sin = np.sum(centre_freq_sin)
    centre_freq_cos = np.sum(centre_freq_cos)

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    (
        torch_centre_freq_sin,
        torch_centre_freq_cos,
    ) = ear_model.gammatone_bandwidth_demodulation(
        npts=100,
        tpt=0.001,
        center_freq=1000,
    )
    torch_centre_freq_sin = torch.sum(torch_centre_freq_sin).detach().numpy()
    torch_centre_freq_cos = torch.sum(torch_centre_freq_cos).detach().numpy()
    print(
        f"Gammatone bandwidth demodulation - Centre freq sin: difference {centre_freq_sin - torch_centre_freq_sin}"
    )
    print(
        f"Gammatone bandwidth demodulation - Centre freq cos: difference {centre_freq_cos - torch_centre_freq_cos}"
    )


def compare_gammatone_basilar_membrane():
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

    eb_ref_envelope = np.sum(np.abs(reference_envelope))
    eb_ref_basilar_membrane = np.sum(np.abs(reference_basilar_membrane))
    eb_proc_envelope = np.sum(np.abs(processed_envelope))
    eb_proc_basilar_membrane = np.sum(np.abs(processed_basilar_membrane))

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    (
        reference_envelope,
        reference_basilar_membrane,
        processed_envelope,
        processed_basilar_membrane,
    ) = ear_model.gammatone_basilar_membrane(
        reference=torch.tensor(ref),
        reference_bandwidth=1.4,
        processed=torch.tensor(proc),
        processed_bandwidth=2.0,
        center_freq=1000,
        ear_q=9.26449,
        min_bandwidth=24.7,
    )

    torch_ref_envelope = torch.sum(torch.abs(reference_envelope)).detach().numpy()
    torch_ref_basilar_membrane = (
        torch.sum(torch.abs(reference_basilar_membrane)).detach().numpy()
    )
    torch_proc_envelope = torch.sum(torch.abs(processed_envelope)).detach().numpy()
    torch_proc_basilar_membrane = (
        torch.sum(torch.abs(processed_basilar_membrane)).detach().numpy()
    )

    print(
        f"Gammatone Basilar Membrane - Reference envelope: difference {eb_ref_envelope - torch_ref_envelope}"
    )
    print(
        f"Gammatone Basilar Membrane - Reference basilar membrane: difference {eb_ref_basilar_membrane - torch_ref_basilar_membrane}"
    )
    print(
        f"Gammatone Basilar Membrane - Processed envelope: difference {eb_proc_envelope - torch_proc_envelope}"
    )
    print(
        f"Gammatone Basilar Membrane - Processed basilar membrane: difference {eb_proc_basilar_membrane - torch_proc_basilar_membrane}"
    )


def compare_bandwidth_adjust():
    scale = 1000.0
    bw_min = 1.0
    bw_max = 2.0

    bw_adjusted = bandwidth_adjust(
        control=scale * np.array([1, -1, 1]),
        bandwidth_min=bw_min,
        bandwidth_max=bw_max,
        level1=1,
    )

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    bw_adjusted_torch = ear_model.bandwidth_adjust(
        control=torch.tensor(scale * np.array([1, -1, 1]), dtype=torch.float32),
        bandwidth_min=bw_min,
        bandwidth_max=bw_max,
        level1=1,
    )
    bw_adjusted_torch = torch.sum(bw_adjusted_torch).detach().numpy()

    print(f"Bandwidth adjust: Difference {bw_adjusted - bw_adjusted_torch}")


def compare_env_compress_basilar_membrane():
    """Test env_compress_basilar_membrane"""
    np.random.seed(0)
    sig_len = 600
    env_sig = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001
    control = np.random.random(size=sig_len)
    compressed_signal, compressed_basilar_membrane = env_compress_basilar_membrane(
        env_sig,
        bm,  # pylint: disable=invalid-name
        control,
        attn_ohc=0.01,
        threshold_low=70.0,
        compression_ratio=0.1,
        fsamp=24000,
        level1=140,
        small=1e-30,
        threshold_high=100,
    )
    # check shapes
    eb_compressed_signal = np.mean(np.abs(compressed_signal))
    eb_compressed_basilar_membrane = np.mean(np.abs(compressed_basilar_membrane))

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    (
        compressed_signal,
        compressed_basilar_membrane,
    ) = ear_model.env_compress_basilar_membrane(
        torch.tensor(env_sig, dtype=torch.float32),
        torch.tensor(bm, dtype=torch.float32),
        torch.tensor(control, dtype=torch.float32),
        attn_ohc=0.01,
        threshold_low=70.0,
        compression_ratio=0.1,
        level1=140,
        threshold_high=100,
    )

    torch_compressed_signal = torch.mean(torch.abs(compressed_signal)).detach().numpy()
    torch_compressed_basilar_membrane = (
        torch.mean(torch.abs(compressed_basilar_membrane)).detach().numpy()
    )

    print(
        f"Env compress basilar membrane - Compressed signal: difference {eb_compressed_signal - torch_compressed_signal}"
    )
    print(
        f"Env compress basilar membrane - Compressed basilar membrane: difference {eb_compressed_basilar_membrane - torch_compressed_basilar_membrane}"
    )


def compare_envelope_align():
    """Test envelope align"""

    np.random.seed(0)
    sig_len = 600
    scale = 1.1
    reference = np.random.random(size=sig_len)

    # Make output look like a shifted copy of the reference
    output = reference.copy()
    output[50:] = output[:-50]
    output[0:50] = 0
    output *= scale

    aligned_output = envelope_align(
        reference, output, freq_sample=24000, corr_range=100
    )
    eb_envelope_align = np.sum(np.abs(aligned_output))

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    aligned_output = ear_model.envelope_align(
        torch.tensor(reference, dtype=torch.float32),
        torch.tensor(output, dtype=torch.float32),
        corr_range=100,
    )
    torch_envelope_align = torch.sum(torch.abs(aligned_output)).detach().numpy()

    print(
        f"Envelope align - Aligned output: difference {eb_envelope_align - torch_envelope_align}"
    )


def compare_envelope_sl():
    """Test envelope sl"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001

    reference, basilar_membrane = envelope_sl(
        reference=ref,
        basilar_membrane=bm,
        attenuated_ihc=40.0,
        level1=120,
        small=1e-30,
    )

    eb_reference = np.sum(np.abs(reference))
    eb_basilar_membrane = np.sum(np.abs(basilar_membrane))

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    reference, basilar_membrane = ear_model.envelope_sl(
        torch.tensor(ref, dtype=torch.float32),
        torch.tensor(bm, dtype=torch.float32),
        attenuated_ihc=40.0,
        level1=120,
    )

    torch_reference = torch.sum(torch.abs(reference)).detach().numpy()
    torch_basilar_membrane = torch.sum(torch.abs(basilar_membrane)).detach().numpy()

    print(f"Envelope sl - Reference: difference {eb_reference - torch_reference}")
    print(
        f"Envelope sl - Basilar membrane: difference {eb_basilar_membrane - torch_basilar_membrane}"
    )


def compare_inner_hair_cell_adaptation():
    """Test inner hair cell adaptation"""

    np.random.seed(0)
    sig_len = 600
    ref = np.random.random(size=sig_len)
    bm = np.random.random(size=sig_len) * 0.001

    output_db, output_basilar_membrane = inner_hair_cell_adaptation(
        reference_db=ref, reference_basilar_membrane=bm, delta=1.00, freq_sample=24000
    )

    eb_output_db = np.sum(np.abs(output_db))
    eb_output_basilar_membrane = np.sum(np.abs(output_basilar_membrane))

    ear_model = EarModel(itype=1, nchan=10, shift=None)
    output_db, output_basilar_membrane = ear_model.inner_hair_cell_adaptation(
        torch.tensor(ref, dtype=torch.float32),
        torch.tensor(bm, dtype=torch.float32),
        delta=1.00,
    )

    torch_output_db = torch.sum(torch.abs(output_db)).detach().numpy()
    torch_output_basilar_membrane = (
        torch.sum(torch.abs(output_basilar_membrane)).detach().numpy()
    )

    print(
        f"Inner hair cell adaptation - Output db: difference {eb_output_db - torch_output_db}"
    )
    print(
        f"Inner hair cell adaptation - Output basilar membrane: difference {eb_output_basilar_membrane - torch_output_basilar_membrane}"
    )


if __name__ == "__main__":
    compare_center_frequency()
    compare_loss_parameters()
    compare_resample()
    compare_input_align()
    compare_middle_ear()
    compare_gammatone_bandwidth_demodulation()
    compare_gammatone_basilar_membrane()
    compare_bandwidth_adjust()
    compare_env_compress_basilar_membrane()
    compare_envelope_align()
    compare_envelope_sl()
    compare_inner_hair_cell_adaptation()
