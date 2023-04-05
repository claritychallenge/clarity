"""HAAQI-RMS metric"""
import numpy as np
from scipy.signal import correlate, correlation_lags

from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.signal_processing import compute_rms


def compute_haaqi_rms(
    processed_signal: np.ndarray,
    reference_signal: np.ndarray,
    audiogram: np.ndarray,
    audiogram_frequencies: np.ndarray,
    sample_rate: int,
    silence_length: float = 2.0,
) -> float:
    """Compute HAAQI-RMS metric

    Metric is a combination of HAAQI and RMS. Signals are split into
    non-silence and silence segments based on the reference signal.
    HAAQI is computed on the non-silence parts and RMS is computed
    on the silence parts.

    The HAAQI part of the metric measure the music quality and the RMS
    part of the metric punishes bad separations. I.e. if the RMS is high
    it means that parts that should be silence contains residual or
    artefact errors.

    Args:
        processed_signal (np.ndarray): Output signal with noise, distortion, HA gain,
            and/or processing.
        reference_signal (np.ndarray): Input reference speech signal with no
            noise or distortion. If a hearing loss is specified, NAL-R
            equalization is optional.
        audiogram (np.ndarray): Vector of hearing loss at the audiogram_frequencies
        audiogram_frequencies (np.ndarray): Audiogram frequencies
        sample_rate (int): Sample rate in Hz.
        silence_length (float): Minimum length of silence in seconds to use
            for RMS calculation.
            Segments of silence shorter than this are included in non_silence segments.
            Defaults to 2.

    Returns:
        float: HAAQI-RMS metric
    """

    # align signals
    processed_signal = align_signals(processed_signal, reference_signal)

    # find silence segments
    silence, non_silence = find_silence_segments(
        reference_signal, sample_rate, silence_length
    )

    # join non-silence segments for processed and reference signals
    new_processed_signal = []
    new_reference_signal = []
    for start, end in non_silence:
        new_processed_signal.append(processed_signal[start:end])
        new_reference_signal.append(reference_signal[start:end])
    new_reference_signal = np.concatenate(new_reference_signal)
    new_processed_signal = np.concatenate(new_processed_signal)

    # join silence segments for processed signal
    silence_processed = []
    for start, end in silence:
        silence_processed.append(processed_signal[start:end])
    silence_processed = np.concatenate(silence_processed)

    # Compute haaqi on music segments
    haaqi_score = compute_haaqi(
        processed_signal=new_processed_signal,
        reference_signal=new_reference_signal,
        sample_rate=sample_rate,
        audiogram=audiogram,
        audiogram_frequencies=audiogram_frequencies,
        equalisation=1,
        scale_reference=True,
    )

    # Compute rms on silence segments
    rms_score = compute_rms(silence_processed)

    return (
        haaqi_score * len(new_reference_signal) - rms_score * len(silence_processed)
    ) / (len(new_reference_signal) + len(silence_processed))


def align_signals(
    processed_signal: np.ndarray, reference_signal: np.ndarray
) -> np.ndarray:
    """Align processed signal to reference signals

    Args:
        processed_signal (np.ndarray): Output signal with noise, distortion, HA gain,
            and/or processing.
        reference_signal (np.ndarray): Input reference speech signal with no
            noise or distortion. If a hearing loss is specified, NAL-R
            equalization is optional.
    Returns:
        np.ndarray: Aligned processed signal
    """

    processed_n = len(processed_signal)

    corr = correlate(processed_signal, reference_signal, mode="same")
    lags = correlation_lags(processed_signal.size, reference_signal.size, mode="same")
    delay = lags[np.argmax(corr)]

    if delay > 0:
        # Output delayed relative to the reference
        processed_signal = np.concatenate(
            (processed_signal[delay:processed_n], np.zeros(delay))
        )
    else:
        # Output advanced relative to the reference
        processed_signal = np.concatenate(
            (np.zeros(-delay), processed_signal[: processed_n + delay])
        )

    return processed_signal


def find_silence_segments(
    signal: np.ndarray, sample_rate: int, min_silence_length: float = 1
) -> tuple[list, list]:
    """Find silence segments in signal

    Args:
        signal (np.ndarray): Input signal
        sample_rate (int): Sample rate in Hz
        min_silence_length (float): Minimum length of silence in
            seconds classify silence or non_silence segment.
            Defaults to 1.
    Returns:
        tuple[list, list]: Silence and non-silence segments
    """
    # Find the start and end of the noiseless reference sequence
    reference_abs = np.abs(signal)
    reference_max = np.max(reference_abs)
    threshold = 0.001 * reference_max

    # Find silence frames
    silence = np.where(np.abs(signal) < threshold)[0]
    silence = np.split(silence, np.where(np.diff(silence) != 1)[0] + 1)
    silence = [
        [group[0], group[-1]]
        for group in silence
        if len(group) > sample_rate * min_silence_length
    ]

    # find silence segments
    non_silence = []
    for start, end in zip(silence[:-1], silence[1:]):
        non_silence.append([start[-1] + 1, end[0] - 1])

    return silence, non_silence
