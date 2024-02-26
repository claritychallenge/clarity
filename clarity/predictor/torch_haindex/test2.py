import numpy as np
import matplotlib.pyplot as plt
import math

### UTILITY FUNCTIONS ###


def erb_scale_2_freq_hz(freq_erb):
    # Convert frequency on ERB scale to frequency in Hertz
    freq_hz = (np.exp(freq_erb / 9.265) - 1) * 24.7 * 9.265
    return freq_hz


def freq_hz_2_erb_scale(freq_hz):
    # Convert frequency in Hertz to frequency on ERB scale
    freq_erb = 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))
    return freq_erb


def normalize_filters(filterbank):
    # Normalizes a filterbank such that all filters
    # have the same root mean square (RMS).
    rms_per_filter = np.sqrt(np.mean(np.square(filterbank), axis=1))
    rms_normalization_values = 1.0 / (rms_per_filter / np.amax(rms_per_filter))
    normalized_filterbank = filterbank * rms_normalization_values[:, np.newaxis]
    return normalized_filterbank


### GAMMATONE IMPULSE RESPONSE ###


def gammatone_impulse_response(
    samplerate_hz, length_in_seconds, center_freq_hz, phase_shift
):
    # Generate single parametrized gammatone filter
    p = 2  # filter order
    erb = 24.7 + 0.108 * center_freq_hz  # equivalent rectangular bandwidth
    divisor = (
        np.pi * math.factorial(2 * p - 2) * np.power(2, float(-(2 * p - 2)))
    ) / np.square(math.factorial(p - 1))
    b = erb / divisor  # bandwidth parameter
    a = 1.0  # amplitude. This is varied later by the normalization process.
    L = int(np.floor(samplerate_hz * length_in_seconds))
    t = np.linspace(1.0 / samplerate_hz, length_in_seconds, L)
    gammatone_ir = (
        a
        * np.power(t, p - 1)
        * np.exp(-2 * np.pi * b * t)
        * np.cos(2 * np.pi * center_freq_hz * t + phase_shift)
    )
    return gammatone_ir


### MP-GTF CONSTRUCTION ###


def generate_mpgtf(samplerate_hz, length_in_seconds, N):
    # Set parameters
    center_freq_hz_min = 100
    n_center_freqs = 24
    L = int(np.floor(samplerate_hz * length_in_seconds))

    # Initialize variables
    index = 0
    filterbank = np.zeros((N, L))
    current_center_freq_hz = center_freq_hz_min

    # Determine number of phase shifts per center frequency
    phase_pair_count = (
        np.ones(n_center_freqs) * np.floor(N / 2 / n_center_freqs)
    ).astype(int)
    remaining_phase_pairs = ((N - np.sum(phase_pair_count) * 2) / 2).astype(int)
    if remaining_phase_pairs > 0:
        phase_pair_count[:remaining_phase_pairs] = (
            phase_pair_count[:remaining_phase_pairs] + 1
        )

    # Generate all filters for each center frequencies
    for i in range(n_center_freqs):
        # Generate all filters for all phase shifts
        for phase_index in range(phase_pair_count[i]):
            # First half of filtes: phase_shifts in [0,pi)
            current_phase_shift = float(phase_index) / phase_pair_count[i] * np.pi
            filterbank[index, :] = gammatone_impulse_response(
                samplerate_hz,
                length_in_seconds,
                current_center_freq_hz,
                current_phase_shift,
            )
            index = index + 1

        # Second half of filtes: phase_shifts in [pi, 2*pi)
        filterbank[index : index + phase_pair_count[i], :] = -filterbank[
            index - phase_pair_count[i] : index, :
        ]

        # Prepare for next center frequency
        index = index + phase_pair_count[i]
        current_center_freq_hz = erb_scale_2_freq_hz(
            freq_hz_2_erb_scale(current_center_freq_hz) + 1
        )

    filterbank = normalize_filters(filterbank)
    return filterbank


### GENERATE AND PLOT ###


def generate_example_and_plot():
    # Set parameters
    N = 128  # Number of filters
    samplerate_in_hz = 8000
    filterlength_in_seconds = 0.002

    # Generate the MP-GTF
    filterbank = generate_mpgtf(samplerate_in_hz, filterlength_in_seconds, N)

    # Plot MP-GTF
    plt.figure(figsize=(10, 5))
    plt.imshow(filterbank, origin="lower", cmap="bwr")
    plt.xlabel("Time (in samples)")
    plt.ylabel("Filter index n")
    plt.show()


if __name__ == "__main__":
    generate_example_and_plot()
