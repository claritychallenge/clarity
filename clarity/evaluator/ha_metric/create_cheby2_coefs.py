from scipy.signal import butter, cheby2
import json

order = 7
attenuation = 30  # sidelobe attenuation in dB


coefs_cheby2 = {}

for reference_freq_khz in [1000, 4000, 8000, 12000, 16000, 22050, 24000, 44100]:
    reference_freq_cut = 21 / reference_freq_khz / 1000
    a, b = cheby2(order, attenuation, reference_freq_cut)
    coefs_cheby2[reference_freq_khz] = {
        "a": a.tolist(),
        "b": b.tolist(),
    }

with open("precomputed/cheby2_coefs.json", "w") as f:
    json.dump(coefs_cheby2, f, indent=4)


coefs_middle_ear = {}

for reference_freq_khz in [24000]:
    coefs_middle_ear[reference_freq_khz] = {}
    butterworth_low_pass, low_pass = butter(1, 5000 / (0.5 * reference_freq_khz))
    butterworth_high_pass, high_pass = butter(
        2, 350 / (0.5 * reference_freq_khz), "high"
    )

    coefs_middle_ear[reference_freq_khz][
        "butterworth_low_pass"
    ] = butterworth_low_pass.tolist()
    coefs_middle_ear[reference_freq_khz]["low_pass"] = low_pass.tolist()
    coefs_middle_ear[reference_freq_khz][
        "butterworth_high_pass"
    ] = butterworth_high_pass.tolist()
    coefs_middle_ear[reference_freq_khz]["high_pass"] = high_pass.tolist()

    with open("precomputed/middle_ear_coefs.json", "w") as f:
        json.dump(coefs_middle_ear, f, indent=4)


coef_compress_basilar_membrane = {}
for fsamp in [24000]:
    flp = 800
    coef_compress_basilar_membrane[fsamp] = {}
    b, a = butter(1, flp / (0.5 * fsamp))
    coef_compress_basilar_membrane[fsamp]["a"] = a.tolist()
    coef_compress_basilar_membrane[fsamp]["b"] = b.tolist()

    with open("precomputed/compress_basilar_membrane_coefs.json", "w") as f:
        json.dump(coef_compress_basilar_membrane, f, indent=4)
