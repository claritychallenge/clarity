import numpy as np


class Compressor:
    def __init__(
        self,
        fs=44100,
        attack=5,
        release=20,
        threshold=1,
        attenuation=0.0001,
        rms_buffer_size=0.2,
        makeup_gain=1,
    ):
        self.fs = fs
        self.rms_buffer_size = rms_buffer_size
        self.set_attack(attack)
        self.set_release(release)
        self.threshold = threshold
        self.attenuation = attenuation
        self.eps = 1e-8
        self.makeup_gain = makeup_gain

        # window for computing rms
        self.win_len = int(self.rms_buffer_size * self.fs)
        self.window = np.ones(self.win_len)

    def set_attack(self, t_msec):
        t_sec = t_msec / 1000
        reciprocal_time = 1 / t_sec
        self.attack = reciprocal_time / self.fs

    def set_release(self, t_msec):
        t_sec = t_msec / 1000
        reciprocal_time = 1 / t_sec
        self.release = reciprocal_time / self.fs

    def process(self, signal):
        padded_signal = np.concatenate((np.zeros(self.win_len - 1), signal))
        rms = np.sqrt(
            np.convolve(padded_signal ** 2, self.window, mode="valid") / self.win_len
            + self.eps
        )
        comp_ratios = []
        curr_comp = 1
        for i in range(len(rms)):
            if rms[i] > self.threshold:
                temp_comp = (rms[i] * self.attenuation) + (
                    (1 - self.attenuation) * self.threshold
                )
                curr_comp = (curr_comp * (1 - self.attack)) + (temp_comp * self.attack)
            else:
                curr_comp = (1 * self.release) + curr_comp * (1 - self.release)
            comp_ratios.append(curr_comp)
        return signal * np.array(comp_ratios) * self.makeup_gain, rms, comp_ratios
