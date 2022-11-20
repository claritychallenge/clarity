import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class AudiometricFIR(nn.Module):
    def __init__(self, sr=44100, nfir=220, device=None):
        super().__init__()
        if device is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.window_size = nfir + 1
        self.padding = nfir // 2

        aud = np.array([250, 500, 1000, 2000, 4000, 6000])
        aud_fv = np.append(np.append(0, aud), sr // 2)  # Audiometric frequency vector
        linear_fv = (
            np.linspace(0, nfir, nfir + 1) / nfir * sr // 2
        )  # linear frequency vector
        interval_freq = np.zeros([len(linear_fv), 2])
        interval_idx = np.zeros([len(linear_fv), 2], dtype=int)
        for i, linear_fv_i in enumerate(linear_fv):
            for j in range(len(aud_fv) - 1):
                if aud_fv[j] <= linear_fv_i < aud_fv[j + 1]:
                    interval_freq[i, 0] = aud_fv[j]
                    interval_freq[i, 1] = aud_fv[j + 1]
                    interval_idx[i, 0] = j
                    interval_idx[i, 1] = j + 1
        interval_freq[-1, 0] = aud_fv[-2]
        interval_freq[-1, 1] = aud_fv[-1]
        interval_idx[-1, 0] = len(aud_fv) - 2
        interval_idx[-1, 1] = len(aud_fv) - 1

        self.interval_idx = interval_idx
        x2_minus_x1 = interval_freq[:, 1] - interval_freq[:, 0]
        x_minus_x1 = linear_fv - interval_freq[:, 0]
        self.x2_minus_x1 = torch.tensor(
            x2_minus_x1, dtype=torch.float32, device=self.device
        )
        self.x_minus_x1 = torch.tensor(
            x_minus_x1, dtype=torch.float32, device=self.device
        )

        self.amp = nn.Parameter(
            torch.tensor(
                np.ones_like(aud_fv[1:-1]), dtype=torch.float32, device=self.device
            )
        )

    def forward(self, x):
        amp = torch.pow(10, torch.abs(self.amp) / 20.0)
        amp = torch.cat((torch.cat((amp[:1], amp)), amp[-1:]))
        y = amp[self.interval_idx]
        y2_minus_y1 = y[:, 1] - y[:, 0]
        y1 = y[:, 0]
        gain = y2_minus_y1 * self.x_minus_x1 / self.x2_minus_x1 + y1

        # firwin
        phase = torch.zeros_like(gain, device=self.device)
        gain = gain.unsqueeze(1)
        phase = phase.unsqueeze(1)
        magnitudes = torch.view_as_complex(torch.cat([gain, phase], dim=1).unsqueeze(0))
        impulse_response = torch.fft.irfft(magnitudes, dim=1)

        window_size = self.window_size + 1
        window = torch.hann_window(window_size, device=self.device)
        ir_size = int(impulse_response.shape[-1])
        half_idx = (window_size + 1) // 2
        padding = ir_size - window_size
        window = torch.cat(
            [
                window[half_idx:],
                torch.zeros([padding], device=self.device),
                window[:half_idx],
            ],
            dim=0,
        )
        impulse_response = impulse_response * window
        first_half_start = ir_size - (half_idx - 1)
        second_half_end = half_idx
        fir_filter = torch.cat(
            [
                impulse_response[:, first_half_start:],
                impulse_response[:, :second_half_end],
            ],
            dim=-1,
        ).unsqueeze(0)

        # x = x.unsqueeze(1)
        output = F.conv1d(x, fir_filter, padding=self.padding, bias=None)
        return output
