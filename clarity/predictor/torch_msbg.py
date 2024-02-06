"""
An FIR-based torch implementation of approximated MSBG hearing loss model
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.signal import ellip, firwin, firwin2, freqz
from torch import nn

from clarity.evaluator.msbg.msbg_utils import (
    DF_ED,
    FF_ED,
    HZ,
    ITU_ERP_DRP,
    ITU_HZ,
    MIDEAR,
)
from clarity.evaluator.msbg.smearing import make_smear_mat3

EPS = 1e-8
# old msbg matlab
# set RMS so that peak of output file so that no clipping occurs, set so that
# equiv0dBfileSPL > 100dB for LOUD input files
REF_RMS_DB: Final = -31.2

# what RMS of INPUT speech file translates to in real world (unweighted)
CALIB_DB_SPL: Final = 65

# what 0dB file signal would translate to in dB SPL:
# constant for cochlea_simulate function
EQUIV_0_DB_FILE_SPL: Final = CALIB_DB_SPL - REF_RMS_DB

# clarity msbg
AHR: Final = 20
EQUIV_0_DB_SPL: Final = 100 + AHR


class MSBGHearingModel(nn.Module):
    def __init__(
        self,
        audiogram: np.ndarray,
        audiometric: np.ndarray,
        sr: int = 44100,
        spl_cali: bool = True,
        src_position: str = "ff",
        kernel_size: int = 1025,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.spl_cali = spl_cali
        self.src_position = src_position
        self.kernel_size = kernel_size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # settings for audiogram

        audiogram = np.append(audiogram, audiogram[-1])
        audiometric = np.append(audiometric, 16000)
        audiogram = np.append(audiogram[0], audiogram)
        audiometric = np.append(125, audiometric)

        audiogram_cfs = (
            np.array([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 14, 16])
            * 1000
        )
        interp_f = interp1d(audiometric, audiogram)
        audiogram = interp_f(audiogram_cfs)

        # settings for src_to_cochlea_filt

        if src_position == "ff":
            src_corrn = FF_ED
        elif src_position == "df":
            src_corrn = DF_ED
        elif src_position == "ITU":
            interf_itu = interp1d(ITU_HZ, ITU_ERP_DRP)
            src_corrn = interf_itu(HZ)
        nyquist = sr / 2
        ixf_useful = np.where(HZ < nyquist)[0]
        hz_used = np.append(HZ[ixf_useful], nyquist)
        corrn = src_corrn - MIDEAR
        interf_corrn = interp1d(HZ, corrn)
        last_corrn = interf_corrn(nyquist)
        corrn_used = np.append(corrn[ixf_useful], last_corrn)
        corrn_forward = 10 ** (0.05 * corrn_used)
        corrn_backward = 10 ** (0.05 * -1 * corrn_used)
        n_wdw = int(2 * np.floor((sr / 16e3) * 368 / 2))

        cochlea_filter_forward = firwin2(
            n_wdw + 1, hz_used / nyquist, corrn_forward, window=("kaiser", 4)
        )
        cochlea_filter_backward = firwin2(
            n_wdw + 1, hz_used / nyquist, corrn_backward, window=("kaiser", 4)
        )
        self.cochlea_padding = len(cochlea_filter_forward) // 2
        self.cochlea_filter_forward = (
            torch.tensor(
                cochlea_filter_forward, dtype=torch.float32, device=self.device
            )
            .unsqueeze(0)
            .unsqueeze(1)
        )
        self.cochlea_filter_backward = (
            torch.tensor(
                cochlea_filter_backward, dtype=torch.float32, device=self.device
            )
            .unsqueeze(0)
            .unsqueeze(1)
        )

        # Settings for smearing

        catch_up = 105.0  # dBHL where impaired catches up with normal

        # recruitment simulation comes with 3 degrees of broadening of auditory filters:
        # different set of centre freqs between simulations.
        # check and categorise audiogram: currently ALWAYS recruit with x2 broadening:
        # it's the smearing that changes

        impaired_freqs = np.where((audiogram_cfs >= 2000) & (audiogram_cfs <= 8000))[0]
        impaired_degree = np.mean(audiogram[impaired_freqs])

        # impairment degree affects smearing simulation, and now recruitment,
        # (assuming we do not have too much SEVERE losses present)

        current_dir = Path(__file__).parent
        gtf_dir = current_dir / "../evaluator/msbg/msbg_hparams"
        if impaired_degree > 56:
            severe_not_moderate = 1
            gt4_bank_file = gtf_dir / "GT4FBank_Brd3.0E_Spaced2.3E_44100Fs.json"
            bw_broaden_coef = 3
        elif impaired_degree > 35:
            severe_not_moderate = 0
            gt4_bank_file = gtf_dir / "GT4FBank_Brd2.0E_Spaced1.5E_44100Fs.json"
            bw_broaden_coef = 2
        elif impaired_degree > 15:
            severe_not_moderate = -1
            gt4_bank_file = gtf_dir / "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs.json"
            bw_broaden_coef = 1
        else:
            severe_not_moderate = -2
            gt4_bank_file = gtf_dir / "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs.json"
            bw_broaden_coef = 1
        # gt4_bank = loadmat(gt4_bank_file)
        with gt4_bank_file.open("r", encoding="utf-8") as fp:
            gt4_bank = json.load(fp)

        if severe_not_moderate > 0:
            f_smear = make_smear_mat3(4, 2, sr)
        elif severe_not_moderate == 0:
            f_smear = make_smear_mat3(2.4, 1.6, sr)
        elif severe_not_moderate == -1:
            f_smear = make_smear_mat3(1.6, 1.1, sr)
        elif severe_not_moderate == -2:
            f_smear = make_smear_mat3(1.001, 1.001, sr)

        self.smear_nfft = 512
        self.smear_win_len = 256
        self.smear_hop_len = 64
        smear_window = (
            0.5
            - 0.5
            * np.cos(
                2
                * np.pi
                * (np.arange(1, self.smear_win_len + 1) - 0.5)
                / self.smear_win_len
            )
        ) / np.sqrt(1.5)
        self.smear_window = torch.tensor(
            smear_window, dtype=torch.float32, device=self.device
        )
        self.f_smear = torch.tensor(f_smear, dtype=torch.float32, device=self.device)

        """ settings for recruitment"""
        cf_expansion = 0 * np.array(gt4_bank["GTn_CentFrq"])
        eq_loud_db = 0 * np.array(gt4_bank["GTn_CentFrq"])
        for ix_cfreq in range(len(gt4_bank["GTn_CentFrq"])):
            if gt4_bank["GTn_CentFrq"][ix_cfreq] < audiogram_cfs[0]:
                cf_expansion[ix_cfreq] = catch_up / (catch_up - audiogram[0])
            elif gt4_bank["GTn_CentFrq"][ix_cfreq] > audiogram_cfs[-1]:
                cf_expansion[ix_cfreq] = catch_up / (catch_up - audiogram[-1])
            else:
                interp_audiogram = interp1d(audiogram_cfs, audiogram)
                audiog_cf = interp_audiogram(gt4_bank["GTn_CentFrq"][ix_cfreq])
                cf_expansion[ix_cfreq] = catch_up / (catch_up - audiog_cf)
            eq_loud_db[ix_cfreq] = catch_up

        self.n_chans = gt4_bank["NChans"]
        self.gtn_denoms = torch.tensor(
            gt4_bank["GTn_denoms"], dtype=torch.float32, device=self.device
        )
        self.gtn_nums = torch.tensor(
            gt4_bank["GTn_nums"], dtype=torch.float32, device=self.device
        )
        self.hp_denoms = torch.tensor(
            gt4_bank["HP_denoms"], dtype=torch.float32, device=self.device
        )
        self.hp_nums = torch.tensor(
            gt4_bank["HP_nums"], dtype=torch.float32, device=self.device
        )
        self.ngamma = int(gt4_bank["NGAMMA"])
        self.gtn_delays = gt4_bank["GTnDelays"]
        self.start_2_pole_hp = gt4_bank["Start2PoleHP"]

        erbn_centre_freq = gt4_bank["ERBn_CentFrq"]
        chan_lpf_b = []
        chan_lpf_a = []
        fir_lpf = []
        for ixch in range(self.n_chans):
            fc_envelope = (30 / 40) * np.min([100, erbn_centre_freq[ixch]])
            chan_lpf_b_ch, chan_lpf_a_ch = ellip(
                2, 0.25, 35, fc_envelope / (self.sr / 2)
            )
            chan_lpf_b.append(chan_lpf_b_ch)
            chan_lpf_a.append(chan_lpf_a_ch)
            fir_lpf_ch = firwin(
                self.kernel_size, fc_envelope / (self.sr / 2), pass_zero="lowpass"
            ) / np.sqrt(
                2
            )  # sqrt(2) is for the consistency with IIR
            fir_lpf.append(fir_lpf_ch)
        self.chan_lpf_b = torch.tensor(
            np.array(chan_lpf_b), dtype=torch.float32, device=self.device
        )
        self.chan_lpf_a = torch.tensor(
            np.array(chan_lpf_a), dtype=torch.float32, device=self.device
        )
        self.fir_lpf = torch.tensor(
            np.array(fir_lpf), dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        self.expansion_m1 = torch.tensor(
            cf_expansion - 1, dtype=torch.float32, device=self.device
        )
        # self.envlp_max = torch.tensor(10 ** (0.05 * (eq_loud_db - equiv0dBfileSPL)),
        # dtype=torch.float32, device=self.device)
        self.envelope_max = torch.tensor(
            10 ** (0.05 * (eq_loud_db - EQUIV_0_DB_SPL)),
            dtype=torch.float32,
            device=self.device,
        )

        recombination_db = gt4_bank["Recombination_dB"]
        self.recruitment_out_coef = torch.tensor(
            10 ** (-0.05 * recombination_db), dtype=torch.float32, device=self.device
        )

        "settings for FIR Gammatone Filters"
        gt_cfreq = np.array(gt4_bank["GTn_CentFrq"])
        gt_bw = np.array(gt4_bank["ERBn_CentFrq"]) * 1.1019 * bw_broaden_coef

        self.padding = (self.kernel_size - 1) // 2
        n_lin = torch.linspace(
            0, self.kernel_size - 1, self.kernel_size, device=self.device
        )
        window_ = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size)
        n_ = (
            torch.arange(
                0, self.kernel_size, dtype=torch.float32, device=self.device
            ).view(1, -1)
            / self.sr
        )
        center_hz = (
            torch.tensor(
                gt_cfreq / self.sr, dtype=torch.float32, device=self.device
            ).view(-1, 1)
            * self.sr
        )
        f_times_t = torch.matmul(center_hz, n_)
        carrier = torch.cos(2 * np.pi * f_times_t)
        carrier_sin = torch.sin(2 * np.pi * f_times_t)
        band_hz = (
            torch.tensor(gt_bw / self.sr, dtype=torch.float32, device=self.device).view(
                -1, 1
            )
            * self.sr
        )
        b_times_t = torch.matmul(band_hz, n_)
        kernel = torch.pow(n_, 4 - 1) * torch.exp(-2 * np.pi * b_times_t)
        gammatone = kernel * carrier

        self.peaks = torch.argmax(gammatone, dim=1)  # for gammatone delay calibration

        gammatone_sin = kernel * carrier_sin
        filters = (gammatone * window_).view(self.n_chans, 1, self.kernel_size)
        # To get the normalised amplitude
        filters = filters.squeeze(1).cpu().numpy()
        fr_max = np.zeros(self.n_chans)
        for i in range(self.n_chans):
            fr = np.abs(fft(filters[i]))
            fr_ = fr[: int(self.kernel_size / 2)]
            fr_max[i] = np.max(fr_)
        amp = torch.tensor(fr_max, dtype=torch.float32, device=self.device)
        gammatone = gammatone / amp.unsqueeze(1)
        gammatone_sin = gammatone_sin / amp.unsqueeze(1)
        self.gt_fir = (gammatone * window_).view(self.n_chans, 1, self.kernel_size)
        self.gt_fir_sin = (gammatone_sin * window_).view(
            self.n_chans, 1, self.kernel_size
        )

        "settings for spl calibration"
        win_sec = 0.01
        self.db_relative_rms = -12
        self.win_len = int(self.sr * win_sec)

    def measure_rms(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute RMS level of a signal.

        Measures total power of all 10 msec frames that are above a specified
          threshold of db_relative_rms

        Args:
            wav: input signal

        Returns:
            RMS level in dB
        """
        bs = wav.shape[0]
        average_rms = torch.sqrt(torch.mean(wav**2, dim=1) + EPS)
        threshold_db = 20 * torch.log10(average_rms + EPS) + self.db_relative_rms

        num_frames = wav.shape[1] // self.win_len
        wav_reshaped = torch.reshape(
            wav[:, : num_frames * self.win_len], [bs, num_frames, self.win_len]
        )
        db_frames = 10 * torch.log10(torch.mean(wav_reshaped**2, dim=2) + EPS)

        key_frames = (
            torch.where(
                db_frames > threshold_db.unsqueeze(1),
                torch.tensor(1, dtype=torch.float32, device=self.device),
                torch.tensor(0, dtype=torch.float32, device=self.device),
            )
            .unsqueeze(-1)
            .repeat([1, 1, self.win_len])
            .reshape([bs, num_frames * self.win_len])
        )
        key_rms = torch.sqrt(
            torch.sum((wav[:, : num_frames * self.win_len] * key_frames) ** 2, dim=1)
            / (torch.sum(key_frames, dim=1) + EPS)
            + EPS
        )
        return key_rms.unsqueeze(1)

    def calibrate_spl(self, x: torch.Tensor) -> torch.Tensor:
        if self.spl_cali:
            level_re_sample_rate = 10 * torch.log10(
                torch.mean(x**2, dim=1, keepdim=True) + EPS
            )
            level_db_spl = EQUIV_0_DB_SPL + level_re_sample_rate
            rms = self.measure_rms(x)
            change_db = level_db_spl - (EQUIV_0_DB_SPL + 20 * torch.log10(rms + EPS))
            x = x * 10 ** (0.05 * change_db)
        return x

    def src_to_cochlea_filt(
        self, x: torch.Tensor, cochlea_filter: torch.Tensor
    ) -> torch.Tensor:
        return F.conv1d(x, cochlea_filter, padding=self.cochlea_padding)

    def smear(self, x: torch.Tensor) -> torch.Tensor:
        """Padding issue needs to be worked out"""
        length = x.shape[2]
        x = x.view(x.shape[0], x.shape[2])
        spec = torch.stft(
            x,
            n_fft=self.smear_nfft,
            hop_length=self.smear_hop_len,
            win_length=self.smear_win_len,
            window=self.smear_window,
            return_complex=True,
        )

        mag = torch.abs(spec[:, : self.smear_nfft // 2, :])
        power = torch.square(mag)

        phasor = spec[:, : self.smear_nfft // 2, :] / (mag + EPS)
        smeared_power = (
            torch.matmul(
                power.transpose(-1, -2), self.f_smear.transpose(0, 1)
            ).transpose(-1, -2)
            + EPS
        )
        smeared_power = torch.clamp(smeared_power, min=0)
        smeared_spec_nyquist = torch.sqrt(smeared_power + EPS) * phasor
        smeared_spec_mid = torch.zeros(
            [smeared_power.shape[0], 1, smeared_power.shape[2]],
            dtype=torch.float32,
            device=self.device,
        )

        smeared_spec = torch.cat([smeared_spec_nyquist, smeared_spec_mid], dim=1)

        smeared_wav = torch.istft(
            smeared_spec,
            n_fft=self.smear_nfft,
            hop_length=self.smear_hop_len,
            win_length=self.smear_win_len,
            window=self.smear_window,
            length=length,
        )
        return smeared_wav.unsqueeze(1)

    def recruitment(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[-1]
        ixhp = 0
        outputs = []
        for ixch in range(self.n_chans):
            # Gammaton filtering
            pass_n = torchaudio.functional.lfilter(
                x, self.gtn_denoms[ixch, :], self.gtn_nums[ixch, :]
            )
            for _ixg in range(self.ngamma - 1):
                pass_n = torchaudio.functional.lfilter(
                    pass_n, self.gtn_denoms[ixch, :], self.gtn_nums[ixch, :]
                )
            dly = self.gtn_delays[ixch]

            pass_n_cali = torch.zeros_like(pass_n)
            pass_n_cali[:, :, : n_samples - dly] = pass_n[:, :, dly:n_samples]
            # Tail control
            if ixch >= self.start_2_pole_hp:
                ixhp += 1
                pass_n_cali = torchaudio.functional.lfilter(
                    pass_n_cali, self.hp_denoms[ixhp - 1, :], self.hp_nums[ixhp - 1, :]
                )

            # Get the envelope
            envelope_out = torchaudio.functional.lfilter(
                torch.abs(pass_n_cali),
                self.chan_lpf_a[ixch, :],
                self.chan_lpf_b[ixch, :],
            )
            envelope_out = torch.flip(envelope_out, dims=[-1])
            envelope_out = torchaudio.functional.lfilter(
                envelope_out, self.chan_lpf_a[ixch, :], self.chan_lpf_b[ixch, :]
            )
            envelope_out = torch.flip(envelope_out, dims=[-1])

            envelope_out = torch.clamp(
                envelope_out, min=EPS, max=float(self.envelope_max[ixch])
            )
            gain = (envelope_out / self.envelope_max[ixch]) ** self.expansion_m1[ixch]
            outputs.append(gain * pass_n_cali)

        y = torch.stack(outputs, dim=-1).sum(dim=-1)
        y = y * self.recruitment_out_coef
        return y

    def recruitment_fir(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[-1]
        x = x.repeat([1, self.n_chans, 1])
        real = F.conv1d(
            x, self.gt_fir, bias=None, padding=self.padding, groups=self.n_chans
        )
        imag = F.conv1d(
            x, self.gt_fir_sin, bias=None, padding=self.padding, groups=self.n_chans
        )
        real_cali = torch.zeros_like(real)
        imag_cali = torch.zeros_like(imag)
        for i in range(self.n_chans):
            real_cali[:, i, : n_samples - self.peaks[i]] = real[
                :, i, self.peaks[i] : n_samples
            ]
            imag_cali[:, i, : n_samples - self.peaks[i]] = imag[
                :, i, self.peaks[i] : n_samples
            ]

        env = torch.sqrt(real_cali * real_cali + imag_cali * imag_cali + EPS)
        env = F.conv1d(
            env, self.fir_lpf, bias=None, padding=self.padding, groups=self.n_chans
        )

        env_max = self.envelope_max.unsqueeze(0).unsqueeze(-1).repeat([1, 1, n_samples])
        gain = torch.clamp(env / env_max, min=EPS, max=1)
        gain = gain ** self.expansion_m1.unsqueeze(0).unsqueeze(-1).repeat(
            [1, 1, n_samples]
        )
        y = torch.sum(gain * real_cali, dim=1, keepdim=True)
        y = y * self.recruitment_out_coef
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.calibrate_spl(x)
        x = x.unsqueeze(1)
        x = self.src_to_cochlea_filt(x, self.cochlea_filter_forward)
        x = self.smear(x)
        # x = self.recruitment(x)
        x = self.recruitment_fir(x)
        y = self.src_to_cochlea_filt(x, self.cochlea_filter_backward)
        return y.squeeze(1)


class torchloudnorm(nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        norm_lufs: int = -36,
        kernel_size: int = 1025,
        block_size: float = 0.4,
        overlap: float = 0.75,
        gamma_a: int = -70,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.norm_lufs = norm_lufs
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # for frequency weighting filters - account for the acoustic respose
        # of the head and auditory system
        pyln_high_shelf_b = np.array([1.53090959, -2.65116903, 1.16916686])
        pyln_high_shelf_a = np.array([1.0, -1.66375011, 0.71265753])

        # fir high_shelf
        w_high_shelf, h_high_shelf = freqz(
            pyln_high_shelf_b, pyln_high_shelf_a, fs=sample_rate
        )
        freq_high_shelf = np.append(w_high_shelf, sample_rate / 2)
        gain_high_shelf = np.append(np.abs(h_high_shelf), np.abs(h_high_shelf)[-1])
        fir_high_shelf = firwin2(
            kernel_size, freq_high_shelf, gain_high_shelf, fs=sample_rate
        )

        # fir high_pass
        fc_high_pass = 38.0
        fir_high_pass = firwin(
            kernel_size, fc_high_pass, pass_zero="highpass", fs=sample_rate
        )

        self.high_shelf = (
            torch.tensor(fir_high_shelf, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(1)
        )
        self.high_pass = (
            torch.tensor(fir_high_pass, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(1)
        )

        "rms measurement"
        self.frame_size = int(block_size * sample_rate)
        self.frame_shift = int(block_size * sample_rate * (1 - overlap))
        self.unfold = torch.nn.Unfold(
            (1, self.frame_size), stride=(1, self.frame_shift)
        )
        self.gamma_a = gamma_a

    def apply_filter(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv1d(x, self.high_shelf, padding=self.padding)
        x = F.conv1d(x, self.high_pass, padding=self.padding)
        return x

    def integrated_loudness(self, x: torch.Tensor) -> torch.Tensor:
        x = self.apply_filter(x)
        x_unfold = self.unfold(x.unsqueeze(2))

        z = (
            torch.sum(x_unfold**2, dim=1) / self.frame_size
        )  # mean square for each frame
        el = -0.691 + 10 * torch.log10(z + EPS)

        idx_a = torch.where(el > self.gamma_a, 1, 0)
        z_ave_gated_a = torch.sum(z * idx_a, dim=1, keepdim=True) / (
            torch.sum(idx_a, dim=1, keepdim=True) + 1e-8
        )
        gamma_r = -0.691 + 10 * torch.log10(z_ave_gated_a + EPS) - 10

        idx_r = torch.where(el > gamma_r, 1, 0)
        idx_a_r = idx_a * idx_r
        z_ave_gated_a_r = torch.sum(z * idx_a_r, dim=1, keepdim=True) / (
            torch.sum(idx_a_r, dim=1, keepdim=True) + 1e-8
        )
        lufs = -0.691 + 10 * torch.log10(z_ave_gated_a_r + EPS)  # loudness
        return lufs

    def normalize_loudness(self, x: torch.Tensor, lufs: torch.Tensor) -> torch.Tensor:
        delta_loudness = self.norm_lufs - lufs
        gain = torch.pow(10, delta_loudness / 20)
        return gain * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loudness = self.integrated_loudness(x.unsqueeze(1))
        y = self.normalize_loudness(x, loudness)
        return y
