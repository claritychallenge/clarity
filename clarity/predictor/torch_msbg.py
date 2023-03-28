"""
An FIR-based torch implementation of approximated MSBG hearing loss model
"""
import json
from pathlib import Path

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

EPS = 1e-8
# old msbg matlab
# set RMS so that peak of output file so that no clipping occurs, set so that
# equiv0dBfileSPL > 100dB for LOUD input files
REF_RMS_DB = -31.2

# what RMS of INPUT speech file translates to in real world (unweighted)
CALIB_DB_SPL = 65

# what 0dB file signal would translate to in dB SPL:
# constant for cochlear_simulate function
EQUIV_0_DB_FILE_SPL = CALIB_DB_SPL - REF_RMS_DB

# clarity msbg
EQUIV_0_DB_SPL = 100
AHR = 20
EQUIV_0_DB_SPL = EQUIV_0_DB_SPL + AHR


def generate_key_percent(signal, thr_db, win_len):
    if win_len != np.floor(win_len):
        win_len = int(np.floor(win_len))
        print("\nGenerate_key_percent: \t Window length must be integer")

    signal_len = len(signal)
    expected = thr_db.copy()  # expected threshold
    non_zero = 10.0 ** ((expected - 30) / 10)  # put floor into histogram distribution

    n_frames = 0
    total_frames = int(np.floor(signal_len / win_len))
    every_db = np.zeros(total_frames)

    for ix in range(total_frames):
        start = ix * win_len
        this_sum = np.sum(signal[start : start + win_len] ** 2)  # sum of squares
        every_db[n_frames] = 10 * np.log10(non_zero + this_sum / win_len)
        n_frames += 1

    used_thr_db = expected.copy()

    # histogram should produce a two-peaked curve: thresh should be set in valley
    # between the two peaks, and set a bit above that, as it heads for main peak
    frame_idx = np.where(every_db >= expected)[0]
    valid_frames = len(frame_idx)
    key = np.zeros(valid_frames * win_len, dtype=int)

    # convert frame numbers into indices for signal
    for ix in range(valid_frames):
        key[ix * win_len : ix * win_len + win_len] = np.arange(
            frame_idx[ix] * win_len, frame_idx[ix] * win_len + win_len, dtype=int
        )
    return key, used_thr_db


def measure_rms(signal, sr, db_rel_rms):
    """Compute RMS level of a signal.

    Measures total power of all 10 msec frames that are above a user-specified threshold

    Args:
        signal: input signal
        sr: sampling rate
        db_rel_rms: threshold relative to first-stage rms (if it is made of a 2*1 array,
            second value over rules. only single value supported currently)

    Returns:
        tuple: The percentage of frames that are required to be tracked for measuring
        RMS (useful when DR compression changes histogram shape)
    """
    win_secs = 0.01
    # first RMS is of all signal
    first_stage_rms = np.sqrt(np.mean(signal**2))
    # use this RMS to generate key threshold to more accurate RMS
    key_thr_db = np.max([20 * np.log10(first_stage_rms) + db_rel_rms, -80])
    key, used_thr_db = generate_key_percent(
        signal, key_thr_db, int(np.round(win_secs * sr))
    )
    # active = 100.0 * len(key) / len(signal)
    rms = np.sqrt(np.mean(signal[key] ** 2))
    rel_db_thresh = used_thr_db - 20 * np.log10(rms)
    return rms, key, rel_db_thresh


def makesmearmat3(rl, ru, sr):
    fft_size = 512
    nyquist = int(fft_size // 2)
    f_nor = audfilt(1, 1, nyquist, sr)
    f_wid = audfilt(rl, ru, nyquist, sr)
    f_next = np.hstack([f_nor, np.zeros([nyquist, nyquist // 2])])

    for i in np.arange(nyquist // 2 + 1, nyquist + 1, dtype=int):
        f_next[i - 1, nyquist : np.min([2 * i - 1, 3 * nyquist // 2])] = np.flip(
            f_nor[
                i - 1, np.max([1, 2 * i - 3 * nyquist // 2]) - 1 : (2 * i - nyquist - 1)
            ]
        )
    f_smear = np.linalg.lstsq(f_next, f_wid)[
        0
    ]  # https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
    f_smear = f_smear[:nyquist, :]

    return f_smear


def audfilt(rl, ru, size, sr):
    """Calculate an auditory filter array

    Args:
        rl: broadening factor on the lower side
        ru: broadening factor on the upper side
        size:
        sr:

    Returns:
        np.ndarray
    """
    aud_filter = np.zeros([size, size])
    aud_filter[0, 0] = 1.0
    aud_filter[0, 0] = aud_filter[0, 0] / ((rl + ru) / 2)

    g = np.zeros(size)
    for i in np.arange(2, size + 1, 1, dtype=int):
        f_hz = (i - 1) * sr / (2 * size)
        erb_hz = 24.7 * ((f_hz * 0.00437) + 1)
        pl = 4 * f_hz / (erb_hz * rl)
        pu = 4 * f_hz / (erb_hz * ru)
        j = np.arange(1, i, dtype=int)
        g[j - 1] = np.abs((i - j) / (i - 1))
        aud_filter[i - 1, j - 1] = (1 + (pl * g[j - 1])) * np.exp(-pl * g[j - 1])
        j = np.arange(i, size + 1, dtype=int)
        g[j - 1] = np.abs((i - j) / (i - 1))
        aud_filter[i - 1, j - 1] = (1 + (pu * g[j - 1])) * np.exp(-pu * g[j - 1])
        aud_filter[i - 1, :] = aud_filter[i - 1, :] / (erb_hz * (rl + ru) / (2 * 24.7))
    return aud_filter


class MSBGHearingModel(nn.Module):
    def __init__(
        self,
        audiogram,
        audiometric,
        sr=44100,
        spl_cali=True,
        src_position="ff",
        kernel_size=1025,
        device=None,
    ):
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

        audiogram = np.array(audiogram)
        # audiometric = np.array([250, 500, 1000, 2000, 4000, 6000])
        audiometric = np.array(audiometric)
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
        cochlear_filter_forward = firwin2(
            n_wdw + 1, hz_used / nyquist, corrn_forward, window=("kaiser", 4)
        )
        cochlear_filter_backward = firwin2(
            n_wdw + 1, hz_used / nyquist, corrn_backward, window=("kaiser", 4)
        )
        self.cochlear_padding = len(cochlear_filter_forward) // 2
        self.cochlear_filter_forward = (
            torch.tensor(
                cochlear_filter_forward, dtype=torch.float32, device=self.device
            )
            .unsqueeze(0)
            .unsqueeze(1)
        )
        self.cochlear_filter_backward = (
            torch.tensor(
                cochlear_filter_backward, dtype=torch.float32, device=self.device
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
            f_smear = makesmearmat3(4, 2, sr)
        elif severe_not_moderate == 0:
            f_smear = makesmearmat3(2.4, 1.6, sr)
        elif severe_not_moderate == -1:
            f_smear = makesmearmat3(1.6, 1.1, sr)
        elif severe_not_moderate == -2:
            f_smear = makesmearmat3(1.001, 1.001, sr)

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

    def measure_rms(self, wav):
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

    def calibrate_spl(self, x):
        if self.spl_cali:
            level_re_fs = 10 * torch.log10(
                torch.mean(x**2, dim=1, keepdim=True) + EPS
            )
            level_db_spl = EQUIV_0_DB_SPL + level_re_fs
            rms = self.measure_rms(x)
            change_db = level_db_spl - (EQUIV_0_DB_SPL + 20 * torch.log10(rms + EPS))
            x = x * 10 ** (0.05 * change_db)
        return x

    def src_to_cochlea_filt(self, x, cochlear_filter):
        return F.conv1d(x, cochlear_filter, padding=self.cochlear_padding)

    def smear(self, x):
        """Padding issue needs to be worked out"""
        length = x.shape[2]
        x = x.view(x.shape[0], x.shape[2])
        spec = torch.stft(
            x,
            n_fft=self.smear_nfft,
            hop_length=self.smear_hop_len,
            win_length=self.smear_win_len,
            window=self.smear_window,
        )
        power = torch.square(spec[:, : self.smear_nfft // 2, :, 0]) + torch.square(
            spec[:, : self.smear_nfft // 2, :, 1]
        )
        mag = torch.sqrt(power + EPS).unsqueeze(-1)
        phasor = spec[:, : self.smear_nfft // 2, :, :] / (mag + EPS)

        smeared_power = (
            torch.matmul(power.transpose(-1, -2), self.f_smear.transpose(0, 1))
            .transpose(-1, -2)
            .unsqueeze(-1)
            + EPS
        )
        smeared_power = torch.clamp(smeared_power, min=0)
        smeared_spec_nyquist = torch.sqrt(smeared_power + EPS) * phasor
        smeared_spec_mid = torch.zeros(
            [smeared_power.shape[0], 1, smeared_power.shape[2], 2],
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

    def recruitment(self, x):
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
                envelope_out, min=EPS, max=self.envelope_max[ixch]
            )
            gain = (envelope_out / self.envelope_max[ixch]) ** self.expansion_m1[ixch]
            outputs.append(gain * pass_n_cali)

        y = torch.stack(outputs, dim=-1).sum(dim=-1)
        y = y * self.recruitment_out_coef
        return y

    def recruitment_fir(self, x):
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

    def forward(self, x):
        x = self.calibrate_spl(x)
        x = x.unsqueeze(1)
        x = self.src_to_cochlea_filt(x, self.cochlear_filter_forward)
        x = self.smear(x)
        # x = self.recruitment(x)
        x = self.recruitment_fir(x)
        y = self.src_to_cochlea_filt(x, self.cochlear_filter_backward)
        return y.squeeze(1)


class torchloudnorm(nn.Module):
    def __init__(
        self,
        sr=44100,
        norm_lufs=-36,
        kernel_size=1025,
        block_size=0.4,
        overlap=0.75,
        gamma_a=-70,
    ):
        super().__init__()
        self.sr = sr
        self.norm_lufs = norm_lufs
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # for frequency weighting filters - account for the acoustic respose
        # of the head and auditory system
        pyln_high_shelf_b = np.array([1.53090959, -2.65116903, 1.16916686])
        pyln_high_shelf_a = np.array([1.0, -1.66375011, 0.71265753])

        # fir high_shelf
        w_high_shelf, h_high_shelf = freqz(pyln_high_shelf_b, pyln_high_shelf_a, fs=sr)
        freq_high_shelf = np.append(w_high_shelf, sr / 2)
        gain_high_shelf = np.append(np.abs(h_high_shelf), np.abs(h_high_shelf)[-1])
        fir_high_shelf = firwin2(kernel_size, freq_high_shelf, gain_high_shelf, fs=sr)

        # fir high_pass
        fc_high_pass = 38.0
        fir_high_pass = firwin(kernel_size, fc_high_pass, pass_zero="highpass", fs=sr)

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
        self.frame_size = int(block_size * sr)
        self.frame_shift = int(block_size * sr * (1 - overlap))
        self.unfold = torch.nn.Unfold(
            (1, self.frame_size), stride=(1, self.frame_shift)
        )
        self.gamma_a = gamma_a

    def apply_filter(self, x):
        x = F.conv1d(x, self.high_shelf, padding=self.padding)
        x = F.conv1d(x, self.high_pass, padding=self.padding)
        return x

    def integrated_loudness(self, x):
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

    def normalize_loudness(self, x, lufs):
        delta_loudness = self.norm_lufs - lufs
        gain = torch.pow(10, delta_loudness / 20)
        return gain * x

    def forward(self, x):
        loudness = self.integrated_loudness(x.unsqueeze(1))
        y = self.normalize_loudness(x, loudness)
        return y
