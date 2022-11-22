"""
An FIR-based torch implementation of approximated MSBG hearing loss model
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.signal import ellip, firwin, firwin2, freqz

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
ref_RMSdB = -31.2
calib_dBSPL = (
    65  # what RMS of INPUT speech file translates to in real world (unweighted)
)
equiv0dBfileSPL = (
    calib_dBSPL - ref_RMSdB
)  # what 0dB file signal would translate to in dB SPL: constant for cochlear_simulate function

# clarity msbg
equiv0dBSPL = 100
ahr = 20
equiv_0dB_SPL = equiv0dBSPL + ahr


def generate_key_percent(signal, thr_dB, winlen):
    if winlen != np.floor(winlen):
        winlen = np.int(np.floor(winlen))
        print("\nGenerate_key_percent: \t Window length must be integer")

    siglen = len(signal)
    expected = thr_dB.copy()  # expected threshold
    non_zero = 10.0 ** ((expected - 30) / 10)  # put floor into histogram distribution

    nframes = 0
    totframes = np.int(np.floor(siglen / winlen))
    every_dB = np.zeros(totframes)

    for ix in range(totframes):
        start = ix * winlen
        this_sum = np.sum(signal[start : start + winlen] ** 2)  # sum of squares
        every_dB[nframes] = 10 * np.log10(non_zero + this_sum / winlen)
        nframes += 1

    used_thr_dB = expected.copy()

    # histogram should produce a two-peaked curve: thresh should be set in valley
    # between the two peaks, and set a bit above that, as it heads for main peak
    frame_idx = np.where(every_dB >= expected)[0]
    valid_frames = len(frame_idx)
    key = np.zeros(valid_frames * winlen, dtype=np.int)

    # convert frame numbers into indices for signal
    for ix in range(valid_frames):
        key[ix * winlen : ix * winlen + winlen] = np.arange(
            frame_idx[ix] * winlen, frame_idx[ix] * winlen + winlen, dtype=np.int
        )
    return key, used_thr_dB


def measure_rms(signal, sr, dB_rel_rms):
    """Measures toatal power of all 10 msec frams that are above a user-specified threshold

    Args:
        signal: input signal
        sr: sampling rate
        dB_rel_rms: threshold relative to first-stage rms (if it is made of a 2*1 array, second value over rules.)
                    only single value supported currently
    Returns:
        tuple: The percentage of frames that are required to be tracked for measuring RMS (useful when DR compression
               changes histogram shape)
    """
    win_secs = 0.01
    # first RMS is of all signal
    first_stage_rms = np.sqrt(np.mean(signal**2))
    # use this RMS to generate key threshold to more accurate RMS
    key_thr_dB = np.max([20 * np.log10(first_stage_rms) + dB_rel_rms, -80])
    key, used_thr_dB = generate_key_percent(
        signal, key_thr_dB, np.int(np.round(win_secs * sr))
    )
    # active = 100.0 * len(key) / len(signal)
    rms = np.sqrt(np.mean(signal[key] ** 2))
    rel_dB_thresh = used_thr_dB - 20 * np.log10(rms)
    return rms, key, rel_dB_thresh


def makesmearmat3(rl, ru, sr):
    fftsize = 512
    nyquist = np.int(fftsize // 2)
    fnor = audfilt(1, 1, nyquist, sr)
    fwid = audfilt(rl, ru, nyquist, sr)
    fnext = np.hstack([fnor, np.zeros([nyquist, nyquist // 2])])

    for i in np.arange(nyquist // 2 + 1, nyquist + 1, dtype=np.int):
        fnext[i - 1, nyquist : np.min([2 * i - 1, 3 * nyquist // 2])] = np.flip(
            fnor[
                i - 1, np.max([1, 2 * i - 3 * nyquist // 2]) - 1 : (2 * i - nyquist - 1)
            ]
        )
    fsmear = np.linalg.lstsq(fnext, fwid)[
        0
    ]  # https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator
    fsmear = fsmear[:nyquist, :]

    return fsmear


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
    for i in np.arange(2, size + 1, 1, dtype=np.int):
        fhz = (i - 1) * sr / (2 * size)
        erbhz = 24.7 * ((fhz * 0.00437) + 1)
        pl = 4 * fhz / (erbhz * rl)
        pu = 4 * fhz / (erbhz * ru)
        j = np.arange(1, i, dtype=np.int)
        g[j - 1] = np.abs((i - j) / (i - 1))
        aud_filter[i - 1, j - 1] = (1 + (pl * g[j - 1])) * np.exp(-pl * g[j - 1])
        j = np.arange(i, size + 1, dtype=np.int)
        g[j - 1] = np.abs((i - j) / (i - 1))
        aud_filter[i - 1, j - 1] = (1 + (pu * g[j - 1])) * np.exp(-pu * g[j - 1])
        aud_filter[i - 1, :] = aud_filter[i - 1, :] / (erbhz * (rl + ru) / (2 * 24.7))
    return aud_filter


class MSBGHearingModel(nn.Module):
    def __init__(
        self,
        audiogram,
        audiometric,
        sr=44100,
        spl_cali=True,
        src_posn="ff",
        kernel_size=1025,
        device=None,
    ):
        super().__init__()
        self.sr = sr
        self.spl_cali = spl_cali
        self.src_posn = src_posn
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
        interpf = interp1d(audiometric, audiogram)
        audiogram = interpf(audiogram_cfs)

        # settings for src_to_cochlea_filt

        if src_posn == "ff":
            src_corrn = FF_ED
        elif src_posn == "df":
            src_corrn = DF_ED
        elif src_posn == "ITU":
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
        n_wdw = np.int(2 * np.floor((sr / 16e3) * 368 / 2))
        coch_filter_forward = firwin2(
            n_wdw + 1, hz_used / nyquist, corrn_forward, window=("kaiser", 4)
        )
        coch_filter_backward = firwin2(
            n_wdw + 1, hz_used / nyquist, corrn_backward, window=("kaiser", 4)
        )
        self.coch_padding = len(coch_filter_forward) // 2
        self.coch_filter_forward = (
            torch.tensor(coch_filter_forward, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(1)
        )
        self.coch_filter_backward = (
            torch.tensor(coch_filter_backward, dtype=torch.float32, device=self.device)
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

        current_dir = os.path.dirname(os.path.abspath(__file__))
        gtf_dir = os.path.join(current_dir, "../evaluator/msbg/msbg_hparams")
        if impaired_degree > 56:
            severe_not_moderate = 1
            GT4Bankfile = os.path.join(
                gtf_dir, "GT4FBank_Brd3.0E_Spaced2.3E_44100Fs.json"
            )
            bw_broaden_coef = 3
        elif impaired_degree > 35:
            severe_not_moderate = 0
            GT4Bankfile = os.path.join(
                gtf_dir, "GT4FBank_Brd2.0E_Spaced1.5E_44100Fs.json"
            )
            bw_broaden_coef = 2
        elif impaired_degree > 15:
            severe_not_moderate = -1
            GT4Bankfile = os.path.join(
                gtf_dir, "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs.json"
            )
            bw_broaden_coef = 1
        else:
            severe_not_moderate = -2
            GT4Bankfile = os.path.join(
                gtf_dir, "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs.json"
            )
            bw_broaden_coef = 1
        # GT4Bank = loadmat(GT4Bankfile)
        with open(GT4Bankfile, "r", encoding="utf-8") as fp:
            GT4Bank = json.load(fp)

        if severe_not_moderate > 0:
            fsmear = makesmearmat3(4, 2, sr)
        elif severe_not_moderate == 0:
            fsmear = makesmearmat3(2.4, 1.6, sr)
        elif severe_not_moderate == -1:
            fsmear = makesmearmat3(1.6, 1.1, sr)
        elif severe_not_moderate == -2:
            fsmear = makesmearmat3(1.001, 1.001, sr)

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
        self.fsmear = torch.tensor(fsmear, dtype=torch.float32, device=self.device)

        """ settings for recruitment"""
        cf_expnsn = 0 * np.array(GT4Bank["GTn_CentFrq"])
        eq_loud_db = 0 * np.array(GT4Bank["GTn_CentFrq"])
        for ix_cfreq in range(len(GT4Bank["GTn_CentFrq"])):
            if GT4Bank["GTn_CentFrq"][ix_cfreq] < audiogram_cfs[0]:
                cf_expnsn[ix_cfreq] = catch_up / (catch_up - audiogram[0])
            elif GT4Bank["GTn_CentFrq"][ix_cfreq] > audiogram_cfs[-1]:
                cf_expnsn[ix_cfreq] = catch_up / (catch_up - audiogram[-1])
            else:
                interp_audiog = interp1d(audiogram_cfs, audiogram)
                audiog_cf = interp_audiog(GT4Bank["GTn_CentFrq"][ix_cfreq])
                cf_expnsn[ix_cfreq] = catch_up / (catch_up - audiog_cf)
            eq_loud_db[ix_cfreq] = catch_up

        self.nchans = GT4Bank["NChans"]
        self.gtn_denoms = torch.tensor(
            GT4Bank["GTn_denoms"], dtype=torch.float32, device=self.device
        )
        self.gtn_nums = torch.tensor(
            GT4Bank["GTn_nums"], dtype=torch.float32, device=self.device
        )
        self.hp_denoms = torch.tensor(
            GT4Bank["HP_denoms"], dtype=torch.float32, device=self.device
        )
        self.hp_nums = torch.tensor(
            GT4Bank["HP_nums"], dtype=torch.float32, device=self.device
        )
        self.ngamma = int(GT4Bank["NGAMMA"])
        self.gtn_delays = GT4Bank["GTnDelays"]
        self.start2polehp = GT4Bank["Start2PoleHP"]

        erbn_centfrq = GT4Bank["ERBn_CentFrq"]
        chan_lpfB = []
        chan_lpfA = []
        fir_lpf = []
        for ixch in range(self.nchans):
            fc_envlp = (30 / 40) * np.min([100, erbn_centfrq[ixch]])
            chan_lpfB_ch, chan_lpfA_ch = ellip(2, 0.25, 35, fc_envlp / (self.sr / 2))
            chan_lpfB.append(chan_lpfB_ch)
            chan_lpfA.append(chan_lpfA_ch)
            fir_lpf_ch = firwin(
                self.kernel_size, fc_envlp / (self.sr / 2), pass_zero="lowpass"
            ) / np.sqrt(
                2
            )  # sqrt(2) is for the consistency with IIR
            fir_lpf.append(fir_lpf_ch)
        self.chan_lpfB = torch.tensor(
            np.array(chan_lpfB), dtype=torch.float32, device=self.device
        )
        self.chan_lpfA = torch.tensor(
            np.array(chan_lpfA), dtype=torch.float32, device=self.device
        )
        self.fir_lpf = torch.tensor(
            np.array(fir_lpf), dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        self.expnsn_m1 = torch.tensor(
            cf_expnsn - 1, dtype=torch.float32, device=self.device
        )
        # self.envlp_max = torch.tensor(10 ** (0.05 * (eq_loud_db - equiv0dBfileSPL)),
        # dtype=torch.float32, device=self.device)
        self.envlp_max = torch.tensor(
            10 ** (0.05 * (eq_loud_db - equiv_0dB_SPL)),
            dtype=torch.float32,
            device=self.device,
        )

        recombination_dB = GT4Bank["Recombination_dB"]
        self.recruitmnet_out_coef = torch.tensor(
            10 ** (-0.05 * recombination_dB), dtype=torch.float32, device=self.device
        )

        "settings for FIR Gammatone Filters"
        gt_cfreq = np.array(GT4Bank["GTn_CentFrq"])
        gt_bw = np.array(GT4Bank["ERBn_CentFrq"]) * 1.1019 * bw_broaden_coef

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
        filters = (gammatone * window_).view(self.nchans, 1, self.kernel_size)
        # To get the normalised amplitude
        filters = filters.squeeze(1).cpu().numpy()
        fr_max = np.zeros(self.nchans)
        for i in range(self.nchans):
            fr = np.abs(fft(filters[i]))
            fr_ = fr[: int(self.kernel_size / 2)]
            fr_max[i] = np.max(fr_)
        amp = torch.tensor(fr_max, dtype=torch.float32, device=self.device)
        gammatone = gammatone / amp.unsqueeze(1)
        gammatone_sin = gammatone_sin / amp.unsqueeze(1)
        self.gt_fir = (gammatone * window_).view(self.nchans, 1, self.kernel_size)
        self.gt_fir_sin = (gammatone_sin * window_).view(
            self.nchans, 1, self.kernel_size
        )

        "settings for spl calibration"
        win_sec = 0.01
        self.db_relative_rms = -12
        self.win_len = int(self.sr * win_sec)

    def measure_rms(self, wav):
        bs = wav.shape[0]
        ave_rms = torch.sqrt(torch.mean(wav**2, dim=1) + EPS)
        thr_db = 20 * torch.log10(ave_rms + EPS) + self.db_relative_rms

        num_frames = wav.shape[1] // self.win_len
        wav_reshaped = torch.reshape(
            wav[:, : num_frames * self.win_len], [bs, num_frames, self.win_len]
        )
        db_frames = 10 * torch.log10(torch.mean(wav_reshaped**2, dim=2) + EPS)

        key_frames = (
            torch.where(
                db_frames > thr_db.unsqueeze(1),
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
            levelreFS = 10 * torch.log10(torch.mean(x**2, dim=1, keepdim=True) + EPS)
            leveldBSPL = equiv_0dB_SPL + levelreFS
            rms = self.measure_rms(x)
            change_dB = leveldBSPL - (equiv_0dB_SPL + 20 * torch.log10(rms + EPS))
            x = x * 10 ** (0.05 * change_dB)
        return x

    def src_to_cochlea_filt(self, x, coch_filter):
        return F.conv1d(x, coch_filter, padding=self.coch_padding)

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
            torch.matmul(power.transpose(-1, -2), self.fsmear.transpose(0, 1))
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
        nsamps = x.shape[-1]
        ixhp = 0
        outputs = []
        for ixch in range(self.nchans):
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
            pass_n_cali[:, :, : nsamps - dly] = pass_n[:, :, dly:nsamps]
            # Tail control
            if ixch >= self.start2polehp:
                ixhp += 1
                pass_n_cali = torchaudio.functional.lfilter(
                    pass_n_cali, self.hp_denoms[ixhp - 1, :], self.hp_nums[ixhp - 1, :]
                )

            # Get the envelope
            envlp_out = torchaudio.functional.lfilter(
                torch.abs(pass_n_cali), self.chan_lpfA[ixch, :], self.chan_lpfB[ixch, :]
            )
            envlp_out = torch.flip(envlp_out, dims=[-1])
            envlp_out = torchaudio.functional.lfilter(
                envlp_out, self.chan_lpfA[ixch, :], self.chan_lpfB[ixch, :]
            )
            envlp_out = torch.flip(envlp_out, dims=[-1])

            envlp_out = torch.clamp(envlp_out, min=EPS, max=self.envlp_max[ixch])
            gain = (envlp_out / self.envlp_max[ixch]) ** self.expnsn_m1[ixch]
            outputs.append(gain * pass_n_cali)

        y = torch.stack(outputs, dim=-1).sum(dim=-1)
        y = y * self.recruitmnet_out_coef
        return y

    def recruitment_fir(self, x):
        nsamps = x.shape[-1]
        x = x.repeat([1, self.nchans, 1])
        real = F.conv1d(
            x, self.gt_fir, bias=None, padding=self.padding, groups=self.nchans
        )
        imag = F.conv1d(
            x, self.gt_fir_sin, bias=None, padding=self.padding, groups=self.nchans
        )
        real_cali = torch.zeros_like(real)
        imag_cali = torch.zeros_like(imag)
        for i in range(self.nchans):
            real_cali[:, i, : nsamps - self.peaks[i]] = real[
                :, i, self.peaks[i] : nsamps
            ]
            imag_cali[:, i, : nsamps - self.peaks[i]] = imag[
                :, i, self.peaks[i] : nsamps
            ]

        env = torch.sqrt(real_cali * real_cali + imag_cali * imag_cali + EPS)
        env = F.conv1d(
            env, self.fir_lpf, bias=None, padding=self.padding, groups=self.nchans
        )

        env_max = self.envlp_max.unsqueeze(0).unsqueeze(-1).repeat([1, 1, nsamps])
        gain = torch.clamp(env / env_max, min=EPS, max=1)
        gain = gain ** self.expnsn_m1.unsqueeze(0).unsqueeze(-1).repeat([1, 1, nsamps])
        y = torch.sum(gain * real_cali, dim=1, keepdim=True)
        y = y * self.recruitmnet_out_coef
        return y

    def forward(self, x):
        x = self.calibrate_spl(x)
        x = x.unsqueeze(1)
        x = self.src_to_cochlea_filt(x, self.coch_filter_forward)
        x = self.smear(x)
        # x = self.recruitment(x)
        x = self.recruitment_fir(x)
        y = self.src_to_cochlea_filt(x, self.coch_filter_backward)
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
