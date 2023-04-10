from __future__ import annotations

import torch
from torch.nn.functional import grid_sample
from torchaudio.functional import lfilter
from torchaudio.transforms import Resample
from torchinterp1d import interp1d


class EarModel(torch.nn.Module):
    def __init__(
        self,
        nchan=32,
        m_delay=1,
    ):
        super().__init__()
        self.nchan = nchan
        self.m_delay = m_delay

    def forward(
        self,
        reference,
        reference_freq,
        processed,
        processed_freq,
        hearing_loss,
        equalisation,
        level1,
        shift: torch.FloatTensor | None = None,
    ):
        _center_freq = self.center_frequency()

        (
            attn_ohc_y,
            bandwidth_min_y,
            low_knee_y,
            compression_ratio_y,
            attn_ihc_y,
        ) = self.loss_parameters(hearing_loss, _center_freq)

        if equalisation == 0:
            hearing_loss_x = torch.zeros(hearing_loss.size())
        else:
            hearing_loss_x = hearing_loss
        [
            attn_ohc_x,
            bandwidth_min_x,
            low_knee_x,
            compression_ratio_x,
            attn_ihc_x,
        ] = self.loss_parameters(hearing_loss_x, _center_freq)

        hl_max = torch.ones(hearing_loss.size()) * 100

        _center_freq_control = self.center_frequency(shift)
        # Maximum BW for the control
        _, bandwidth_1, _, _, _ = self.loss_parameters(hl_max, _center_freq_control)

        # Input signal adjustments
        # Convert the signals to 24 kHz sampling rate.
        # Using 24 kHz guarantees that all of the cochlear filters have the same shape
        # independent of the incoming signal sampling rates
        reference_24hz, _ = self.resample(reference, reference_freq)
        processed_24hz, freq_sample = self.resample(processed, processed_freq)

        # Check file sizes
        min_signal_length = min(len(reference_24hz), len(processed_24hz))
        reference_24hz = reference_24hz[:min_signal_length]
        processed_24hz = processed_24hz[:min_signal_length]

        # Bulk broadband signal alignment
        # TODO - this is not implemented yet
        reference_24hz, processed_24hz = self.input_align(
            reference_24hz, processed_24hz
        )
        nsamp = len(reference_24hz)

        # For HASQI, here add NAL-R equalization if the quality reference doesn't
        # already have it.
        # TODO - NARL equalization is not implemented yet in torch
        # if itype == 1:
        #     nfir = 140  # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
        #     enhancer = NALR(nfir, freq_sample)
        #     aud = [250, 500, 1000, 2000, 4000, 6000]
        #     nalr_fir, _ = enhancer.build(hearing_loss, aud)
        #     reference_24hz = convolve(reference_24hz, nalr_fir)  # Apply the NAL-R filter
        #     reference_24hz = reference_24hz[nfir: nfir + nsamp]

        reference_mid = self.middle_ear(reference_24hz, freq_sample)
        processed_mid = self.middle_ear(processed_24hz, freq_sample)

        reference_db = torch.zeros((self.nchan, nsamp))
        processed_db = torch.zeros((self.nchan, nsamp))

        # Reference and processed average spectral values
        reference_average = torch.zeros(self.nchan)
        processed_average = torch.zeros(self.nchan)
        reference_control_average = torch.zeros(self.nchan)
        processed_control_average = torch.zeros(self.nchan)

        # Filter bandwidths adjusted for intensity
        reference_bandwidth = torch.zeros(self.nchan)
        processed_bandwidth = torch.zeros(self.nchan)

        reference_b = torch.zeros((self.nchan, nsamp))
        processed_b = torch.zeros((self.nchan, nsamp))

        # Loop over each filter in the auditory filter bank
        for n in range(self.nchan):
            # Control signal envelopes for the reference and processed signals
            # TODO - implement gammatone_basilar_membrane
            (
                reference_control,
                _,
                processed_control,
                _,
            ) = self.gammatone_basilar_membrane(
                reference_mid,
                bandwidth_1[n],
                processed_mid,
                bandwidth_1[n],
                freq_sample,
                _center_freq_control[n],
            )

            # Adjust the auditory filter bandwidths for the average signal level
            # TODO - implement bandwidth_adjust
            reference_bandwidth[n] = self.bandwidth_adjust(
                reference_control, bandwidth_min_x[n], bandwidth_1[n], level1
            )
            processed_bandwidth[n] = self.bandwidth_adjust(
                processed_control, bandwidth_min_y[n], bandwidth_1[n], level1
            )

            # Envelopes and BM motion of the reference and processed signals
            xenv, xbm, yenv, ybm = self.gammatone_basilar_membrane(
                reference_mid,
                reference_bandwidth[n],
                processed_mid,
                processed_bandwidth[n],
                freq_sample,
                _center_freq[n],
            )

            # RMS levels of the ref and output envelopes for linear metric
            reference_average[n] = torch.sqrt(torch.mean(xenv**2))
            processed_average[n] = torch.sqrt(torch.mean(yenv**2))
            reference_control_average[n] = torch.sqrt(
                torch.mean(reference_control**2)
            )
            processed_control_average[n] = torch.sqrt(
                torch.mean(processed_control**2)
            )

            # Cochlear compression for the signal envelopes and BM motion
            # TODO - implement env_compress_basilar_membrane
            (
                reference_cochlear_compression,
                reference_b[n],
            ) = self.env_compress_basilar_membrane(
                xenv,
                xbm,
                reference_control,
                attn_ohc_x[n],
                low_knee_x[n],
                compression_ratio_x[n],
                freq_sample,
                level1,
            )
            (
                processed_cochlear_compression,
                processed_b[n],
            ) = self.env_compress_basilar_membrane(
                yenv,
                ybm,
                processed_control,
                attn_ohc_y[n],
                low_knee_y[n],
                compression_ratio_y[n],
                freq_sample,
                level1,
            )

            # Correct for the delay between the reference and output
            # TODO - implement envelope_align
            processed_cochlear_compression = self.envelope_align(
                reference_cochlear_compression, processed_cochlear_compression
            )  # Align processed envelope to reference
            processed_b[n] = self.envelope_align(
                reference_b[n], processed_b[n]
            )  # Align processed BM motion to reference

            # Convert the compressed envelopes and BM vibration envelopes to dB SPL
            # TODO - implement envelope_sl
            reference_cochlear_compression, reference_b[n] = self.envelope_sl(
                reference_cochlear_compression, reference_b[n], attn_ihc_x[n], level1
            )
            processed_cochlear_compression, processed_b[n] = self.envelope_sl(
                processed_cochlear_compression, processed_b[n], attn_ihc_y[n], level1
            )

            # Apply the IHC rapid and short-term adaptation
            delta = 2  # Amount of overshoot
            # TODO - implement inner_hair_cell_adaptation
            reference_db[n], reference_b[n] = self.inner_hair_cell_adaptation(
                reference_cochlear_compression, reference_b[n], delta, freq_sample
            )
            processed_db[n], processed_b[n] = self.inner_hair_cell_adaptation(
                processed_cochlear_compression, processed_b[n], delta, freq_sample
            )

        # Additive noise level to give the auditory threshold
        ihc_threshold = -10  # Additive noise level, dB re: auditory threshold
        # TODO - implement basilar_membrane_add_noise
        reference_basilar_membrane = self.basilar_membrane_add_noise(
            reference_b, ihc_threshold, level1
        )
        processed_basilar_membrane = self.basilar_membrane_add_noise(
            processed_b, ihc_threshold, level1
        )

        # Correct for the gammatone filterbank interchannel group delay.
        if self.m_delay > 0:
            # TODO - implement group_delay_compensate
            reference_db = self.group_delay_compensate(
                reference_db, reference_bandwidth, _center_freq, freq_sample
            )

            processed_db = self.group_delay_compensate(
                processed_db, reference_bandwidth, _center_freq, freq_sample
            )
            reference_basilar_membrane = self.group_delay_compensate(
                reference_basilar_membrane,
                reference_bandwidth,
                _center_freq,
                freq_sample,
            )
            processed_basilar_membrane = self.group_delay_compensate(
                processed_basilar_membrane,
                reference_bandwidth,
                _center_freq,
                freq_sample,
            )

        # Convert average gammatone outputs to dB SPL
        reference_sl = self.convert_rms_to_sl(
            reference_average,
            reference_control_average,
            attn_ohc_x,
            low_knee_x,
            compression_ratio_x,
            attn_ihc_x,
            level1,
        )
        processed_sl = self.convert_rms_to_sl(
            processed_average,
            processed_control_average,
            attn_ohc_y,
            low_knee_y,
            compression_ratio_y,
            attn_ihc_y,
            level1,
        )

        return (
            reference_db,
            reference_basilar_membrane,
            processed_db,
            processed_basilar_membrane,
            reference_sl,
            processed_sl,
            freq_sample,
        )

    def center_frequency(
        self,
        shift: torch.FloatTensor | None = None,
        low_freq: torch.IntTensor | int = 80,
        high_freq: torch.IntTensor | int = 8000,
        ear_q: torch.FloatTensor | float = 9.26449,
        min_bw: torch.FloatTensor | float = 24.7,
    ):
        """
        Compute the Equivalent Rectangular Bandwidth_[1] frequency spacing for the
        gammatone filterbank. The equation comes from Malcolm Slaney[2].

        Arguments:
            nchan (int): number of filters in the filter bank
            low_freq (int): Low Frequency level.
            high_freq (int): High Frequency level.
            shift (): optional frequency shift of the filter bank specified as a fractional
                shift in distance along the BM. A positive shift is an increase in frequency
                (basal shift), and negative is a decrease in frequency (apical shift). The
                total length of the BM is normalized to 1. The frequency-to-distance map is
                from D.D. Greenwood[3].
            ear_q (float):
            min_bw (float):


        Returns:
            center_freq (torch.FloatTensor): Center frequencies of the gammatone filterbank.

        References:
        .. [1] Moore BCJ, Glasberg BR (1983) Suggested formulae for calculating
               auditory-filter bandwidths and excitation patterns. J Acoustical
               Soc America 74:750-753. Available at
               <https://doi.org/10.1121/1.389861>
        .. [2] Slaney M (1993) An Efficient Implemtnation of the Patterson-
               Holdsworth Auditory Filter Bank. Available at:
               <https://asset-pdf.scinapse.io/prod/396690109/396690109.pdf>.
        .. [3] Greenwood DD (1990) A cochlear frequency-position function for
               several species--29 years later. J Acoust Soc Am 87(6):2592-
               2605. Available at
               <https://doi.o10.1121/1.399052>

        Updates:
        James M. Kates, 25 January 2007.
        Frequency shift added 22 August 2008.
        Lower and upper frequencies fixed at 80 and 8000 Hz, 19 June 2012.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        if not isinstance(low_freq, torch.IntTensor):
            low_freq = torch.tensor(low_freq).int()
        if not isinstance(high_freq, torch.IntTensor):
            high_freq = torch.tensor(high_freq).int()
        if not isinstance(ear_q, torch.FloatTensor):
            ear_q = torch.tensor(ear_q).float()
        if not isinstance(min_bw, torch.FloatTensor):
            min_bw = torch.tensor(min_bw).float()

        if shift is not None:
            k = 1
            A = 165.4  # pylint: disable=invalid-name
            a = 2.1  # shift specified as a fraction of the total length
            # Locations of the low and high frequencies on the BM between 0 and 1
            x_low = (1 / a) * torch.log10(k + (low_freq / A))
            x_high = (1 / a) * torch.log10(k + (high_freq / A))
            # Shift the locations
            x_low = x_low * (1 + shift)
            x_high = x_high * (1 + shift)
            # Compute the new frequency range
            low_freq = A * (10 ** (a * x_low) - k)
            high_freq = A * (10 ** (a * x_high) - k)

        # All of the following expressions are derived in Apple TR #35,
        # "An Efficient Implementation of the Patterson-Holdsworth Cochlear
        # Filter Bank" by Malcolm Slaney.
        # https://engineering.purdue.edu/~malcolm/apple/tr35/PattersonsEar.pdf
        _center_freq = -(ear_q * min_bw) + torch.exp(
            torch.arange(1, self.nchan)
            * (
                -torch.log(high_freq + ear_q * min_bw)
                + torch.log(low_freq + ear_q * min_bw)
            )
            / (self.nchan - 1)
        ) * (high_freq + ear_q * min_bw)

        _center_freq = torch.cat((high_freq.unsqueeze(0), _center_freq), dim=0)
        _center_freq = torch.flip(_center_freq, dims=(0,))
        return _center_freq

    @staticmethod
    def loss_parameters(hearing_loss, center_freq, audiometric_freq=None):
        """
        Compute the loss parameters for the given hearing loss.

        Parameters
        ----------
        hearing_loss : torch.FloatTensor
            Hearing loss in dB.

        Returns
        -------
        loss_parameters : torch.FloatTensor
            Loss parameters for the given hearing loss.

        """
        if audiometric_freq is None:
            audiometric_freq = torch.Tensor([250, 500, 1000, 2000, 4000, 6000])

        nfilt = len(center_freq)
        f_v = torch.cat(
            (
                center_freq[0].unsqueeze(0),
                audiometric_freq,
                center_freq[-1].unsqueeze(0),
            ),
            dim=0,
        )

        loss = interp1d(
            f_v,
            torch.cat(
                (
                    hearing_loss[0].unsqueeze(0),
                    hearing_loss,
                    hearing_loss[-1].unsqueeze(0),
                ),
                dim=0,
            ),
            center_freq,
        )
        loss = torch.maximum(loss, torch.zeros(loss.size())).squeeze(0)

        compression_ratio = 1.25 + 2.25 * torch.arange(nfilt) / (nfilt - 1)

        max_ohc = 70 * (
            1 - (1 / compression_ratio)
        )  # HC loss that results in 1:1 compression
        theoretical_ohc = (
            1.25 * max_ohc
        )  # Loss threshold for adjusting the OHC parameters

        # Apportion the loss in dB to the outer and inner hair cells based on the data of
        # Moore et al (1999), JASA 106, 2761-2778.

        # Reduce the CR towards 1:1 in proportion to the OHC loss.
        attenuated_ohc = 0.8 * torch.clone(loss)
        attnenuated_ihc = 0.2 * torch.clone(loss)

        # create a boolean mask where loss >= theoretical_ohc
        mask = loss >= theoretical_ohc
        attenuated_ohc[mask] = 0.8 * theoretical_ohc[mask]
        attnenuated_ihc[mask] = (
            0.2 * theoretical_ohc[mask] + loss[mask] - theoretical_ohc[mask]
        )

        # Adjust the OHC bandwidth in proportion to the OHC loss
        bandwidth = torch.ones(nfilt)
        bandwidth = (
            bandwidth + (attenuated_ohc / 50.0) + 2.0 * (attenuated_ohc / 50.0) ** 6
        )

        # Compute the compression lower kneepoint and compression ratio
        low_knee = attenuated_ohc + 30
        upamp = 30 + (70 / compression_ratio)  # Output level for an input of 100 dB SPL

        compression_ratio = (100 - low_knee) / (
            upamp + attenuated_ohc - low_knee
        )  # OHC loss Compression ratio

        return attenuated_ohc, bandwidth, low_knee, compression_ratio, attnenuated_ihc

    @staticmethod
    def resample(signal, sample_rate, new_sample_rate=24000):
        """
        Resample the signal to 24 kHz.

        Returns
        -------
        resampled_signal : torch.FloatTensor
            Resampled signal.

        """
        coef_reference_a = torch.Tensor(
            [
                1.0,
                -0.07081207237077872,
                1.2647594875422048,
                0.2132405823253818,
                0.4820212559269799,
                0.13421541556794442,
                0.06248563152819375,
                0.010693174482029118,
            ]
        )
        coef_reference_b = torch.Tensor(
            [
                0.10526806659004136,
                0.2673828276910548,
                0.5089236138475818,
                0.6667272293722993,
                0.6667272293722992,
                0.5089236138475817,
                0.2673828276910549,
                0.1052680665900414,
            ]
        )
        coef_target_a = torch.Tensor(
            [
                1.0,
                5.657986938256279,
                14.00815896651005,
                19.634707135261287,
                16.803741671162324,
                8.771318394792921,
                2.5835900814553923,
                0.3310596846351593,
            ]
        )
        coef_target_b = torch.Tensor(
            [
                0.5753778624919913,
                3.8728648973844546,
                11.32098778566558,
                18.626050890494696,
                18.626050890494696,
                11.320987785665578,
                3.872864897384454,
                0.5753778624919911,
            ]
        )

        sampler = Resample(sample_rate, new_sample_rate)
        # Sampling rate information
        sample_rate_target_khz = round(sample_rate / 1000)  # output rate to nearest kHz
        reference_freq_khz = round(new_sample_rate / 1000)

        # Resample the signal
        if reference_freq_khz == sample_rate_target_khz:
            # No resampling performed if the rates match
            return signal, sample_rate

        if reference_freq_khz < sample_rate_target_khz:
            # Resample for the input rate lower than the output

            resample_signal = sampler(signal)(
                signal, sample_rate_target_khz, reference_freq_khz
            )

            # Match the RMS level of the resampled signal to that of the input
            reference_rms = torch.sqrt(torch.mean(signal**2))
            resample_rms = torch.sqrt(torch.mean(resample_signal**2))
            resample_signal = (reference_rms / resample_rms) * resample_signal

            return resample_signal, new_sample_rate

        resample_signal = sampler(signal)

        # Reduce the input signal bandwidth to 21 kHz (-10.5 to +10.5 kHz)
        # The power equalization is designed to match the signal intensities
        # over the frequency range spanned by the gammatone filter bank.
        # Chebyshev Type 2 LP
        reference_filter = lfilter(signal, coef_reference_a, coef_reference_b)

        target_filter = lfilter(signal, coef_target_a, coef_target_b)

        # Compute the input and output RMS levels within the 21 kHz bandwidth and
        # match the output to the input
        reference_rms = torch.sqrt(torch.mean(reference_filter**2))
        resample_rms = torch.sqrt(torch.mean(target_filter**2))
        resample_signal = (reference_rms / resample_rms) * resample_signal

        return resample_signal, new_sample_rate

    @staticmethod
    def convert_rms_to_sl(
        reference,
        control,
        attnenuated_ohc,
        threshold_low,
        compression_ratio,
        attnenuated_ihc,
        level1,
        threshold_high=100,
        small=1e-30,
    ):
        """
        Covert the Root Mean Square average output of the gammatone filter bank
        into dB SL. The gain is linear below the lower threshold, compressive
        with a compression ratio of CR:1 between the lower and upper thresholds,
        and reverts to linear above the upper threshold. The compressor
        assumes that auditory thresold is 0 dB SPL.

        Arguments:
            reference (): analytic signal envelope (magnitude) returned by the
            gammatone filter bank, RMS average level
            control (): control signal envelope
            attenuated_ohc (): OHC attenuation at the input to the compressor
            threshold_low (): kneepoint for the low-level linear amplification
            compression_ratio (): compression ratio
            attenuated_ihc (): IHC attenuation at the input to the synapse
            level1 (): dB reference level: a signal having an RMS value of 1 is
                    assigned to Level1 dB SPL.
            threshold_high (int):
            small (float):

        Returns:
            reference_db (): compressed output in dB above the impaired threshold

        Updates:
            James M. Kates, 6 August 2007.
            Version for two-tone suppression, 29 August 2008.
            Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # Initialize the compression parameters
        threshold_high = 100  # Upper compression threshold

        # Convert the control to dB SPL
        small = 1e-30
        control_db_spl = torch.maximum(control, torch.ones(control.size()) * small)
        control_db_spl = level1 + 20 * torch.log10(control_db_spl)
        control_db_spl = torch.minimum(
            control_db_spl, torch.ones(control_db_spl.size()) * threshold_high
        )
        control_db_spl = torch.maximum(
            control_db_spl, torch.ones(control_db_spl.size()) * threshold_low
        )

        # Compute compression gain in dB
        gain = -attnenuated_ohc - (control_db_spl - threshold_low) * (
            1 - (1 / compression_ratio)
        )

        # Convert the signal envelope to dB SPL
        control_db_spl = torch.maximum(reference, torch.ones(control.size()) * small)
        control_db_spl = level1 + 20 * torch.log10(control_db_spl)
        control_db_spl = torch.maximum(
            control_db_spl, torch.zeros(control_db_spl.size())
        )
        reference_db = control_db_spl + gain - attnenuated_ihc
        reference_db = torch.maximum(reference_db, torch.zeros(reference_db.size()))

        return reference_db
