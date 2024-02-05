"""A torch implementation of the HAAQI, HASQI and HASPI ear model."""
from __future__ import annotations

# pylint: disable=import-error
import torch
from torchaudio.functional import lfilter
from torchaudio.transforms import Resample

from clarity.predictor.torch_haindex import (
    COMPRESS_BASILAR_MEMBRANE_COEFS,
    CORRECT_DELAY_COEFS,
    MIDDLE_EAR_COEF,
    full_correlation,
    interp1d,
)


class EarModel(torch.nn.Module):
    """
    Class that implements a cochlear model that includes the middle ear,
    auditory filter bank, Outer Hair Cell (OHC) dynamic-range compression,
    and Inner Hair Cell (IHC) attenuation.

    """

    def __init__(
        self,
        nchan: int = 32,
        m_delay: int = 1,
        shift: float = 0,
        freq_sample: int = 24000,
        small: float = 1e-30,
        min_bandwidth: float = 24.7,
        device="cuda",
    ):
        """
        Constructor takes the reference and processed signals that are to be
        compared. The reference is at the reference intensity (e.g. 65 dB SPL
        or with NAL-R amplification) and has no other processing. The processed
        signal is the hearing-aid output, and is assumed to have the same or
        greater group delay compared to the reference.

        Arguments:
            itype (int): purpose for the calculation:
                0=intelligibility: reference is normal hearing and must not
                    include NAL-R EQ
                1=quality: reference does not include NAL-R EQ
                2=quality: reference already has NAL-R EQ applied

            nchan (int): auditory frequency bands.
            m_delay (int): Compensate for the gammatone group delay.
            shift (float): optional frequency shift of the filter bank specified as
                a fractional shift in distance along the BM. A positive shift is an
                increase in frequency (basal shift), and negative is a decrease in
                frequency (apical shift). The total length of the BM is normalized to 1.
                The frequency-to-distance map is from D.D. Greenwood[3].
            target_sample_rate (int): target sample rate for the resampling
                of the signal.
            small (float): small value to avoid division by zero.
        """
        super().__init__()
        self.device = device

        self.audiometric_freq = torch.tensor([250, 500, 1000, 2000, 4000, 6000]).to(
            self.device
        )

        self.nchan = nchan
        self.m_delay = m_delay
        self.freq_sample = freq_sample
        self.small = small
        self.min_bandwidth = min_bandwidth

        # General Precomputed parameters
        # -------------------------------
        self._center_freq = self.center_frequency().to(self.device)
        self._center_freq_control = self.center_frequency(
            shift=None if shift == 0 else shift
        ).to(self.device)

        # Loss parameters precomputed
        # ----------------------------
        self.nfilt = len(self._center_freq)
        self.compression_ratio = 1.25 + 2.25 * torch.arange(
            self.nfilt, device=self.device
        ) / (self.nfilt - 1)
        # HC loss that results in 1:1 compression
        self.max_ohc = 70 * (1 - (1 / self.compression_ratio))
        # Loss threshold for adjusting the OHC parameters
        self.theoretical_ohc = 1.25 * self.max_ohc
        # Output level for an input of 100 dB SPL
        self.upamp = 30 + (70 / self.compression_ratio)

        self.batch_size = 1

    def forward(
        self,
        reference,
        processed,
        hearing_loss,
        equalisation,
        level1,
    ):
        # Add batch dimension when running a single signal
        if 1 in [reference.ndim, processed.ndim, hearing_loss.ndim]:
            raise ValueError(
                f"Reference signal ({reference.size()}), "
                f"processed signal ({processed.size()}) and, "
                f"hearing loss ({hearing_loss.size()}) must "
                "all have the same batch size. Min batch size = 1"
            )

        if equalisation not in [0, 1, 2]:
            raise ValueError("equalisation must be 0, 1 or 2")

        self.batch_size = reference.size(dim=0)

        self._center_freq = self._center_freq.unsqueeze(0).repeat(self.batch_size, 1)
        self._center_freq_control = self._center_freq_control.unsqueeze(0).repeat(
            self.batch_size, 1
        )
        self.audiometric_freq = self.audiometric_freq.unsqueeze(0).repeat(
            self.batch_size, 1
        )
        self.theoretical_ohc = self.theoretical_ohc.unsqueeze(0).repeat(
            self.batch_size, 1
        )

        (
            attn_ohc_y,
            bandwidth_min_y,
            low_knee_y,
            compression_ratio_y,
            attn_ihc_y,
        ) = self.loss_parameters(hearing_loss, self._center_freq)

        if equalisation == 0:
            # for equalisation = 0
            [
                attn_ohc_x,
                bandwidth_min_x,
                low_knee_x,
                compression_ratio_x,
                attn_ihc_x,
            ] = self.loss_parameters(
                torch.zeros(self.audiometric_freq.size(), device=self.device),
                self._center_freq,
            )

            hl_max = (torch.ones(self.audiometric_freq.size()) * 100).to(self.device)

            # Maximum BW for the control
            _, self.bandwidth_1, _, _, _ = self.loss_parameters(
                hl_max, self._center_freq_control
            )
        else:
            attn_ohc_x = attn_ohc_y.clone()
            bandwidth_min_x = bandwidth_min_y.clone()
            low_knee_x = low_knee_y.clone()
            compression_ratio_x = compression_ratio_y.clone()
            attn_ihc_x = attn_ihc_y.clone()

        # Check file sizes
        min_signal_length = min(reference.size()[1], processed.size()[1])
        reference_24hz = reference[:, :min_signal_length]
        processed_24hz = processed[:, :min_signal_length]

        # Bulk broadband signal alignment
        # reference_24hz, processed_24hz = self.input_align(
        #     reference_24hz, processed_24hz
        # )
        nsamp = reference_24hz.size()[1]

        # For HASQI, here add NAL-R equalization if the quality reference doesn't
        # already have it.
        # TODO - NALR equalization in torch
        # if itype == 1:
        #     nfir = 140  # Length in samples of the FIR NAL-R EQ filter (24-kHz rate)
        #     enhancer = NALR(nfir, freq_sample)
        #     aud = [250, 500, 1000, 2000, 4000, 6000]
        #     nalr_fir, _ = enhancer.build(hearing_loss, aud)
        # Apply the NAL-R filter
        #     reference_24hz = convolve(reference_24hz, nalr_fir)
        #     reference_24hz = reference_24hz[nfir: nfir + nsamp]

        reference_mid = self.middle_ear(reference_24hz)
        processed_mid = self.middle_ear(processed_24hz)

        reference_db = torch.zeros(
            (self.batch_size, self.nchan, nsamp), device=self.device
        )
        processed_db = torch.zeros(
            (self.batch_size, self.nchan, nsamp), device=self.device
        )

        # Reference and processed average spectral values
        reference_average = torch.zeros(
            (self.batch_size, self.nchan), device=self.device
        )
        processed_average = torch.zeros(
            (self.batch_size, self.nchan), device=self.device
        )
        reference_control_average = torch.zeros(
            (self.batch_size, self.nchan), device=self.device
        )
        processed_control_average = torch.zeros(
            (self.batch_size, self.nchan), device=self.device
        )

        # Filter bandwidths adjusted for intensity
        reference_bandwidth = torch.zeros(
            (self.batch_size, self.nchan), device=self.device
        )
        processed_bandwidth = torch.zeros(
            (self.batch_size, self.nchan), device=self.device
        )

        reference_b = torch.zeros(
            (self.batch_size, self.nchan, nsamp), device=self.device
        )
        processed_b = torch.zeros(
            (self.batch_size, self.nchan, nsamp), device=self.device
        )

        (
            reference_control,
            _,
            processed_control,
            _,
        ) = self.gammatone_basilar_membrane(
            reference_mid,
            self.bandwidth_1,
            processed_mid,
            self.bandwidth_1,
            self.freq_sample,
            self._center_freq_control,
        )

        # Loop over each filter in the auditory filter bank
        for n in range(self.nchan):
            # Control signal envelopes for the reference and processed signals
            (
                reference_control,
                _,
                processed_control,
                _,
            ) = self.gammatone_basilar_membrane(
                reference_mid,
                self.bandwidth_1[:, n, None],
                processed_mid,
                self.bandwidth_1[:, n, None],
                self.freq_sample,
                self._center_freq_control[:, n, None],
            )

            # Adjust the auditory filter bandwidths for the average signal level
            reference_bandwidth[:, n, None] = self.bandwidth_adjust(
                reference_control,
                bandwidth_min_x[:, n, None],
                self.bandwidth_1[:, n, None],
                level1,
            )
            processed_bandwidth[:, n, None] = self.bandwidth_adjust(
                processed_control,
                bandwidth_min_y[:, n, None],
                self.bandwidth_1[:, n, None],
                level1,
            )

            # Envelopes and BM motion of the reference and processed signals
            xenv, xbm, yenv, ybm = self.gammatone_basilar_membrane(
                reference_mid,
                reference_bandwidth[:, n, None],
                processed_mid,
                processed_bandwidth[:, n, None],
                self.freq_sample,
                self._center_freq[:, n, None],
            )

            # RMS levels of the ref and output envelopes for linear metric
            reference_average[:, n] = torch.sqrt(torch.mean(xenv**2))
            processed_average[:, n] = torch.sqrt(torch.mean(yenv**2))
            reference_control_average[:, n] = torch.sqrt(
                torch.mean(reference_control**2)
            )
            processed_control_average[:, n] = torch.sqrt(
                torch.mean(processed_control**2)
            )

            # Cochlear compression for the signal envelopes and BM motion
            (
                reference_cochlear_compression,
                reference_b[:, n, :],
            ) = self.env_compress_basilar_membrane(
                xenv,
                xbm,
                reference_control,
                attn_ohc_x[..., n],
                low_knee_x[..., n],
                compression_ratio_x[..., n],
                self.freq_sample,
                level1,
            )
            (
                processed_cochlear_compression,
                processed_b[:, n, :],
            ) = self.env_compress_basilar_membrane(
                yenv,
                ybm,
                processed_control,
                attn_ohc_y[:, n],
                low_knee_y[:, n],
                compression_ratio_y[:, n],
                self.freq_sample,
                level1,
            )

            # Correct for the delay between the reference and output
            processed_cochlear_compression = self.envelope_align(
                reference_cochlear_compression, processed_cochlear_compression
            )  # Align processed envelope to reference
            processed_b[:, n] = self.envelope_align(
                reference_b[:, n], processed_b[:, n]
            )  # Align processed BM motion to reference

            # Convert the compressed envelopes and BM vibration envelopes to dB SPL
            reference_cochlear_compression, reference_b[:, n] = self.envelope_sl(
                reference_cochlear_compression,
                reference_b[:, n],
                attn_ihc_x[:, n],
                level1,
            )
            processed_cochlear_compression, processed_b[:, n] = self.envelope_sl(
                processed_cochlear_compression,
                processed_b[:, n],
                attn_ihc_y[:, n],
                level1,
            )

            # Apply the IHC rapid and short-term adaptation
            delta = 2  # Amount of overshoot
            reference_db[:, n], reference_b[:, n] = self.inner_hair_cell_adaptation(
                reference_cochlear_compression, reference_b[:, n], delta
            )
            processed_db[:, n], processed_b[:, n] = self.inner_hair_cell_adaptation(
                processed_cochlear_compression, processed_b[:, n], delta
            )

        # Additive noise level to give the auditory threshold
        # Additive noise level, dB re: auditory threshold
        ihc_threshold = (
            torch.tensor(-10, device=self.device)
            .unsqueeze(0)
            .repeat(self.batch_size, 1)
        )
        reference_basilar_membrane = self.basilar_membrane_add_noise(
            reference_b, ihc_threshold, level1
        )
        processed_basilar_membrane = self.basilar_membrane_add_noise(
            processed_b, ihc_threshold, level1
        )

        # Correct for the gammatone filterbank interchannel group delay.
        if self.m_delay > 0:
            reference_db = self.group_delay_compensate(
                reference_db,
                reference_bandwidth,  # self._center_freq, freq_sample
            )

            processed_db = self.group_delay_compensate(
                processed_db,
                reference_bandwidth,  # self._center_freq, freq_sample
            )
            reference_basilar_membrane = self.group_delay_compensate(
                reference_basilar_membrane,
                reference_bandwidth,
            )
            processed_basilar_membrane = self.group_delay_compensate(
                processed_basilar_membrane,
                reference_bandwidth,
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
        )

    def center_frequency(
        self,
        low_freq: int = 80,
        high_freq: int = 8000,
        shift: float | None = None,
        ear_q: float = 9.26449,
        min_bw: float = 24.7,
    ):
        """
        Compute the Equivalent Rectangular Bandwidth_[1] frequency spacing for the
        gammatone filterbank. The equation comes from Malcolm Slaney[2].

        Arguments:
            low_freq (int): Low Frequency level.
            high_freq (int): High Frequency level.
            shift (float): optional frequency shift of the filter bank specified as
                a fractional shift in distance along the BM. A positive shift is an
                increase in frequency (basal shift), and negative is a decrease in
                frequency (apical shift). The total length of the BM is normalized to 1.
                The frequency-to-distance map is from D.D. Greenwood[3].
            ear_q (float):
            min_bw (float):


        Returns:
            center_freq (torch.FloatTensor): Center frequencies of the gammatone
                filterbank.

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
        return torch.flip(_center_freq, dims=(0,))

    def loss_parameters(self, hearing_loss: torch.tensor, center_freq: torch.tensor):
        """
        Compute the loss parameters for the given hearing loss.

        Arguments:
            hearing_loss : torch.tensor
                Hearing loss in dB.
            center_freq : torch.tensor

        Returns:
            attenuated_ohc : torch.tensor
                Attenuated outer hair cell loss in dB.
            bandwidth : float
            low_knee : float
            compression_ratio : float
            attnenuated_ihc: torch.tensor
        """
        f_v = torch.cat(
            (
                center_freq[:, 0, None],
                self.audiometric_freq,
                center_freq[:, -1, None],
            ),
            dim=1,
        )
        loss = interp1d(
            f_v,
            torch.cat(
                (
                    hearing_loss[:, 0, None],
                    hearing_loss,
                    hearing_loss[:, -1, None],
                ),
                dim=1,
            ),
            center_freq,
        )
        loss = torch.maximum(loss, torch.zeros(loss.size(), device=self.device))

        # Apportion the loss in dB to the outer and inner hair cells based on the data
        # of
        # Moore et al (1999), JASA 106, 2761-2778.

        # Reduce the CR towards 1:1 in proportion to the OHC loss.
        attenuated_ohc = 0.8 * torch.clone(loss)
        attnenuated_ihc = 0.2 * torch.clone(loss)

        # create a boolean mask where loss >= theoretical_ohc
        mask = loss >= self.theoretical_ohc
        attenuated_ohc[mask] = 0.8 * self.theoretical_ohc[mask]
        attnenuated_ihc[mask] = (
            0.2 * self.theoretical_ohc[mask] + loss[mask] - self.theoretical_ohc[mask]
        )

        # Adjust the OHC bandwidth in proportion to the OHC loss
        bandwidth = torch.ones((self.batch_size, self.nfilt), device=self.device)
        bandwidth = (
            bandwidth + (attenuated_ohc / 50.0) + 2.0 * (attenuated_ohc / 50.0) ** 6
        )

        # Compute the compression lower kneepoint and compression ratio
        low_knee = attenuated_ohc + 30

        compression_ratio = (100 - low_knee) / (
            self.upamp + attenuated_ohc - low_knee
        )  # OHC loss Compression ratio

        return attenuated_ohc, bandwidth, low_knee, compression_ratio, attnenuated_ihc

    def resample(self, signal, sample_rate):
        """
        Resample the signal to 24 kHz.

        Returns
        -------
        resampled_signal : torch.FloatTensor
            Resampled signal.

        """

        sampler = Resample(sample_rate, self.freq_sample)

        # Resample the signal
        if sample_rate == self.freq_sample:
            # No resampling performed if the rates match
            return signal, sample_rate

        return sampler(signal), self.freq_sample

    def input_align(self, reference, processed):
        # Match the length of the processed output to the reference for the purposes
        # of computing the cross-covariance
        reference_n = len(reference)
        processed_n = len(processed)
        min_sample_length = min(reference_n, processed_n)

        # Determine the delay of the output relative to the reference
        reference_processed_correlation = full_correlation(
            reference[:min_sample_length] - torch.mean(reference[:min_sample_length]),
            processed[:min_sample_length] - torch.mean(processed[:min_sample_length]),
        )
        index = torch.argmax(torch.abs(reference_processed_correlation))
        delay = min_sample_length - index - 1

        # Back up 2 msec to allow for dispersion
        delay = torch.round(delay - 2 * self.freq_sample / 1000).int()  # Back up 2 ms

        # Align the output with the reference allowing for the dispersion
        if delay > 0:
            # Output delayed relative to the reference
            processed = torch.cat((processed[delay:processed_n], torch.zeros(delay)))
        else:
            # Output advanced relative to the reference
            processed = torch.cat(
                (torch.zeros(-delay), processed[: processed_n + delay])
            )

        # Find the start and end of the noiseless reference sequence
        reference_abs = torch.abs(reference)
        reference_max = torch.max(reference_abs)
        reference_threshold = 0.001 * reference_max  # Zero detection threshold

        above_threshold = torch.where(reference_abs > reference_threshold)[0]
        reference_n_above_threshold = above_threshold[0]
        reference_n_below_threshold = above_threshold[-1]

        # Prune the sequences to remove the leading and trailing zeros
        reference_n_below_threshold = min(reference_n_below_threshold, processed_n)

        return (
            reference[reference_n_above_threshold : reference_n_below_threshold + 1],
            processed[reference_n_above_threshold : reference_n_below_threshold + 1],
        )

    def middle_ear(self, reference):
        """
        Design the middle ear filters and process the input through the
        cascade of filters. The middle ear model is a 2-pole HP filter
        at 350 Hz in series with a 1-pole LP filter at 5000 Hz. The
        result is a rough approximation to the equal-loudness contour
        at threshold.

        Arguments:
        reference (np.ndarray):	input signal
        freq_sample (int): sampling rate in Hz

        Returns:
        xout (): filtered output

        Updates:
        James M. Kates, 18 January 2007.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # Design the 1-pole Butterworth LP using the bi-linear transformation
        butterworth_low_pass = torch.tensor(
            MIDDLE_EAR_COEF[str(self.freq_sample)]["butterworth_low_pass"]
        ).to(self.device)

        low_pass = torch.tensor(
            MIDDLE_EAR_COEF[str(self.freq_sample)]["low_pass"],
            dtype=torch.float64,
        ).to(self.device)

        # LP filter the input
        y = lfilter(reference, low_pass, butterworth_low_pass, clamp=False)

        # Design the 2-pole Butterworth HP using the bilinear transformation
        butterworth_high_pass = torch.tensor(
            MIDDLE_EAR_COEF[str(self.freq_sample)]["butterworth_high_pass"]
        ).to(self.device)

        high_pass = torch.tensor(
            MIDDLE_EAR_COEF[str(self.freq_sample)]["high_pass"],
            dtype=torch.float64,
        ).to(self.device)

        # HP fitler the signal
        return lfilter(y, high_pass, butterworth_high_pass, clamp=False)

    def gammatone_basilar_membrane(
        self,
        reference,
        reference_bandwidth,
        processed,
        processed_bandwidth,
        center_freq,
        ear_q=9.26449,
    ):
        """
        4th-order gammatone auditory filter.
        This implementation is based on the c program published on-line by Ning Ma,
        U. Sheffield, UK[1]_ that gives an implementation of the
        Martin Cooke filters[2]_: an impulse-invariant transformation of the gammatone
        filter. The signal is demodulated down to baseband using a complex exponential,
        and then passed through a cascade of four one-pole low-pass filters.

        This version filters two signals that have the same sampling rate and the same
        gammatone filter center frequencies. The lengths of the two signals should match
        if they don't, the signals are truncated to the shorter of the two lengths.

        Arguments:
            reference (): first sequence to be filtered
            reference_bandwidth: bandwidth for x relative to that of a normal ear
            processed (): second sequence to be filtered
            processed_bandwidth (): bandwidth for x relative to that of a normal ear
            freq_sample (): sampling rate in Hz
            center_frequency (int): filter center frequency in Hz
            ear_q: (float): ???
            min_bandwidth (float): ???

        Returns:
            reference_envelope (): filter envelope output (modulated down to baseband)
                1st signal
            reference_basilar_membrane (): Basilar Membrane for the first signal
            processed_envelope (): filter envelope output (modulated down to baseband)
                2nd signal
            processed_basilar_membrane (): Basilar Membrane for the second signal

        References:
        .. [1] Ma N, Green P, Barker J, Coy A (2007) Exploiting correlogram
               structure for robust speech recognition with multiple speech
               sources. Speech Communication, 49 (12): 874-891. Availab at
               <https://doi.org/10.1016/j.specom.2007.05.003>
               <https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/gammatone/>
        .. [2] Cooke, M. (1993) Modelling auditory processing and organisation.
               Cambridge University Press

        Updates:
        James M. Kates, 8 Jan 2007.
        Vectorized version for efficient MATLAB execution, 4 February 2007.
        Cosine and sine generation, 29 June 2011.
        Output sine and cosine sequences, 19 June 2012.
        Cosine/sine loop speed increased, 9 August 2013.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        # Filter Equivalent Rectangular Bandwidth from Moore and Glasberg (1983)
        # doi: 10.1121/1.389861
        erb = self.min_bandwidth + (center_freq / ear_q)

        # Check the lengths of the two signals and trim to shortest
        min_sample = min(reference.size(-1), processed.size(-1))
        x = reference[:, :min_sample]
        y = processed[:, :min_sample]

        # Filter the first signal
        # Initialize the filter coefficients
        tpt = 2 * torch.pi / self.freq_sample
        tpt_bw = reference_bandwidth.double() * tpt * erb * 1.019
        a = torch.exp(-tpt_bw)
        a_1 = 4.0 * a
        a_2 = -6.0 * a * a
        a_3 = 4.0 * a * a * a
        a_4 = -a * a * a * a
        a_5 = 4.0 * a * a
        gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

        # Initialize the complex demodulation
        npts = x.size(dim=1)
        sincf, coscf = self.gammatone_bandwidth_demodulation(npts, tpt, center_freq)

        # Filter the real and imaginary parts of the signal
        one_tensor = torch.ones((self.batch_size, self.nchan), device=self.device)
        zero_tensor = torch.zeros((self.batch_size, self.nchan), device=self.device)
        ureal = lfilter(
            x * coscf,
            torch.cat([one_tensor, -a_1, -a_2, -a_3, -a_4], dim=1),
            torch.cat([one_tensor, a_1, a_5, zero_tensor, zero_tensor], dim=1),
            clamp=False,
        )
        uimag = lfilter(
            x * sincf,
            torch.cat([one_tensor, -a_1, -a_2, -a_3, -a_4], dim=1),
            torch.cat([one_tensor, a_1, a_5, zero_tensor, zero_tensor], dim=1),
            clamp=False,
        )

        ureal = ureal.unsqueeze(1).repeat(1, self.nchan, 1)
        uimag = uimag.unsqueeze(1).repeat(1, self.nchan, 1)
        coscf = coscf.unsqueeze(1).repeat(1, self.nchan, 1)
        sincf = sincf.unsqueeze(1).repeat(1, self.nchan, 1)
        gain = gain.unsqueeze(1).repeat(1, self.nchan, 1)
        # Extract the BM velocity and the envelope
        reference_basilar_membrane = gain * (ureal * coscf + uimag * sincf)
        reference_envelope = gain * torch.sqrt(ureal * ureal + uimag * uimag)

        # Filter the second signal using the existing cosine and sine sequences
        tpt_bw = processed_bandwidth.double() * tpt * erb * 1.019
        a = torch.exp(-tpt_bw)
        a_1 = 4.0 * a
        a_2 = -6.0 * a * a
        a_3 = 4.0 * a * a * a
        a_4 = -a * a * a * a
        a_5 = 4.0 * a * a
        gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)

        # Filter the real and imaginary parts of the signal
        ureal = lfilter(
            y * coscf,
            torch.cat([one_tensor, -a_1, -a_2, -a_3, -a_4], dim=1),
            torch.cat([one_tensor, a_1, a_5, zero_tensor, zero_tensor], dim=1),
            clamp=False,
        )
        uimag = lfilter(
            y * sincf,
            torch.cat([one_tensor, -a_1, -a_2, -a_3, -a_4], dim=1),
            torch.cat([one_tensor, a_1, a_5, zero_tensor, zero_tensor], dim=1),
            clamp=False,
        )

        # Extract the BM velocity and the envelope
        processed_basilar_membrane = gain * (ureal * coscf + uimag * sincf)
        processed_envelope = gain * torch.sqrt(ureal * ureal + uimag * uimag)

        return (
            reference_envelope,
            reference_basilar_membrane,
            processed_envelope,
            processed_basilar_membrane,
        )

    def gammatone_bandwidth_demodulation(
        self,
        npts,
        tpt,
        center_freq,
    ):
        """Gamma tone bandwidth demodulation

        Arguments:
            npts (): ???
            tpt (): ???
            center_freq (): ???
            center_freq_cos (): ???
            sincf (): ???

        Returns:
            sincf (): ???
            coscf (): ???
        """
        center_freq_cos = torch.zeros((self.batch_size, npts), device=self.device)
        center_freq_sin = torch.zeros((self.batch_size, npts), device=self.device)

        if not isinstance(tpt, torch.FloatTensor):
            tpt = torch.tensor(tpt).to(self.device)

        cos_n = torch.cos(tpt * center_freq)
        sin_n = torch.sin(tpt * center_freq)
        cold = 1.0
        sold = 0.0
        center_freq_cos[:, 0] = cold
        center_freq_sin[:, 0] = sold
        for n in range(1, npts):
            arg = cold * cos_n + sold * sin_n
            sold = sold * cos_n - cold * sin_n
            cold = arg
            center_freq_cos[:, n] = cold
            center_freq_sin[:, n] = sold

        return center_freq_sin, center_freq_cos

    @staticmethod
    def bandwidth_adjust(
        control: torch.tensor,
        bandwidth_min: float,
        bandwidth_max: float,
        level1: float,
    ) -> torch.tensor:
        """
        Compute the increase in auditory filter bandwidth in response to high signal
            levels.

        Arguments:
            control (): envelope output in the control filter band
            bandwidth_min (): auditory filter bandwidth computed for the loss (or NH)
            bandwidth_max (): auditory filter bandwidth at maximum OHC damage
            level1 ():     RMS=1 corresponds to Level1 dB SPL

        Returns:
            bandwidth (): filter bandwidth increased for high signal levels

        Updates:
        James M. Kates, 21 June 2011.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # Compute the control signal level
        control_rms = torch.sqrt(torch.mean(control**2))
        control_db = 20 * torch.log10(control_rms) + level1

        # Adjust the auditory filter bandwidth
        if control_db < 50:
            # No BW adjustment for a signal below 50 dB SPL
            return bandwidth_min
        if control_db > 100:
            # Maximum BW if signal is above 100 dB SPL
            return bandwidth_max

        return bandwidth_min + ((control_db - 50) / 50) * (
            bandwidth_max - bandwidth_min
        )

    def env_compress_basilar_membrane(
        self,
        envsig: torch.tensor,
        bm: torch.tensor,  # pylint: disable=invalid-name
        control: torch.tensor,
        attn_ohc: torch.tensor,
        threshold_low: torch.tensor,
        compression_ratio: torch.tensor,
        level1: float,
        threshold_high: int = 100,
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Compute the cochlear compression in one auditory filter band. The gain is linear
        below the lower threshold, compressive with a compression ratio of CR:1 between
        the lower and upper thresholds, and reverts to linear above the upper threshold.
        The compressor assumes that auditory threshold is 0 dB SPL.

        Arguments:
            envsig (): analytic signal envelope (magnitude) returned by the
                    gammatone filter bank
            bm (): BM motion output by the filter bank
            control (): analytic control envelope returned by the wide control
                    path filter bank
            attn_ohc (): OHC attenuation at the input to the compressor
            threshold_Low (): kneepoint for the low-level linear amplification
            compression_ratio (): compression ratio
            fsamp (): sampling rate in Hz
            level1 (): dB reference level: a signal having an RMS value of 1 is
                    assigned to Level1 dB SPL.
            threshold_high: kneepoint for the high-level linear amplification

        Returns:
            compressed_signal (): compressed version of the signal envelope
            compressed_basilar_membrane (): compressed version of the BM motion

        Updates:
        James M. Kates, 19 January 2007.
        LP filter added 15 Feb 2007 (Ref: Zhang et al., 2001)
        Version to compress the envelope, 20 Feb 2007.
        Change in the OHC I/O function, 9 March 2007.
        Two-tone suppression added 22 August 2008.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        # Convert the control envelope to dB SPL
        logenv = torch.maximum(control, torch.tensor(self.small))

        logenv = level1 + 20 * torch.log10(logenv)
        # Clip signal levels above the upper threshold
        logenv = torch.minimum(logenv, torch.tensor(threshold_high))
        # Clip signal at the lower threshold
        threshold_low = threshold_low.unsqueeze(1).repeat(1, logenv.shape[1])
        logenv = torch.max(logenv, threshold_low)

        attn_ohc = attn_ohc.unsqueeze(1).repeat(1, logenv.shape[1])
        compression_ratio = compression_ratio.unsqueeze(1).repeat(1, logenv.shape[1])
        # Compute the compression gain in dB
        gain = -attn_ohc - (logenv - threshold_low) * (1 - (1 / compression_ratio))

        # Convert the gain to linear and apply a LP filter to give a 0.2 ms delay
        gain = torch.pow(10, gain / 20)
        coefs = COMPRESS_BASILAR_MEMBRANE_COEFS[str(self.freq_sample)]

        gain = lfilter(
            gain.double(),
            torch.tensor(coefs["a"], dtype=torch.double).to(self.device),
            torch.tensor(coefs["b"], dtype=torch.double).to(self.device),
            clamp=False,
        )

        # Apply the gain to the signals
        compressed_signal = gain * envsig
        compressed_basilar_membrane = gain * bm

        return compressed_signal, compressed_basilar_membrane

    def envelope_align(
        self,
        reference: torch.tensor,
        output: torch.tensor,
        corr_range: int = 100,
    ) -> torch.tensor:
        """
        Align the envelope of the processed signal to that of the reference signal.

        Arguments:
            reference (): envelope or BM motion of the reference signal
            output (): envelope or BM motion of the output signal
            freq_sample (int): Frequency sample rate in Hz
            corr_range (int): range in msec for the correlation

        Returns:
            y (): shifted output envelope to match the input

        Updates:
        James M. Kates, 28 October 2011.
        Absolute value of the cross-correlation peak removed, 22 June 2012.
        Cross-correlation range reduced, 13 August 2013.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # The MATLAB code limits the range of lags to search (to 100 ms) to save
        # computation
        # time - no such option exists in numpy, but the code below limits the delay
        # to the
        # same range as in Matlab, for consistent results
        lags = torch.round(
            0.001
            * torch.tensor(corr_range)
            .unsqueeze(0)
            .repeat(self.batch_size, 1)
            .to(self.device)
            * self.freq_sample
        ).int()

        # Range in samples
        npts = (
            torch.tensor(reference.size(dim=1))
            .unsqueeze(0)
            .repeat(self.batch_size, 1)
            .to(self.device)
        )
        lags = torch.min(lags, npts)

        ref_out_correlation = full_correlation(reference, output)

        # Limit the range in which

        location = torch.zeros(lags.shape, device=self.device)
        for n in range(location.shape[0]):
            location[n] = torch.argmax(
                ref_out_correlation[n, npts[n] - lags[n] : npts[n] + lags[n]],
                keepdim=True,
            )

        delay = lags - location - 1

        out = []
        # Time shift the output sequence
        for n in range(self.batch_size):
            if delay[n] > 0:
                # Output delayed relative to the reference
                out.append(
                    torch.cat(
                        (
                            output[n, int(delay[n]) : int(npts[n])],
                            torch.zeros(int(delay[n]), device=self.device),
                        )
                    )
                )
            else:
                # Output advanced relative to the reference
                out.append(
                    torch.cat(
                        (
                            torch.zeros(int(-delay[n]), device=self.device),
                            output[n, : int(npts[n] + delay[n])],
                        )
                    )
                )
        return torch.stack(out)
        # if delay > 0:
        #     # Output delayed relative to the reference
        #     return torch.cat(
        #         (
        #             output[:, delay:npts],
        #             torch.zeros((self.batch_size, delay), device=self.device),
        #         ),
        #         dim=1,
        #     )
        # return torch.cat(
        #     (
        #         torch.zeros((self.batch_size, -delay), device=self.device),
        #         output[:, : torch.add(npts, delay)],
        #     ),
        #     dim=1,
        # )

    def convert_rms_to_sl(
        self,
        reference,
        control,
        attnenuated_ohc,
        threshold_low,
        compression_ratio,
        attnenuated_ihc,
        level1,
        threshold_high=100,
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


        Returns:
            reference_db (): compressed output in dB above the impaired threshold

        Updates:
            James M. Kates, 6 August 2007.
            Version for two-tone suppression, 29 August 2008.
            Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """

        # Convert the control to dB SPL
        control_db_spl = torch.maximum(
            control, torch.ones(control.size(), device=self.device) * self.small
        )
        control_db_spl = level1 + 20 * torch.log10(control_db_spl)
        control_db_spl = torch.minimum(
            control_db_spl,
            torch.ones(control_db_spl.size(), device=self.device) * threshold_high,
        )
        control_db_spl = torch.maximum(
            control_db_spl,
            torch.ones(control_db_spl.size(), device=self.device) * threshold_low,
        )

        # Compute compression gain in dB
        gain = -attnenuated_ohc - (control_db_spl - threshold_low) * (
            1 - (1 / compression_ratio)
        )

        # Convert the signal envelope to dB SPL
        control_db_spl = torch.maximum(
            reference, torch.ones(control.size(), device=self.device) * self.small
        )
        control_db_spl = level1 + 20 * torch.log10(control_db_spl)
        control_db_spl = torch.maximum(
            control_db_spl, torch.zeros(control_db_spl.size(), device=self.device)
        )
        reference_db = control_db_spl + gain - attnenuated_ihc
        reference_db = torch.maximum(
            reference_db, torch.zeros(reference_db.size(), device=self.device)
        )

        return reference_db

    def envelope_sl(
        self,
        reference: torch.tensor,
        basilar_membrane: torch.tensor,
        attenuated_ihc: torch.tensor,
        level1: float,
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Convert the compressed envelope returned by cochlear_envcomp to dB SL.

        Arguments:
            reference (): linear envelope after compression
            basilar_membrane (): linear Basilar Membrane vibration after compression
            attenuated_ihc (): IHC attenuation at the input to the synapse
            level1 (): level in dB SPL corresponding to 1 RMS
            small (float): ???

        Returns:
            _reference (): reference envelope in dB SL
            _basilar_membrane (): Basilar Membrane vibration with envelope converted to
                dB SL

        Updates:
        James M. Kates, 20 Feb 07.
        IHC attenuation added 9 March 2007.
        Basilar membrane vibration conversion added 2 October 2012.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        # Convert the envelope to dB SL
        attenuated_ihc = attenuated_ihc.unsqueeze(1).repeat(1, reference.size(1))
        _reference = level1 - attenuated_ihc + 20 * torch.log10(reference + self.small)
        _reference = torch.maximum(
            _reference, torch.zeros(reference.size(), device=self.device)
        )

        # Convert the linear BM motion to have a dB SL envelope
        gain = (_reference + self.small) / (reference + self.small)
        _basilar_membrane = gain * basilar_membrane

        return _reference, _basilar_membrane

    def inner_hair_cell_adaptation(
        self, reference_db, reference_basilar_membrane, delta
    ):
        """
        Provide inner hair cell (IHC) adaptation. The adaptation is based on an
        equivalent RC circuit model, and the derivatives are mapped into
        1st-order backward differences. Rapid and short-term adaptation are
        provided. The input is the signal envelope in dB SL, with IHC attenuation
        already applied to the envelope. The outputs are the envelope in dB SL
        with adaptation providing overshoot of the long-term output level, and
        the BM motion is multiplied by a gain vs. time function that reproduces
        the adaptation. IHC attenuation and additive noise for the equivalent
        auditory threshold are provided by a subsequent call to eb_BMatten.

        Arguments:
            reference_db (np.ndarray): signal envelope in one frequency band in dB SL
                 contains OHC compression and IHC attenuation
            reference_basilar_membrane (): basilar membrane vibration with OHC
                compression but no IHC attenuation
            delta (): overshoot factor = delta x steady-state
            freq_sample (int): sampling rate in Hz

        Returns:
            output_db (): envelope in dB SL with IHC adaptation
            output_basilar_membrane (): Basilar Membrane multiplied by the IHC
                adaptation gain function

        Updates:
        James M. Kates, 1 October 2012.
        Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        # Test the amount of overshoot
        dsmall = 1.0001
        delta = max(delta, dsmall)

        # Initialize adaptation time constants
        tau1 = 2  # Rapid adaptation in msec
        tau2 = 60  # Short-term adaptation in msec
        tau1 = 0.001 * tau1  # Convert to seconds
        tau2 = 0.001 * tau2

        # Equivalent circuit parameters
        freq_sample_inverse = 1 / self.freq_sample
        r_1 = 1 / delta
        r_2 = 0.5 * (1 - r_1)
        r_3 = r_2
        c_1 = tau1 * (r_1 + r_2) / (r_1 * r_2)
        c_2 = tau2 / ((r_1 + r_2) * r_3)

        # Intermediate values used for the voltage update matrix inversion
        a11 = r_1 + r_2 + r_1 * r_2 * (c_1 / freq_sample_inverse)
        a12 = -r_1
        a21 = -r_3
        a22 = r_2 + r_3 + r_2 * r_3 * (c_2 / freq_sample_inverse)
        denom = 1 / ((a11 * a22) - (a21 * a12))

        # Additional intermediate values
        r_1_inv = 1 / r_1
        product_r1_r2_c1 = r_1 * r_2 * (c_1 / freq_sample_inverse)
        product_r2_r3_c2 = r_2 * r_3 * (c_2 / freq_sample_inverse)

        # Initialize the outputs and state of the equivalent circuit
        nsamp = len(reference_db)
        gain = torch.ones_like(
            reference_db
        )  # Gain vector to apply to the BM motion, default is 1
        output_db = torch.zeros_like(reference_db)
        v_1 = 0
        v_2 = 0
        small = 1e-30

        # Loop to process the envelope signal
        # The gain asymptote is 1 for an input envelope of 0 dB SPL
        for n in range(nsamp):
            v_0 = reference_db[n]
            b_1 = v_0 * r_2 + product_r1_r2_c1 * v_1
            b_2 = product_r2_r3_c2 * v_2
            v_1 = denom * (a22 * b_1 - a12 * b_2)
            v_2 = denom * (-a21 * b_1 + a11 * b_2)
            out = (v_0 - v_1) * r_1_inv
            output_db[n] = out

        output_db = torch.maximum(output_db, torch.zeros(1, device=self.device))
        gain = (output_db + small) / (reference_db + small)

        output_basilar_membrane = gain * reference_basilar_membrane

        return output_db, output_basilar_membrane

    def basilar_membrane_add_noise(
        self, reference: torch.tensor, threshold: int, level1: float
    ) -> torch.tensor:
        """
        Apply the IHC attenuation to the BM motion and to add a low-level Gaussian
        noise to give the auditory threshold.

        Arguments:
            reference (): BM motion to be attenuated
            threshold (): additive noise level in dB re:auditory threshold
            level1 (): an input having RMS=1 corresponds to Level1 dB SPL

        Returns:
            Attenuated signal with threshold noise added

        Updates:
            James M. Kates, 19 June 2012.
            Just additive noise, 2 Oct 2012.
            Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
        """
        gain = 10 ** ((threshold - level1) / 20)  # Linear gain for the noise

        # Gaussian RMS=1, then attenuated
        noise = gain * torch.randn(reference.size(), device=self.device)
        return reference + noise

    def group_delay_compensate(
        self,
        reference: torch.tensor,
        bandwidths: torch.tensor,
    ) -> torch.tensor:
        """
        Compensate for the group delay of the gammatone filter bank. The group
        delay is computed for each filter at its center frequency. The firing
        rate output of the IHC model is then adjusted so that all outputs have
        the same group delay.

        center_freq: torch.tensor,
        ear_q: float = 9.26449,
        min_bandwidth: float = 24.7,

        Arguments:
            bandwidths (): gammatone filter bandwidths adjusted for loss

        Returns:
            processed (): envelopes or BM motion compensated for the group delay.
        """
        # Processing parameters
        nchan = len(bandwidths)

        # # Filter ERB from Moore and Glasberg (1983)
        # erb = min_bandwidth + (center_freq / ear_q)
        #
        # # Initialize the gammatone filter coefficients
        # tpt = 2 * torch.pi / self.target_sample_rate
        # tpt_bandwidth = tpt * 1.019 * bandwidths * erb
        # a = torch.exp(torch.tensor(-tpt_bandwidth))
        # a_1 = 4.0 * a
        # a_2 = -6.0 * a * a
        # a_3 = 4.0 * a * a * a
        # a_4 = -a * a * a * a
        # a_5 = 4.0 * a * a
        #
        # # Compute the group delay in samples at fsamp for each filter
        # _group_delay = torch.zeros(nchan)
        # for n in range(nchan):
        #     _, _group_delay[n] = group_delay(
        #         ([1, a_1[n], a_5[n]], [1, -a_1[n], -a_2[n], -a_3[n], -a_4[n]]), 1
        #     )
        # _group_delay = torch.round(_group_delay).int()  # convert to integer samples
        #
        # # Compute the delay correlation
        # group_delay_min = torch.min(_group_delay)
        # _group_delay = (
        #     _group_delay - group_delay_min
        # )  # Remove the minimum delay from all the over values
        # group_delay_max = torch.max(_group_delay)
        # correct = (
        #     group_delay_max - _group_delay
        # )  # Samples delay needed to add to give alignment

        # Add delay correction to each frequency band
        processed = torch.zeros(reference.shape, device=self.device)
        for n in range(nchan):
            ref = reference[n]
            npts = len(ref)
            processed[n] = torch.concatenate(
                (
                    torch.zeros(CORRECT_DELAY_COEFS[n], device=self.device),
                    ref[: npts - CORRECT_DELAY_COEFS[n]],
                )
            )

        return processed
