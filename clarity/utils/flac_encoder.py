"""
Class for encoding and decoding audio signals
    using flac compression.
"""
from __future__ import annotations

import logging
import tempfile

# pylint: disable=import-error, protected-access
from pathlib import Path

import numpy as np
import pyflac as pf
import soundfile as sf

logger = logging.getLogger(__name__)


class WavEncoder(pf.encoder._Encoder):
    """
    Class offers an adaptation of the pyflac.encoder.FileEncoder
    to work directly with WAV signals as input.

    """

    def __init__(
        self,
        signal: np.ndarray,
        sample_rate: int,
        output_file: str | Path | None = None,
        compression_level: int = 5,
        blocksize: int = 0,
        streamable_subset: bool = True,
        verify: bool = False,
    ) -> None:
        """
        Initialise the encoder.

        Args:
            signal (np.ndarray): The raw audio data to be encoded.
            sample_rate (int): The sample rate of the audio data.
            output_file (str | Path | None): Path to the output FLAC file,
                a temporary file will be created if unspecified.
            compression_level (int): The compression level parameter that
                varies from 0 (fastest) to 8 (slowest). The default setting
                is 5, see https://en.wikipedia.org/wiki/FLAC for more details.
            blocksize (int): The size of the block to be returned in the
                callback. The default is 0 which allows libFLAC to determine
                the best block size.
            streamable_subset (bool): Whether to use the streamable subset for encoding.
                If true the encoder will check settings for compatibility. If false, the
                settings may take advantage of the full range that the format allows.
            verify (bool): If `True`, the encoder will verify it's own
                encoded output by feeding it through an internal decoder and
                comparing the original signal against the decoded signal.
                If a mismatch occurs, the `process` method will raise a
                `EncoderProcessException`.  Note that this will slow the
                encoding process by the extra time required for decoding and comparison.
        """
        super().__init__()

        self.__raw_audio = signal
        self._sample_rate = sample_rate

        if output_file:
            self.__output_file = (
                Path(output_file) if isinstance(output_file, str) else output_file
            )
        else:
            with tempfile.NamedTemporaryFile(suffix=".flac") as ofile:
                self.__output_file = Path(ofile.name)

        self._blocksize = blocksize
        self._compression_level = compression_level
        self._streamable_subset = streamable_subset
        self._verify = verify
        self._initialised = False

    def _init(self):
        """
        Initialise the encoder to write to a file.

        Raises:
            EncoderInitException: if initialisation fails.
        """
        c_output_filename = pf.encoder._ffi.new(
            "char[]", str(self.__output_file).encode("utf-8")
        )
        rc = pf.encoder._lib.FLAC__stream_encoder_init_file(
            self._encoder,
            c_output_filename,
            pf.encoder._lib._progress_callback,
            self._encoder_handle,
        )
        pf.encoder._ffi.release(c_output_filename)
        if rc != pf.encoder._lib.FLAC__STREAM_ENCODER_INIT_STATUS_OK:
            raise pf.EncoderInitException(rc)

        self._initialised = True

    def process(self) -> bytes:
        """
        Process the audio data from the WAV file.

        Returns:
            (bytes): The FLAC encoded bytes.

        Raises:
            EncoderProcessException: if an error occurs when processing the samples
        """
        super().process(self.__raw_audio)
        self.finish()
        with open(self.__output_file, "rb") as f:
            return f.read()


class FileDecoder(pf.decoder.FileDecoder):
    def process(self) -> tuple[np.ndarray, int]:
        """
        Overwritten version of the process method from the pyflac decoder.
        Original process returns stereo signals in float64 format.

        In this version, the data is returned using the original number
        of channels and in in16 format.

        Returns:
            (tuple): A tuple of the decoded numpy audio array, and the sample rate
                of the audio data.

        Raises:
            DecoderProcessException: if any fatal read, write, or memory allocation
                error occurred (meaning decoding must stop)
        """
        result = pf.decoder._lib.FLAC__stream_decoder_process_until_end_of_stream(
            self._decoder
        )
        if self.state != pf.decoder.DecoderState.END_OF_STREAM and not result:
            raise pf.DecoderProcessException(str(self.state))

        self.finish()
        self.__output.close()
        return sf.read(str(self.__output_file), always_2d=False, dtype="int16")


class FlacEncoder:
    """
    Class for encoding and decoding audio signals using FLAC

    It uses the pyflac library to encode and decode the audio data.
    And offers convenient methods for encoding and decoding audio data.
    """

    def __init__(self, compression_level: int = 5) -> None:
        """
        Initialise the compressor.

        Args:
            compression_level (int): The compression level parameter that
                varies from 0 (fastest) to 8 (slowest). The default setting
                is 5, see https://en.wikipedia.org/wiki/FLAC for more details.
        """
        self.compression_level = compression_level

    def encode(
        self,
        signal: np.ndarray,
        sample_rate: int,
        output_file: str | Path | None = None,
    ) -> bytes:
        """
        Method to encode the audio data using FLAC compressor.

        It creates a WavEncoder object and uses it to encode the audio data.

        Args:
            signal (np.ndarray): The raw audio data to be compressed.
            sample_rate (int): The sample rate of the audio data.
            output_file (str | Path): Path to where to
                save the output FLAC file. If not specified, a temporary file
                will be created.

        Returns:
            (bytes): The FLAC encoded audio signal.

        Raises:
            ValueError: If the audio signal is not in `np.int16` format.
        """
        if signal.dtype != np.int16:
            logger.error(
                f"FLAC encoder only supports 16-bit integer signals, "
                f"but got {signal.dtype}"
            )
            raise ValueError(
                f"FLAC encoder only supports 16-bit integer signals, "
                f"but got {signal.dtype}"
            )

        wav_encoder = WavEncoder(
            signal=signal,
            sample_rate=sample_rate,
            compression_level=self.compression_level,
            output_file=output_file,
        )
        return wav_encoder.process()

    @staticmethod
    def decode(input_filename: Path | str) -> tuple[np.ndarray, float]:
        """
        Method to decode a flac file to wav audio data.

        It uses the pyflac library to decode the flac file.

        Args:
            input_filename (pathlib.Path | str): Path to the input FLAC file.

        Returns:
            (np.ndarray): The raw audio data.

        Raises:
            FileNotFoundError: If the flac file to decode does not exist.
        """
        input_filename = (
            Path(input_filename) if isinstance(input_filename, str) else input_filename
        )

        if not input_filename.exists():
            logger.error(f"File {input_filename} not found.")
            raise FileNotFoundError(f"File {input_filename} not found.")

        decoder = FileDecoder(input_filename)
        signal, sample_rate = decoder.process()

        return signal, float(sample_rate)


def read_flac_signal(filename: Path) -> tuple[np.ndarray, float]:
    """Read a FLAC signal and return it as a numpy array

    Args:
        filename (Path): The path to the FLAC file to read.

    Returns:
        signal (np.ndarray): The decoded signal.
        sample_rate (float): The sample rate of the signal.
    """
    # Create encoder object
    flac_encoder = FlacEncoder()

    # Decode FLAC file
    signal, sample_rate = flac_encoder.decode(
        filename,
    )
    signal = (signal / 32768.0).astype(np.float32)

    # Load scale factor
    if filename.with_suffix(".txt").exists():
        with open(filename.with_suffix(".txt"), encoding="utf-8") as fp:
            max_value = float(fp.read())
            # Scale signal
            signal *= max_value
    return signal, sample_rate
