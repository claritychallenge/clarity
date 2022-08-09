import logging
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import soundfile
from jinja2 import Environment, FileSystemLoader
from soundfile import SoundFile

from clarity.enhancer.gha.gha_utils import format_gaintable, get_gaintable


class GHAHearingAid:
    def __init__(
        self,
        fs=44100,
        ahr=20,
        audf=None,
        cfg_file="prerelease_combination4_smooth",
        noisegatelevels=None,
        noisegateslope=0,
        cr_level=0,
        max_output_level=100,
        equiv0dBSPL=100,
        test_nbits=16,
    ):

        if audf is None:
            audf = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
        if noisegatelevels is None:
            noisegatelevels = [38, 38, 36, 37, 32, 26, 23, 22, 8]

        self.fs = fs
        self.ahr = ahr
        self.audf = audf
        self.cfg_file = cfg_file
        self.noisegatelevels = noisegatelevels
        self.noisegateslope = noisegateslope
        self.cr_level = cr_level
        self.max_output_level = max_output_level
        self.equiv0dBSPL = equiv0dBSPL
        self.test_nbits = test_nbits

    def create_configured_cfgfile(
        self, input_file, output_file, formatted_sGt, cfg_template_file
    ):
        """Using Jinja2, generates cfg file for given configuration.

        Creates template output file and configures with correct filenames, peak level
        out and DC gaintable.

        Args:
            input_file (str): file to process
            output_file (str): file in which to store processed file
            formatted_sGt (ndarray): gaintable formatted for input into cfg file
            cfg_template_file: configuration file template
            ahr (int): amplification headroom

        Returns:
            cfg_filename (str): cfg filename
        """

        if self.fs != 44100:
            logging.error("Current GHA configuration requires 44.1kHz sampling rate.")

        cfg_template_file = pathlib.Path(cfg_template_file)

        # Define cfg filenames
        # Read new file and replace any parameter values necessary
        # Update peaklevel out by adding headroom
        logging.info("Adding %s dB headroom", self.ahr)

        peaklevel_in = int(self.equiv0dBSPL)
        peaklevel_out = int(self.equiv0dBSPL + self.ahr)

        # Render jinja2 template
        file_loader = FileSystemLoader(cfg_template_file.parent)
        env = Environment(loader=file_loader)
        template = env.get_template(cfg_template_file.name)
        output = template.render(
            io_in=input_file,
            io_out=output_file,
            peaklevel_in=f"[{peaklevel_in} {peaklevel_in} {peaklevel_in} {peaklevel_in}]",
            peaklevel_out=f"[{peaklevel_out} {peaklevel_out}]",
            gtdata=formatted_sGt,
        )

        return output

    def process_files(self, infile_names, outfile_name, audiogram, listener=None):
        """Process a set of input signals and generate an output.

        Args:
            infile_names (list[str]): List of input wav files. One stereo wav
                file for each hearing device channel
            outfile_name (str): File in which to store output wav files
            dry_run (bool): perform dry run only
        """
        logging.info("Processing %s with listener %s", outfile_name, listener)

        logging.info("Audiogram severity is %s", audiogram.severity)
        audiogram = audiogram.select_subset_of_cfs(self.audf)

        # Get gain table with noisegate correction
        gaintable = get_gaintable(
            audiogram,
            self.noisegatelevels,
            self.noisegateslope,
            self.cr_level,
            self.max_output_level,
        )
        formatted_sGt = format_gaintable(gaintable, noisegate_corr=True)

        cfg_template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cfg_files",
            f"{self.cfg_file}_template.cfg",
        )

        # Merge CH1 and CH3 files. This is the baseline configuration.
        # CH2 is ignored.
        fd_merged, merged_filename = tempfile.mkstemp(
            prefix="clarity-merged-", suffix=".wav"
        )
        # Only need file name; must immediately close the unused file handle.
        os.close(fd_merged)

        self.create_HA_inputs(infile_names, merged_filename)

        # Create the openMHA config file from the template
        fd_cfg, cfg_filename = tempfile.mkstemp(
            prefix="clarity-openmha-", suffix=".cfg"
        )
        # Again, only need file name; must immediately close the unused file handle.
        os.close(fd_cfg)

        with open(cfg_filename, "w", encoding="utf-8") as f:
            f.write(
                self.create_configured_cfgfile(
                    merged_filename, outfile_name, formatted_sGt, cfg_template
                )
            )

        # Process file using configured cfg file
        # Suppressing OpenMHA output with -q - comment out when testing
        # Append log of OpenMHA commands to /cfg_files/logfile
        subprocess.run(
            [
                "mha",
                "-q",
                "--log=logfile.txt",
                f"?read:{cfg_filename}",
                "cmd=start",
                "cmd=stop",
                "cmd=quit",
            ],
            check=True,
        )

        # Delete temporary files.
        os.remove(merged_filename)
        os.remove(cfg_filename)

        # Check output signal has energy in every channel
        sig = self.read_signal(outfile_name)

        if len(np.shape(sig)) == 1:
            sig = np.expand_dims(sig, axis=1)

        if not np.all(np.sum(abs(sig), axis=0)):
            raise ValueError("Channel empty.")

        self.write_signal(outfile_name, sig, floating_point=True)

        logging.info("OpenMHA processing complete")

    def read_signal(
        self, filename, offset=0, nsamples=-1, nchannels=0, offset_is_samples=False
    ):
        """Read a wavefile and return as numpy array of floats.

        Args:
            filename (string): Name of file to read
            offset (int, optional): Offset in samples or seconds (from start). Defaults to 0.
            nchannels: expected number of channel (default: 0 = any number OK)
            offset_is_samples (bool): measurement units for offset (default: False)
        Returns:
            ndarray: audio signal
        """
        try:
            wave_file = SoundFile(filename)
        except Exception as e:
            # Ensure incorrect error (24 bit) is not generated
            raise Exception(f"Unable to read {filename}.") from e

        if nchannels not in (0, wave_file.channels):
            raise Exception(
                f"Wav file ({filename}) was expected to have {nchannels} channels."
            )

        if wave_file.samplerate != self.fs:
            raise Exception(f"Sampling rate is not {self.fs} for filename {filename}.")

        if not offset_is_samples:  # Default behaviour
            offset = int(offset * wave_file.samplerate)

        if offset != 0:
            wave_file.seek(offset)

        x = wave_file.read(frames=nsamples)

        return x

    def write_signal(self, filename, x, floating_point=True):
        """Write a signal as fixed or floating point wav file."""

        if floating_point is False:
            if self.test_nbits == 16:
                subtype = "PCM_16"
                # If signal is float and we want int16
                x *= 32768
                x = x.astype(np.dtype("int16"))
                assert np.max(x) <= 32767 and np.min(x) >= -32768
            elif self.test_nbits == 24:
                subtype = "PCM_24"
            else:
                raise ValueError("test_nbits must be 16 or 24")
        else:
            subtype = "FLOAT"

        soundfile.write(filename, x, self.fs, subtype=subtype)

    def create_HA_inputs(self, infile_names, merged_filename):
        """Create input signal for baseline hearing aids."""

        if (infile_names[0][-5] != "1") or (infile_names[2][-5] != "3"):
            raise Exception("HA-input signal error: channel mismatch!")

        signal_CH1 = self.read_signal(infile_names[0])
        signal_CH3 = self.read_signal(infile_names[2])

        merged_signal = np.zeros((len(signal_CH1), 4))
        # channel index 0 = front microphone on the left hearing aid
        merged_signal[:, 0] = signal_CH1[:, 0]
        # channel index 1 = front microphone on the right hearing aid
        merged_signal[:, 1] = signal_CH1[:, 1]
        # channel index 2 = rear microphone on the left hearing aid
        merged_signal[:, 2] = signal_CH3[:, 0]
        # channel index 3 = rear microphone on the right hearing aid
        merged_signal[:, 3] = signal_CH3[:, 1]

        self.write_signal(merged_filename, merged_signal, floating_point=True)
